"""Scripts used to train the RNN. """

import logging

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from sklearn.utils import resample
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

from utils.backend import TrainingPar, PlotPar
from utils.general import create_df_of_empty_lists, compare, extract_class_using_criteria, load_model
from utils.z_modules import plot_training_history


def train_rnn(
        t7data,
        rnn_inputs,
        idx_train,
        idx_valid,
        target_columns=None,
        use_regression=False,
        class_criteria=">= 1",
        n_epochs=TrainingPar["n_epochs"],
        initial_learning_rate=TrainingPar["init_learning_rate"],
        model_name=None,
        class_ratio=TrainingPar["class_ratio"],
        use_additional_features=TrainingPar["use_additional_features"],
):
    """Function that prepares data for network training, trains LSTM model
    on training data, and returns model and scores on validation set. If no
    target is provided it defaults to predicting murmur grade."""

    # Split dataset using training and validation indices
    targets_train, targets_valid, rnn_inputs_train, rnn_inputs_valid = split_data(
        t7data,
        rnn_inputs,
        idx_train,
        idx_valid,
        target_columns)

    # Sequnetilize data, and store book-keeping variables for reshaping data to
    # original form
    x_train, y_train, x_valid, y_valid, synch_dataframes, n_inputs = prepare_data_for_rnn(
        targets_train,
        targets_valid,
        rnn_inputs_train,
        rnn_inputs_valid,
        class_ratio,
        class_criteria)

    if model_name is None:
        model = prepare_model(
            n_inputs=y_train.shape[0],
            initial_learning_rate=initial_learning_rate,
            use_additional_features=use_additional_features,
            use_regression=use_regression)
    else:
        model = load_model(f"cv/{model_name}")

    # Set early stopping criteria
    early_stopping = EarlyStopping(
        monitor=TrainingPar["monitor_metric"],
        patience=TrainingPar["patience"],
        verbose=True)
    # if use_regression:
    #     y_valid = extract_class_using_criteria(y_valid, class_criteria)
    #     if class_criteria[0] == "<":
    #         y_valid = (y_valid==False)
    training_history = model.fit(
        x_train,
        y_train,
        epochs=n_epochs,
        batch_size=TrainingPar["batch_size"],
        verbose=TrainingPar["verbose"],
        validation_data=(x_valid, y_valid),
        callbacks=[early_stopping])

    plot_training_history(
        history=training_history,
        metric_to_plot=TrainingPar["metric_to_plot"],
        ax_plot=plt.subplots(1, 2, figsize=PlotPar["sz_history"])[-1])
    plt.show()

    # Get audio level scores by aggragating across positions
    scores_audio_valid = get_audio_scores(
        model=model,
        inputs=x_valid[:n_inputs["valid"]],
        index_df=synch_dataframes["valid"][:n_inputs["valid"]])
    # Track identity of predictions by adding ID column to scores dataframe
    scores_audio_valid["id"] = t7data.loc[idx_valid, "id"].values

    return model, scores_audio_valid, training_history


def split_data(
        t7data,
        rnn_inputs,
        idx_train,
        idx_valid,
        target_columns,
):
    """Takes full dataframes (before any rows have been removed, aside from in
    data cleanup) with tabular data and RNN input arrays, and extracts the rows
    that that corresponds to the training-validation split specified by the
    provided indices."""
    targets_train = t7data[target_columns].iloc[idx_train]
    targets_valid = t7data[target_columns].iloc[idx_valid]
    rnn_inputs_train = rnn_inputs.iloc[idx_train]
    rnn_inputs_valid = rnn_inputs.iloc[idx_valid]
    return targets_train, targets_valid, rnn_inputs_train, rnn_inputs_valid


def prepare_data_for_rnn(
        targets_train,
        targets_valid,
        rnn_inputs_train,
        rnn_inputs_valid,
        class_ratio=None,
        class_criteria=">= 1",
):
    """For training, the segmnents of input data must be turned from array to
    sequence format. This function takes training and validation dataframes
    (contains indices and dataframes for i:th cross-validation fold) and an
    array of MFCC matrices and reorganizes data by sequentializing and
    converting it to tensors. Returns dictionaries with keys 'train' and
    'valid'. Returns x and y, which contain input and output data in
    tensorformat, and synchronization dataframes which contains input and output
    data in sequential form along with original row and column indices (used to
    convert sequential data back to original form)."""

    # Create dictionary with book-keeping variables for reshaping data
    synch_dataframes = {"train": None, "valid": None}
    # Sequentilize MFCC arrays
    synch_dataframes["train"] = unpack_rnn_data(
        rnn_inputs=rnn_inputs_train,
        labels_df=targets_train)
    synch_dataframes["valid"] = unpack_rnn_data(
        rnn_inputs=rnn_inputs_valid,
        labels_df=targets_valid)
    # Save number of unique network inputs
    n_inputs = {"train": len(synch_dataframes["train"]),
                "valid": len(synch_dataframes["valid"])}

    # Resample to balance dataset
    synch_dataframes["train"] = resample_from_class(
        dataframe=synch_dataframes["train"],
        target_column="labels",
        class_ratio=class_ratio,
        class_criteria=class_criteria)
    if TrainingPar["balance_classes_valid"]:
        synch_dataframes["valid"] = resample_from_class(
            dataframe=synch_dataframes["valid"],
            target_column="labels",
            class_ratio=class_ratio,
            class_criteria=class_criteria)

    # Convert to tensors
    x_train, y_train = convert_data_to_tensors(synch_dataframes["train"])
    x_valid, y_valid = convert_data_to_tensors(synch_dataframes["valid"])
    # Transpose MFCC arrays (since python reads row-by-row instead of
    # column-by-column)
    x_train = np.moveaxis(x_train, source=1, destination=2)
    x_valid = np.moveaxis(x_valid, source=1, destination=2)

    return x_train, y_train, x_valid, y_valid, synch_dataframes, n_inputs


def resample_from_class(
        dataframe,
        target_column,
        class_ratio=None,
        seed=1,
        class_criteria=">= 1",
):
    """Resample from binary classes defined by x>=threshold. dataframe is a
    dataframe that contains training labels, and idx_list contains the indices
    of the rows of that dataframe wrt. the original dataframe (before
    train-test-split). Returns indices that corresponds to resampling from the
    target class untill the desired class ratio has been achieved."""
    if class_ratio is None:
        n_resample = 0
        return dataframe

    idx_pos_class = extract_class_using_criteria(
        array=dataframe[target_column],
        class_criteria=class_criteria).astype(bool).values
    n_positive = sum(idx_pos_class==True)
    n_negative = sum(idx_pos_class==False)
    n_resample = np.floor(class_ratio * n_negative) - n_positive
    if n_resample < 0:
        print("No need to resample; positive class sufficiently large")
        return dataframe

    dataframe_resampled = resample(
        dataframe[idx_pos_class],
        replace=True,
        n_samples=n_resample.astype(int),
        random_state=seed)

    dataframe_balanced = pd.concat([dataframe, dataframe_resampled], axis=0)

    return dataframe_balanced


def unpack_rnn_data(
        rnn_inputs,
        labels_df,
):
    """Takes a dataframe containing labels and a dataframe
    where each cell contains the MFCC arrays for the corresponding audio-file
    (2124 by 4 arrays) and returns the input and output in a combined dataframe
    where they are stored in sequential form. The returned dataframe also
    contains row and column indices for reshaping sequences back to the original
    array form."""
    # Sequentilize input and output data
    unpacked_mfccs = unpack_mfcc_dataframe(rnn_inputs)
    unpacked_labels = unpack_rnn_input_labels(
        labels_df=labels_df,
        index_df=unpacked_mfccs[["row_idx", "column_idx"]],
    )
    # Collect sequentilized data into a single dataframe
    sequentilized_data_df = pd.concat([unpacked_mfccs, unpacked_labels], axis=1)
    return sequentilized_data_df


def unpack_mfcc_dataframe(rnn_inputs):
    """Takes a dataframe (2124 by 4) where each cell contains the MFCC arrays of
    an audio file, and returns a dataframe with MFCC-array sequence. This
    dataframe also contains book-keeping variables (row and column indices) for
    later reshaping back to original array form."""
    # Number of people and auscultaiton indices to loop over
    n_id, n_ausc_ind = np.shape(rnn_inputs)

    mfccs = []
    row_idx = []
    column_idx = []
    # Sequentialize MFCC arrays
    for i_row in range(n_id):
        for i_col in range(n_ausc_ind):
            for _, mfcc_array in enumerate(rnn_inputs.iloc[i_row, i_col]):
                mfccs.append(mfcc_array)
                row_idx.append(i_row)
                column_idx.append(i_col)
    # Collect sequentialized data in dataframe
    unpacked_mfccs = pd.DataFrame({
        "mfcc": mfccs,
        "row_idx": row_idx,
        "column_idx": column_idx,
    })
    return unpacked_mfccs


def unpack_rnn_input_labels(labels_df, index_df):
    """Sequentilizes the labels so that they are synchronized with the MFCC
    arrays."""
    unpacked_labels = []
    n_data = len(index_df)
    for i in range(n_data):
        label = labels_df.iloc[index_df["row_idx"][i], index_df["column_idx"][i]]
        unpacked_labels.append(label)
    unpacked_labels = pd.DataFrame({"labels": unpacked_labels})
    return unpacked_labels


def convert_data_to_tensors(data_all_sequential):
    """Takes a dataframe that contains input and output data in sequential form
    and returns data in tensor form."""
    input_tensors = []
    for i in range(len(data_all_sequential)):
        input_tensors.append(data_all_sequential["mfcc"].iloc[i])
    input_tensors = tf.constant(input_tensors, dtype="float32")
    output_tensors = tf.constant(data_all_sequential["labels"], dtype="float32")
    return input_tensors, output_tensors


def prepare_model(
        n_inputs,
        initial_learning_rate=TrainingPar["init_learning_rate"],
        use_additional_features=TrainingPar["use_additional_features"],
        use_regression=False,
):
    """Prepares the model: builds architecture, sets optimization schedule, and
    compiles the model. MFCC arrays are transposed due to python reading arrays
    row by row."""
    par = TrainingPar
    if use_additional_features:
        input_size = list(reversed(par["mfcc_mod_size"]))
    else:
        input_size = list(reversed(par["mfcc_size"]))
    if use_regression:
        activation_final = tf.keras.activations.linear
    else:
        activation_final = par["activation"]

    # Model architecture
    model = tf.keras.models.Sequential()
    model.add(Input(input_size))
    model.add(LSTM(par["n_nodes_lstm_1"], return_sequences=True))
    model.add(LSTM(par["n_nodes_lstm_1"]))
    model.add(Dropout(par["dropout_percentage"]))
    model.add(Dense(par["n_nodes_fully_connected"], activation="relu"))
    model.add(Dense(par["n_nodes_final"]))

    # Set optimization schedule
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=par["n_epochs_per_lr_drop"] * n_inputs,
        decay_rate=par["decay_rate"],
        staircase=True,
    )

    # Compile model
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=lr_scheduler),
        loss=TrainingPar["loss"],
        metrics=TrainingPar["metric"],
    )

    return model


def get_audio_scores(model, inputs, index_df):
    """Predicts scores for each segment and returns scores for each audio by
    taking the median over its segment-wise scores. Output scores are returned
    in array format. df_indices is a dataframe containing row and column indices
    of the segments."""
    # Predict on segments
    scores_segments = model.predict(inputs).flatten()
    # Get audio scores by taking median over segment scores
    scores_audio = get_scores_for_each_audio(
        scores=scores_segments,
        index_df=index_df,
    )
    return scores_audio


def get_scores_for_each_audio(
        scores,
        index_df,
):
    """Takes a sequence of scores together with a dataframe containing row and
    column indices corresponding to the scores (in the orignial dataframe) and
    returns a dataframe (rows-->id, columns--> auscultation position) of scores,
    where each cell corresponds to an audio file. Audio scores are obtained from
    segment-level scores by taking the median across the segment-scores."""
    sequence_reshaped = reshape_sequence_to_array(
        sequence=scores,
        index_df=index_df,
    )
    sequence_reshaped = sequence_reshaped.applymap(np.median)
    return sequence_reshaped


def reshape_sequence_to_array(
        sequence,
        index_df,
):
    """Takes a sequence (containing scores or target values) and a dataframe
    with row and column indices of the elements of the sequence (in original
    dataframe), and returns sequence reshaped back to its original shape."""
    sequence = np.array(sequence)
    n_rows = np.max(index_df["row_idx"]) + 1
    n_cols = np.max(index_df["column_idx"]) + 1
    sequence_reshaped = create_df_of_empty_lists(n_rows, n_cols)

    for i, y_i in enumerate(sequence):
        row_idx = index_df["row_idx"][i]
        column_idx = index_df["column_idx"][i]
        sequence_reshaped.iloc[row_idx, column_idx].append(y_i)

    return sequence_reshaped
