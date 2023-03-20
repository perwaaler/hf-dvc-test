"""Script for training and testing the LSTM network in a single train-validation
set split. Used to quickly test different methods and hyperparamters."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.backend import (TrainingPar, dump_python_object_to_file,
                           load_python_object)
from utils.general import save_model, get_subset_idx
from utils.metrics import get_auc
from utils.partition_data import get_train_test_split
from utils.prepare_data import prepare_t7data
from utils.training import train_rnn
from utils.testing_model import get_test_scores
from utils.z_modules import plot_training_history, summarize_column


def train_on_single_dataset(
    target_columns,
    index_subset=None,
    n_epochs=TrainingPar["n_epochs"],
    model=None,
):
    """Function for training RNN on a training set and testing on a single
    validation set."""

    target_columns=["mur1_presence", "mur2_presence", "mur3_presence", "mur4_presence"]
    model = None
    index_subset = np.arange(50)

    # Load data
    t7data = prepare_t7data(index_subset)
    rnn_inputs = load_python_object('rnn-input/rnn_input_arrays')
    rnn_inputs = rnn_inputs.loc[index_subset]

    # Find rows with complete data
    idx_not_missing = t7data[target_columns].notna().values
    t7data = t7data.loc[idx_not_missing].reset_index()
    if index_subset is not None:
        rnn_inputs = rnn_inputs.loc[idx_not_missing]

    # Get train and test data
    idx_train, idx_valid = get_train_test_split(
        t7data,
        test_size=0.2,
        stratification_columns=target_columns[0],
        seed=1,
    )
    
    model, scores_audio_valid, training_history = train_rnn(
        t7data,
        rnn_inputs,
        idx_train,
        idx_valid,
        target_columns=target_columns,
    )
    scores_audio_valid.loc[:, :3].max(axis=1)
    scores_collapsed = scores_audio_valid.loc[:, :3].sum(axis=1)
    scores_collapsed = pd.DataFrame({"scores":scores_collapsed,
                                     "id": scores_audio_valid.id.values})
    get_test_scores(scores_df=scores_collapsed,
                    target_columns=["HF_dyspne2"],
                    class_threshold=1)
    plt.plot(training_history.history["val_loss"])
    plt.ylim([0.3, 0.5])
    plot_training_history(training_history)
    ##################### MANUAL TRAINING ###############################
    # Fit model and get validation-set scores
    t7data_train, t7data_valid, rnn_inputs_train, rnn_inputs_valid = split_data(
            t7data,
            rnn_inputs,
            idx_train,
            idx_valid,
        )
    x_train, y_train, x_valid, y_valid, synch_dataframes = prepare_data_for_rnn(
        t7data_train,
        t7data_valid,
        rnn_inputs_train,
        rnn_inputs_valid,
        target_columns,
    )
    # Set model architecture
    if model is None:
        model = prepare_model(n_train=y_train.shape[0])
        
    # Fit model
    training_history = model.fit(
        x_train,
        y_train,
        epochs=20,
        batch_size=TrainingPar["batch_size"],
        verbose=True,
        validation_data=(x_valid, y_valid),
    )
    scores_train = model.predict(x_train)
    scores_valid = model.predict(x_valid)
    plot_training_history(training_history,
                          ["val_auc"])
    # tf.losses.binary_crossentropy(y_valid.numpy(), scores_valid.flatten())
    # tf.losses.binary_crossentropy(y_train.numpy(), scores_train.flatten())
    # tf.metrics.binary_accuracy(y_train.numpy(), scores_train.flatten())
    # tf.metrics.binary_accuracy(y_train.numpy(), scores_train.flatten())


    # tf.metrics.binary_accuracy(y_train.numpy(), scores_train.flatten())

    # Predict on each segment (take median across segment scores for each
    # audio-file) and collect audio scores in dataframe
    scores_audio_valid = get_audio_scores(
        model=model,
        inputs=x_valid,
        index_df=synch_dataframes["valid"],
    )
    # Track identity of predictions by adding ID column to scores dataframe
    scores_audio_valid["id"] = t7data.loc[idx_valid, "id"].values

    # Get performance metrics
    validation_data = t7data.iloc[
        get_subset_idx(t7data.id.values, scores_audio_valid.id)
    ][target_columns]
    scores = scores_audio_valid.iloc[:,:4]
    auc_validation = get_auc(
        validation_data.values.flatten(),
        scores.values.flatten(),
    )

    print(training_history.history)
    print(f"Validation set AUC: {auc_validation}")

    return model, scores_audio_valid, training_history