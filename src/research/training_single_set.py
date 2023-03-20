import path_setup

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils.training import (
    TrainingPar,
    EarlyStopping,
    prepare_data_for_rnn,
    split_data,
    plot_training_history,
    prepare_model,
    get_audio_scores)
from utils.testing_model import get_test_scores
from utils.partition_data import get_train_test_split
from utils.backend import dump_python_object_to_file, load_python_object
from utils.prepare_data import prepare_t7data

# ¤¤¤¤ Setup ¤¤¤¤
use_additional_features = False
use_regression = False
save_variables = True
target_columns = ["AS_clinical",
                    "AS_clinical",
                    "AS_clinical",
                    "AS_clinical"]
stratification_columns = ["AS_clinical"]
index_subset = np.arange(1500)
class_ratio = 0.5
n_folds = TrainingPar["n_folds"]
initial_learning_rate = 0.002
# n_epochs = TrainingPar["n_epochs"]
n_epochs = 5



# ¤¤¤¤ Prepare Training Data ¤¤¤¤
# Load data
t7data = prepare_t7data(index_subset)
rnn_inputs_name = "rnn_inputs_mod" if use_additional_features else "rnn_input_arrays"
rnn_inputs = load_python_object(f"rnn-input/{rnn_inputs_name}")
rnn_inputs = rnn_inputs.loc[index_subset]
# Remove NaN values
idx_not_missing = t7data[target_columns[0]].notna().values
t7data = t7data.loc[idx_not_missing].reset_index()
if index_subset is not None:
    rnn_inputs = rnn_inputs.loc[idx_not_missing]
# Get train and test data indices
idx_train, idx_valid = get_train_test_split(
    t7data,
    test_size=TrainingPar["test_size"],
    stratification_columns=stratification_columns,
    seed=1)

# ¤¤ Modify Data to Format Expected by Network ¤¤
# Split dataset using training and validation indices
targets_train, targets_valid, rnn_inputs_train, rnn_inputs_valid = split_data(
    t7data,
    rnn_inputs,
    idx_train,
    idx_valid,
    target_columns)
targets_train/targets_train.max(axis=0)
# Unpack data and store book-keeping variables for later repackaging
x_train, y_train, x_valid, y_valid, synch_dataframes, n_inputs = prepare_data_for_rnn(
    targets_train,
    targets_valid,
    rnn_inputs_train,
    rnn_inputs_valid,
    class_ratio)

# ¤¤ Prepare and Train Model ¤¤
# Set model architecture
model = prepare_model(
    n_inputs=y_train.shape[0],
    use_additional_features=use_additional_features,
    initial_learning_rate=initial_learning_rate)
# Settings for early stoppage based on validation performance
early_stopping = EarlyStopping(
    monitor=TrainingPar["monitor_metric"],
    patience=TrainingPar["patience"],
    verbose=True)
# Fit model
training_history = model.fit(
    x_train,
    y_train,
    epochs=n_epochs,
    batch_size=TrainingPar["batch_size"],
    verbose=True,
    validation_data=(x_valid, y_valid),
    callbacks=[early_stopping])

# ¤¤ Plot Training History ¤¤
plot_training_history(training_history, metric_to_plot="val_auc")
plt.show()

# ¤¤ Get Scores ¤¤
scores_audio_valid = get_audio_scores(
    model=model,
    inputs=x_valid[:n_inputs["valid"]],
    index_df=synch_dataframes["valid"][:n_inputs["valid"]])
# Add ID column to scores dataframe to track the ID of the scores
scores_audio_valid["id"] = t7data.loc[idx_valid, "id"].values
# Save validation scores
if save_variables is True:
    dump_python_object_to_file(
        scores_audio_valid,
        "scores/validation_scores.py")


# ¤¤¤¤ Test Model ¤¤¤¤ 
# Get AUC
auc_validation = get_test_scores(
    scores_df=scores_audio_valid,
    target_columns=target_columns[0],
    ax_plot=plt)
plt.title(f"AUC: {auc_validation:.3}")

dump_python_object_to_file(auc_validation, "metrics/auc_validation")