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
from utils.metrics import mean
from utils.testing_model import get_cv_test_scores, get_test_scores
from utils.partition_data import get_stratified_cross_val_splits
from utils.backend import dump_python_object_to_file, load_python_object
from utils.general import save_model
from utils.prepare_data import prepare_t7data


# ¤¤¤¤ Setup ¤¤¤¤
save_variables = True
target_columns = ["avpg_mean",
                  "avpg_mean",
                  "avpg_mean",
                  "avpg_mean"]
# classifiction = True
index_subset = np.arange(200)
# n_folds = TrainingPar["n_folds"]
# n_epochs = TrainingPar["n_epochs"]
n_folds = 8
n_epochs = 30
initial_learning_rate = 0.002
class_ratio = 0.5


# ¤¤¤¤ Prepare Training Data ¤¤¤¤
# Load data
t7data = prepare_t7data(index_subset)
rnn_inputs = load_python_object('rnn-input/rnn_input_arrays')
rnn_inputs = rnn_inputs.loc[index_subset]
# Get cross validation splits
idx_not_nan = t7data[target_columns[0]].notnull().values
rnn_inputs = rnn_inputs.loc[idx_not_nan]
t7data = t7data.loc[idx_not_nan].reset_index()
cv_splits = get_stratified_cross_val_splits(
    t7data=t7data,
    n_folds=n_folds,
    id_list=t7data.id,
    stratification_columns=target_columns[0])

# Create container that collects dataframes with audio scores
scores_audio_valid = [None]*n_folds
training_histories = [None]*n_folds
for i_fold in range(n_folds):
    print(f"#################################\n"
          f"##### CV iteration {i_fold} ############\n"
          f"#################################")

    # ¤¤ Modify Data to Format Expected by Network ¤¤
    # Split dataset using training and validation indices
    targets_train, targets_valid, rnn_inputs_train, rnn_inputs_valid = split_data(
        t7data,
        rnn_inputs,
        idx_train=cv_splits[i_fold]["idx_train"],
        idx_valid=cv_splits[i_fold]["idx_valid"],
        target_columns=target_columns)
    # Unpack data and store book-keeping variables for later repackaging
    x_train, y_train, x_valid, y_valid, synch_dataframes, n_inputs = prepare_data_for_rnn(
        targets_train,
        targets_valid,
        rnn_inputs_train,
        rnn_inputs_valid,
        class_ratio)

    # ¤¤ Training Settings ¤¤
    # Set model architecture
    model = prepare_model(
        n_inputs=y_train.shape[0],
        initial_learning_rate=initial_learning_rate)
    # Set early stoppage settings
    early_stopping = EarlyStopping(
        monitor=TrainingPar["monitor_metric"],
        patience=TrainingPar["patience"],
        start_from_epoch=TrainingPar["start_from_epoch"],
        verbose=True)

    # ¤¤ Train Model ¤¤
    training_histories[i_fold] = model.fit(
        x_train,
        y_train,
        epochs=n_epochs,
        batch_size=TrainingPar["batch_size"],
        verbose=True,
        validation_data=(x_valid, y_valid),
        callbacks=[early_stopping])
    if save_variables is True:
        save_model(model, f"rnn_cv_{i_fold}")

    # ¤¤ Plot Training History ¤¤
    plot_training_history(
        history=training_histories[i_fold],
        metric_to_plot="val_auc",
        ax_plot=None)
    plt.show()

    # ¤¤ Get Scores ¤¤
    scores_audio_valid[i_fold] = get_audio_scores(
        model=model,
        inputs=x_valid[:n_inputs["valid"]],
        index_df=synch_dataframes["valid"][:n_inputs["valid"]])
    # Track identity of predictions by adding ID column to scores dataframe
    scores_audio_valid[i_fold]["cv_idx"] = i_fold
    scores_audio_valid[i_fold]["id"] = t7data.loc[
        cv_splits[i_fold]["idx_valid"], "id"].values


# ¤¤ Collect Scores in Dataframe ¤¤
scores_audio_valid = pd.concat(scores_audio_valid, axis=0)
# Dump validation scores to file
if save_variables is True:
    dump_python_object_to_file(scores_audio_valid, "scores/validation_scores.py")


# ¤¤¤¤ Test Model ¤¤¤¤
auc_validation = get_cv_test_scores(
    cv_scores_df=scores_audio_valid,
    target_columns=target_columns[0])
auc_range = max(auc_validation) - min(auc_validation)
auc_avg, lower, upper = mean(auc_validation, return_ci=True)
auc_median = np.median(auc_validation)

print(f"Average AUC: {auc_avg:.3} ± ({lower:.2}-{upper:.2})\n"
      f"Median AUC: {auc_median:.3}\n"
      f"Range: {auc_range:.2}")


# ¤¤ Plot Traininig Histories ¤¤
_, ax = plt.subplots(8, 2)
for i in range(n_folds):
    plot_training_history(
        history=training_histories[i],
        metric_to_plot="val_auc",
        ax_plot=ax[i])
plt.show()


# ¤¤ Save Variables ¤¤
if save_variables is True:
    dump_python_object_to_file(
        auc_validation,
        file_path_relative="metrics/auc_cv.py")
    dump_python_object_to_file(
        training_histories,
        file_path_relative="training-history/training_histories.py")


auc_validation = load_python_object(
    "scores/validation_scores.py")
auc_combined = get_test_scores(
    scores_df=auc_validation,
    target_columns="DD_score2",
    ax_plot=plt)
plt.title(f"AUC: {auc_combined:.2}")


