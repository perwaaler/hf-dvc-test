"""Script for obtaining validation scores from cross validation."""

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.z_modules import plot_cv_training_histories
from utils.general import save_model
from utils.backend import PlotPar, dump_python_object_to_file
from utils.backend import load_python_object
from utils.backend import TrainingPar
from utils.prepare_data import prepare_t7data
from utils.general import save_model
from utils.partition_data import get_stratified_cross_val_splits
from utils.training import train_rnn, plot_training_history
from utils.testing_model import get_cv_test_scores, get_test_scores
from utils.metrics import mean


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s',
)
LOGGER = logging.getLogger(__name__)


def run_cross_validation(
        index_subset=None,
        target_columns: list=TrainingPar["target"],
        stratification_columns=None,
        use_regression=False,
        class_criteria=">= 1",
        n_folds=TrainingPar["n_folds"],
        n_epochs=TrainingPar["n_epochs"],
        file_names: dict=None,
        get_scores_only=False,
):
    """Runs cross validation network training and validation. Returns a
    dataframe with validation set validation scores (murmur-grade predictions),
    which is dumped in the output/scores folder."""
    if file_names is None:
        file_names = {"model": "model",
                      "training_history": "training_history",
                      "validation_scores": "validation_scores",
                      "auc_validation": "auc_validation"}
    if index_subset is None:
        index_subset = np.arange(2124)
    if isinstance(target_columns, str):
        target_columns = [target_columns]*4
    elif len(target_columns) == 1:
        target_columns = [target_columns[0]]*4


    if TrainingPar["use_additional_features"]:
        rnn_input_file = "rnn_inputs_mod"
    else:
        rnn_input_file = "rnn_input_arrays"

    # ¤¤¤¤ Prepare Training Data ¤¤¤¤
    # Load data
    t7data = prepare_t7data(index_subset)

    rnn_inputs = load_python_object(f"rnn-input/{rnn_input_file}")
    rnn_inputs = rnn_inputs.loc[index_subset]
    # Get cross validation splits
    idx_not_nan = t7data[target_columns[0]].notnull().values
    rnn_inputs = rnn_inputs.loc[idx_not_nan]
    t7data = t7data.loc[idx_not_nan].reset_index()
    if stratification_columns is None:
        stratification_columns = target_columns[0]
        raise Warning("No stratification column set: using target column as default.")
    
    cv_splits = get_stratified_cross_val_splits(
        t7data=t7data,
        n_folds=n_folds,
        id_list=t7data["id"],
        stratification_columns=stratification_columns)

    # Create container that collects dataframes with audio scores
    scores_audio_valid = [None]*n_folds
    training_histories = [None]*n_folds
    for i in range(n_folds):
        LOGGER.info("CV iteration: %g", i)
        if get_scores_only:
            model_name = f"{file_names['model']}_{i}"
        else:
            model_name = None

        model, scores_audio_valid[i], history_i = train_rnn(
            t7data,
            rnn_inputs,
            idx_train=cv_splits[i]["idx_train"],
            idx_valid=cv_splits[i]["idx_valid"],
            n_epochs=n_epochs,
            target_columns=target_columns,
            use_regression=use_regression,
            class_criteria=class_criteria,
            model_name=model_name)

        training_histories[i] = history_i.history
        # Track identity of predictions by adding ID column to scores dataframe
        scores_audio_valid[i]["cv_idx"] = i
        # ¤¤ Save Model ¤¤
        save_model(model, f"cv/{file_names['model']}_{i}")

    # ¤¤ Collect Scores in Dataframe ¤¤
    scores_audio_valid = pd.concat(scores_audio_valid, axis=0)

    # ¤¤¤¤ Test Model ¤¤¤¤
    # Compute AUCs for each fold
    auc_validation = get_cv_test_scores(
        cv_scores_df=scores_audio_valid,
        target_columns=target_columns,
        class_criteria=class_criteria)

    auc_range = max(auc_validation) - min(auc_validation)
    auc_avg, lower, upper = mean(auc_validation, return_ci=True)
    auc_median = np.median(auc_validation)
    LOGGER.info(f"Average AUC: {auc_avg:.3} ± ({lower:.2}-{upper:.2})\n"
                f"Median AUC: {auc_median:.3}\n"
                f"Range: {auc_range:.2}")

    # Get AUC for validation scores combined across all folds
    ax_plot = plt.subplots(1, 1, figsize=PlotPar["sz_roc_combined"])[-1]
    auc_validation_combined = get_test_scores(
        scores_df=scores_audio_valid,
        class_criteria=class_criteria,
        target_columns=target_columns,
        ax_plot=ax_plot)
    ax_plot.set_title(f"AUC: {auc_validation_combined:.2}")
    plt.show()

    # ¤¤ Save Results ¤¤
    dump_python_object_to_file(training_histories,
                               f"training-history/cv/{file_names['training_history']}.py")
    dump_python_object_to_file(scores_audio_valid,
                               f"scores/cv/{file_names['validation_scores']}.py")
    dump_python_object_to_file(auc_validation,
                               f"metrics/cv/{file_names['auc_validation']}.py")

    LOGGER.info("CV finnished for target%s", target_columns[0])

    return scores_audio_valid, auc_validation

# if __name__ == "__main__":
#     # Get audio filename from command line
#     assigned_states_preds = run_cross_validation()
