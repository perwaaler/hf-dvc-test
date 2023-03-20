import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from utils.prepare_data import prepare_t7data
from utils.general import get_subset_idx, compare, extract_class_using_criteria, get_unique_elements
from utils.backend import PlotPar, TrainingPar
from utils.z_modules import get_auc_and_plot_roc


def get_cv_test_scores(
        cv_scores_df,
        target_columns,
        class_criteria=None,
        invert_scores=False,
        axs_plot=None,
):
    """Returns a list of AUC values for the validation set of each CV-fold.
    cv_scores_df is a dataframe with scores for each audio in each positoion
    that is expected to contain a columns tracking CV-fold index and participant
    id respectively."""
    cv_fold_indices = cv_scores_df["cv_idx"].unique()
    auc_cv = []
    for i in cv_fold_indices:
        position = np.unravel_index(i, shape=[2, 4])
        if axs_plot is None:
            ax_plot=None
        else:
            ax_plot = axs_plot[position]
        auc_cv.append(
            get_test_scores(
                scores_df=cv_scores_df.query("cv_idx==@i"),
                target_columns=get_unique_elements(target_columns),
                class_criteria=class_criteria,
                invert_scores=invert_scores,
                ax_plot=ax_plot)
        )


    return auc_cv


def get_test_scores(
    scores_df,
    target_columns,
    class_criteria=None,
    invert_scores=False,
    ax_plot=None,
):
    """Compute the ROC AUC for the scores stored in a dataframe. target_columns
    is a list of names with the target columns that the scores predicts. A class
    threshold can be provided if the target class is formed by dichotomizing a
    numerical variable (x >= class_thresold)."""

    t7data_full = prepare_t7data()
    # Get index of rows from IDs
    id_validation = scores_df["id"]
    idx_rows = get_subset_idx(t7data_full.id.values, id_validation)
    # Extract scores
    scores = scores_df.drop(columns=["id", "cv_idx"], errors="ignore")

    if len(target_columns) > 1:
        # More than one target (e.g. murmur pos. i): stack scores and targets
        scores = pd.melt(scores)["value"].values
        target = pd.melt(t7data_full.loc[idx_rows, target_columns])["value"].values
    else:
        # Target is assumed position invariant: aggragate scores over positions
        scores = get_aggragate_scores(scores)
        target = t7data_full.loc[idx_rows, target_columns].values

    if class_criteria is not None:
        target = extract_class_using_criteria(target, class_criteria)
    if invert_scores:
        scores = -scores    

    auc_roc = get_auc_and_plot_roc(
        binary_target=target,
        scores=scores,
        ax_plot=ax_plot)

    return auc_roc


def get_aggragate_scores(scores_df, aggregation_method="sum"):
    if aggregation_method == "sum":
        scores_agg = scores_df[[0, 1, 2, 3]].sum(axis=1).values
    else:
        scores_agg = scores_df[[0, 1, 2, 3]].max(axis=1).values
    return scores_agg