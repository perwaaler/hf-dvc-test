"""Script for computing metrics using the validation scores."""

import logging
import numpy as np
import matplotlib.pyplot as plt

from utils.z_modules import dump_dict_to_yaml
from utils.metrics import get_auc, mean
from utils.backend import OUTPUT_DIR, load_python_object, dump_python_object_to_file
from utils.general import get_subset_idx
from utils.prepare_data import prepare_t7data
from utils.testing_model import get_test_scores, get_cv_test_scores, get_aggragate_scores



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s',
)

LOGGER = logging.getLogger(__name__)


def get_validation_auc(
        target_column,
        cv_scores_df,
        result_relative_filepath=None,
        class_criteria=">= 1",
        invert_scores=False,
        axs_plot=None,
        axs_plot_combined=None,
        figure_filename=None,
):
    """Computes the AUC using validation scores obtained during cross
    validation. Outputs are written to the outputs/metrics folder."""
    if axs_plot is None:
        axs_plot = plt.subplots(2, 4)[-1]
    auc_validation = get_cv_test_scores(
        cv_scores_df=cv_scores_df,
        target_columns=target_column,
        class_criteria=class_criteria,
        invert_scores=invert_scores,
        axs_plot=axs_plot)
    auc_range = max(auc_validation) - min(auc_validation)
    auc_median = np.median(auc_validation)
    auc_avg, lower, upper = mean(auc_validation, return_ci=True)
    plt.suptitle(f"AUC: {auc_avg:.3} ± ({lower:.3}-{upper:.3})\n")
    if figure_filename is not None:
        plt.savefig(OUTPUT_DIR + "/figures/roc/" + figure_filename)


    # Get AUC for validation scores combined across all folds
    if axs_plot_combined is None:
        axs_plot_combined = plt.subplots(1, 1)[-1]
    auc_combined = get_test_scores(
        scores_df=cv_scores_df,
        class_criteria=class_criteria,
        invert_scores=invert_scores,
        target_columns=target_column,
        ax_plot=axs_plot_combined)
    lower_combined = lower + (auc_combined-auc_avg)
    upper_combined = upper + (auc_combined-auc_avg)
    if figure_filename is not None and axs_plot_combined is not None:
        plt.title(f"AUC combined: {auc_combined:.3} ± ({lower_combined:.3}-{upper_combined:.3})\n"
                  f"AUC mean:{auc_avg:.3} ± ({lower:.3}-{upper:.3})")
        plt.savefig(OUTPUT_DIR + "/figures/roc/combined/" + figure_filename)
    auc_info = {"range": auc_range,
                "median": auc_median,
                "mean": f"{auc_avg:.3} ± ({lower:.3}-{upper:.3})",
                "combined": f"{auc_combined:.3} ± ({lower_combined:.3}-{upper_combined:.3}"}

    if result_relative_filepath is not None:
        dump_dict_to_yaml(
            path_relative=f"metrics/{figure_filename}.yaml",
            dump_dict=auc_info)
        
        dump_python_object_to_file(
            variable=auc_validation,
            file_path_relative=result_relative_filepath)

    LOGGER.info(f"\nAverage AUC: {auc_avg:.3} ± ({lower:.2}-{upper:.2})\n"
                f"Median AUC: {auc_median:.3}\n"
                f"Range: {auc_range:.2}\n"
                f"AUC all scores: {auc_combined:.3}")

    return auc_validation


# if __name__ == "__main__":
#     # Get audio filename from command line
#     auc_validation = get_validation_auc()
