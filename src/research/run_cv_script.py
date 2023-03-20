# %%
import path_setup
import logging

import numpy as np
import matplotlib.pyplot as plt

from utils.z_modules import plot_cv_training_histories, dump_dict_to_json
from utils.backend import TrainingPar, load_python_object
from cross_validation import run_cross_validation, prepare_t7data
from get_validation_metrics import get_validation_auc
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s',
)

# %% AVPG
scores_audio_valid_avpg, auc_validation_avpg = run_cross_validation(
    # n_folds=2,
    # n_epochs=0,
    # index_subset=np.arange(100),

    use_regression=True,
    target_columns="avpg_mean_scaled",
    stratification_columns=["AS_clinical"],
    class_criteria=">= 0.66666",
    file_names={"model": "AS_regress",
                "training_history": "AS_regress_test",
                "validation_scores": "AS_regress_test",
                "auc_validation": "AS_regress_test"},
)

# %% Ejection Fraction
scores_audio_valid_2, auc_validation_2 = run_cross_validation(
    # n_folds=2,
    # n_epochs=1,
    # index_subset=np.arange(50),

    use_regression=True,
    target_columns="ejection_fraction_scaled",
    class_criteria="< 1.6",
    stratification_columns=["reduced_LVEF"],
    file_names={"model": "LVEF_regress",
                "training_history": "LVEF_regress",
                "validation_scores": "LVEF_regress",
                "auc_validation": "LVEF_regress"},
)


# %% Diastolic Dysfunction
experiment_tag = "DD_regress"
scores_audio_valid_DD, auc_validation_DD = run_cross_validation(
    # n_folds=2,
    # n_epochs=1,
    # index_subset=np.arange(50),

    use_regression=True,
    target_columns="DD_score_ny",
    class_criteria=">= 3",
    stratification_columns=["DD_score2"],
    file_names={"model": experiment_tag,
                "training_history": experiment_tag,
                "validation_scores": experiment_tag,
                "auc_validation": experiment_tag},
)

plot_cv_training_histories(history_filepath=experiment_tag + ".py",
                           figure_filename=experiment_tag + ".png",
                           write_figure_to_file=False)
# %% Diastolic Dysfunction: 4 grades
experiment_tag = "DD_score_4lvl"
scores_audio_valid_DD, auc_validation_DD = run_cross_validation(
    # n_folds=2,
    # n_epochs=1,
    # index_subset=np.arange(50),
    use_regression=True,
    target_columns=experiment_tag,
    class_criteria=">= 2",
    stratification_columns=["DD_score2"],
    file_names={"model": experiment_tag,
                "training_history": experiment_tag,
                "validation_scores": experiment_tag,
                "auc_validation": experiment_tag},
)

plot_cv_training_histories(history_filepath=experiment_tag + ".py",
                           figure_filename=experiment_tag + ".png",
                           write_figure_to_file=False)

auc_cv = get_validation_auc(
    target_column=["DD_score_ny"],
    class_criteria=">= 3",
    cv_scores_df=scores_audio_valid_DD,
    figure_filename=f"cv_roc_{experiment_tag}",
    axs_plot=plt.subplots(2, 4)[-1],
)

# %% Diastolic Dysfunction: Regraded
experiment_tag = "DD_score_regraded"
scores_audio_valid_DD, auc_validation_DD = run_cross_validation(
    # n_folds=2,
    # n_epochs=1,
    # index_subset=np.arange(50),
    use_regression=True,
    target_columns=experiment_tag,
    class_criteria=">= 2",
    stratification_columns=["DD_score2"],
    file_names={"model": experiment_tag,
                "training_history": experiment_tag,
                "validation_scores": experiment_tag,
                "auc_validation": experiment_tag},
)
plot_cv_training_histories(history_filepath=experiment_tag + ".py",
                           figure_filename=experiment_tag + ".png",
                           write_figure_to_file=False)
auc_cv = get_validation_auc(
    target_column=["DD_score_ny"],
    class_criteria=">= 3",
    cv_scores_df=scores_audio_valid_DD,
    figure_filename=f"cv_roc_{experiment_tag}",
    axs_plot=plt.subplots(2, 4)[-1],
    axs_plot_combined=plt.subplots(1, 1)[-1],
)

# %% AS-grade training
experiment_tag = "AS_grade"
scores, auc = run_cross_validation(
    # n_folds=2,
    # n_epochs=1,
    # index_subset=np.arange(100),
    use_regression=True,
    target_columns=experiment_tag,
    class_criteria=">= 1",
    stratification_columns=["AS_clinical"],
    file_names={"model": experiment_tag,
                "training_history": experiment_tag,
                "validation_scores": experiment_tag,
                "auc_validation": experiment_tag},
)
plot_cv_training_histories(history_filepath=experiment_tag + ".py",
                           figure_filename=experiment_tag + ".png",
                           write_figure_to_file=False)
auc_cv = get_validation_auc(
    target_column=["AS_clinical"],
    class_criteria=">= 1",
    cv_scores_df=scores,
    figure_filename=f"cv_roc_{experiment_tag}",
    axs_plot=plt.subplots(2, 4)[-1],
    axs_plot_combined=plt.subplots(1, 1)[-1],
)
# %% AVPG mean regression training
experiment_tag = "avpg_mean_scaled"
scores, auc = run_cross_validation(
    # n_folds=2,
    # n_epochs=1,
    # index_subset=np.arange(100),
    use_regression=True,
    target_columns=experiment_tag,
    class_criteria=">= 1",
    stratification_columns=["AS_clinical"],
    file_names={"model": experiment_tag,
                "training_history": experiment_tag,
                "validation_scores": experiment_tag,
                "auc_validation": experiment_tag},
)
plot_cv_training_histories(history_filepath=experiment_tag + ".py",
                           figure_filename=experiment_tag + ".png",
                           write_figure_to_file=False)
auc_cv = get_validation_auc(
    target_column=["AS_clinical"],
    class_criteria=">= 1",
    cv_scores_df=scores,
    figure_filename=f"cv_roc_{experiment_tag}",
)


# ¤¤¤¤ MODEL EVALUATIONS ¤¤¤¤
# %% DD model evaluation
experiment_tag = "DD_score_regraded"
scores_df = load_python_object(f"scores/cv/{experiment_tag}.py")

auc_cv = get_validation_auc(
    target_column=["DD_score_ny"],
    class_criteria=">= 3",
    cv_scores_df=scores_df,
    figure_filename=f"cv_roc_{experiment_tag}",
)

# %% LVEF model evaluation
experiment_tag = "reduced_LVEF_mod"
scores_df = load_python_object(f"scores/cv/{experiment_tag}.py")

auc_cv = get_validation_auc(
    target_column=["ejection_fraction"],
    class_criteria="< 40",
    invert_scores=False,
    cv_scores_df=scores_df,
    figure_filename=f"cv_roc_{experiment_tag}",
)


# %% AS model evaluation
experiment_tag = "AS_mod"
scores_df = load_python_object(f"scores/cv/{experiment_tag}.py")

auc_cv = get_validation_auc(
    target_column=["AS_clinical"],
    class_criteria=">= 1",
    invert_scores=False,
    cv_scores_df=scores_df,
    figure_filename=f"cv_roc_{experiment_tag}",
)

# %%
plot_cv_training_histories(history_filepath=experiment_tag + ".py",
                           figure_filename=experiment_tag + ".png",
                           write_figure_to_file=True)