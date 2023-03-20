# %% ¤¤¤¤ Setup
import path_setup

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf

from utils.prepare_data import prepare_t7data
from utils.z_modules import summarize_column, get_auc_and_plot_roc
from utils.general import and_recursive, or_recursive, get_subset_idx
from utils.backend import OUTPUT_DIR, load_python_object 
from utils.testing_model import get_aggragate_scores

t7data = prepare_t7data()

# %% ¤¤¤¤ LVEF<40% prevalences
x_geq1 = []
x_geq2 = []
x_geq3 = []

for aa in range(4):
    x_geq1.append(t7data.query(f"murgrade{aa+1}>=1")["reduced_LVEF"].mean()*100)
for aa in range(4):
    x_geq2.append(t7data.query(f"murgrade{aa+1}>=2")["reduced_LVEF"].mean()*100)
for aa in range(4):
    x_geq3.append(t7data.query(f"murgrade{aa+1}>=3")["reduced_LVEF"].mean()*100)

# Prevalence whole dataset
x_geq1.append(t7data.query("murgrade_min>=1")
              ["reduced_LVEF"].mean()*100)
x_geq2.append(t7data.query("murgrade_min>=2")
              ["reduced_LVEF"].mean()*100)
x_geq3.append(t7data.query("murgrade_min>=3")
              ["reduced_LVEF"].mean()*100)

dataset_prevalence = (t7data["reduced_LVEF"]==True).mean()*100

df = pd.DataFrame({"MG>=1": np.round(x_geq1, 2),
                   "MG>=2": np.round(x_geq2, 2),
                   "MG>=3": np.round(x_geq3, 2)})
df.index = ["murmur p1",
            "murmur p2",
            "murmur p3", 
            "murmur p4",
            "murmur any"]

print("Prevalence reduced LVEF conditional on murmurs")
print(df)
print(f"\nDataset prevalence: {dataset_prevalence:.3}")
# Relationships with murmur grade seems to depend on position. Aortic murmur is
# inversely related to murmur


# %% ¤¤¤¤ Linear Regression: Ejection Fraction and risk factors
idx_not_nan = t7data[["reduced_LVEF"]].notna().values

t7data["x"] = or_recursive([t7data["angina"], 
                            t7data["diabetes"],
                            t7data["high_BP"]],
                           return_bool=False)
t7data["y"] = t7data["ejection_fraction"]**2
lm = smf.ols(
    'ejection_fraction ~ x + murgrade1 + age + bmi + np.power(bmi,2) + ' + 
    ' + sex + heart_attack + heartrate + dyspnea_any + smoking_now_or_before',
    data=t7data).fit()

fit_table = pd.DataFrame({"coefficients":lm.params.values,
                           "p-values": lm.pvalues.values}).round(3)
fit_table.index = lm.params.to_dict().keys()
print(fit_table.round(3))
print(f"R_squared: {lm.rsquared:.3}\n"
      f"Adjusted R_squared: {lm.rsquared_adj:.3}")
# Write table to CSV file
fit_table.to_csv(OUTPUT_DIR + "/tabular-modeling/LVEF_regression.csv")

# %% ROC-curve: predicting LVEF<40%
y_pred = lm.predict(t7data)
y_true = t7data.ejection_fraction<40
idx_not_na = y_pred.notna()
y_pred = y_pred[idx_not_na]
y_true = y_true[idx_not_na]

auc_fitted = get_auc_and_plot_roc(
    binary_target=y_true,
    scores=-y_pred)
plt.title(f"Risk factor model for reduced LVEF\n"
          f"AUC (LVEF<40%): {auc_fitted:.3}")
plt.savefig(OUTPUT_DIR + "/figures/roc/linear-models/LVEF_regression_risk_factors.png")

# %% ROC-curve: predicting Systolic Heart Failure
y_pred = lm.predict(t7data)
y_true = t7data["HFrEF_sympt"]
idx_not_na = np.logical_and(y_pred.notna(), y_true.notna())
y_pred = y_pred[idx_not_na]
y_true = y_true[idx_not_na]

auc_fitted = get_auc_and_plot_roc(
    binary_target=y_true,
    scores=-y_pred)
plt.title(f"Using Risk-factor model trained on LVEF\n"
          f"Predicting Systolic HF\n AUC:{auc_fitted:.3}")

plt.savefig(OUTPUT_DIR + "/figures/roc/linear-models/HFrEF_regression_risk_factors.png")

# %% ¤¤¤¤ Using RNN scores and risk factors to predict LVEF
# %% ¤¤ Get RNN-scores
scores_rnn = load_python_object("scores/reduced_LVEF.py")
idx_scores = get_subset_idx(t7data["id"], scores_rnn["id"])

scores = get_aggragate_scores(scores_rnn)
t7data["remaining_illnesses"] = or_recursive([t7data["angina"], 
                            t7data["diabetes"],
                            t7data["high_BP"]])
t7data["scores"] = [np.nan]*2124
t7data.loc[idx_scores, "scores"] = scores**2

# %% ¤¤ Fit regression model
lm = smf.ols(
    'ejection_fraction ~ scores + murgrade1 + age + bmi + np.power(bmi,2) + ' + 
    ' + sex + heart_attack + heartrate + dyspnea_any + ' +
    'remaining_illnesses + smoking_now_or_before',
    data=t7data).fit()

fit_table = pd.DataFrame({"coefficients":lm.params.values,
                           "p-values": lm.pvalues.values}
                         ).round(3)
fit_table.index = lm.params.to_dict().keys()

print(fit_table.round(3))
print(f"Adjusted R_squared: {lm.rsquared_adj:.3}")
print(f"Adjusted R_squared: {lm.rsquared:.3}")

y_pred = lm.predict(t7data)
print(f"N.o. predictions: {y_pred.notna().sum()}")
fit_table.to_csv(OUTPUT_DIR + "/tabular-modeling/LVEF_regression_RNNscores.csv")

# %% ¤¤ ROC-curve: predicting LVEF<40%
y_pred = lm.predict(t7data)
idx_pred = y_pred.notna()
y_true = t7data["reduced_LVEF"]
idx_not_na = and_recursive([y_pred.notna(), y_true.notna()], return_bool=True)
y_pred = y_pred[idx_not_na]
y_true = y_true[idx_not_na]

auc_fitted = get_auc_and_plot_roc(
    binary_target=y_true,
    scores=-y_pred)
plt.title(f"model: LVEF ~ Risk factors + RNN-scores \n"
          f"Predicting Systolic HF\n AUC:{auc_fitted:.3}")
plt.savefig(OUTPUT_DIR + "/figures/roc/linear-models/LVEF_regression_risk_factors_and_scores.png")

# %% ¤¤ ROC-curve: predicting Systolic Heart Failure
y_pred = lm.predict(t7data)
idx_pred = y_pred.notna()
y_true = t7data["HFrEF_sympt"]
idx_not_na = and_recursive([y_pred.notna(), y_true.notna()], return_bool=True)
y_pred = y_pred[idx_not_na]
y_true = y_true[idx_not_na]

auc_fitted = get_auc_and_plot_roc(
    binary_target=y_true,
    scores=1/y_pred)
plt.title(f"model: LVEF ~ Risk factors + RNN-scores \n"
          f"Predicting Systolic HF\n AUC:{auc_fitted:.3}")
plt.savefig(OUTPUT_DIR + "/figures/roc/linear-models/HFrEF_regression_risk_factors_and_scores.png")

# %% ¤¤¤¤ Predicting LV Hypertrophy ¤¤¤¤

