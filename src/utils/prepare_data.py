"""Functions for loading and preparing the tabular data. Preparation includes
renaming variables, removing data rows with incomplete data (e.g. corrupt
audio). Also generates some new variables, such as mean murmur-grade (across
annotators) and max murmur-grade across auscultation positions."""

import datetime
import numpy as np
import pandas as pd

from utils.backend import read_tabular_data
from utils.general import get_subset_idx
from utils.general import or_recursive
from utils.general import and_recursive
from utils.general import compare
from utils.general import and_custom
from utils.general import compare
from utils.general import drop_rows_on_condition


ID_CORRUPT_AUDIO = [10791119, 15187830, 15216015, 15787937, 35205318]


def prepare_t7data(index_subset=None):
    """Prepares the tabular T7 data. Removes incomplete rows, reorderes
    auscultation data according to the date when the order of the positions was
    reversed, renames variables, and finally creates some new variables to be
    used for analysis."""
    t7data = read_tabular_data()
    t7data = find_annotated_rows(t7data)
    t7data = correct_auscultation_order(t7data)

    # Remove row with ID 10492521 (incomplete Echo data)
    t7data = drop_rows_on_condition(t7data.UNIKT_LOPENR == 10492521, t7data)

    t7data = remove_corrupt_audio(t7data)
    t7data = rename_and_define_variables(t7data)

    if index_subset is not None:
        return t7data.iloc[index_subset]

    return t7data


def find_annotated_rows(t7data):
    """Find rows that have been murmur-annotated."""
    t7data = t7data.dropna(subset="MURMUR_1NORMAD_T72").reset_index()
    return t7data


def correct_auscultation_order(t7data):
    """The indexing of the auscultation areas was changed during the
    collection of the data, which is corrected for here. At first, the positions
    (Aortic, Pulmonic, Tricuspid, Mitral) were ordered M, T, P, A, but were
    later reversed to A, P, T, M. The switch happened on Aug. 24 2015."""

    # Get index of rows with wrong order ([4, 3, 2, 1])
    switch_date = datetime.datetime(2015, 8, 24)
    rows_wrong_order = (t7data.ECHO_CHKIN_DATE_T72 < switch_date).values

    # List of column groups ("position_" refers to the position index) that require
    # reordering at specified rows
    column_groups_to_shuffle = [
        "MURMUR_position_NORM_REF_T72",
        "MURMUR_position_SYS_REF_T72",
        "MURMUR_position_DIA_REF_T72",
        "MURMUR_position_NOISE_REF_T72",
        "FIRST_SOUND_position_FAINTAD_T72",
        "FIRST_SOUND_position_INAUDIBLEAD_T72",
        "FIRST_SOUND_position_FAINTSA_T72",
        "FIRST_SOUND_position_INAUDIBLESA_T72",
        "MURMUR_position_SYSAD_T72",
        "MURMUR_position_SYSSA_T72",
        "MURMUR_position_GRADENRAD_T72",
        "MURMUR_position_GRADENRSA_T72",
    ]

    # Iterate over column groups
    for mur_var in column_groups_to_shuffle:
        # Get names of columns in column group
        column_names = [None]*4
        for i_pos in range(4):
            column_names[i_pos] = mur_var.replace("position_", str(i_pos+1))
        # Reshuffle column order in specified rows
        t7data = reverse_column_order(
            dataframe=t7data,
            rows=np.argwhere(rows_wrong_order).flatten(),
            columns=column_names,
        )

    return t7data


def reverse_column_order(dataframe, rows, columns):
    """Reverses the order of the columns at rows specified by indices in
    'rows'."""
    df_slice_with_columns_reversed = pd.DataFrame(
        dataframe.loc[rows][reversed(columns)].values,
        columns=columns,
    )
    df_slice_with_columns_reversed.index = pd.Index(rows)
    dataframe.loc[rows, columns] = df_slice_with_columns_reversed
    return dataframe


def remove_corrupt_audio(t7data):
    """Removes rows with corrupt audio files."""
    idx_corrupt_audio = get_subset_idx(t7data.UNIKT_LOPENR, ID_CORRUPT_AUDIO)
    t7data.drop(idx_corrupt_audio, inplace=True)
    t7data.reset_index(inplace=True, drop=True)
    return t7data


def rename_and_define_variables(t7data):
    """Rename most used variables for convenience."""
    # Grade 0 has been encoded as missing; set to grade 0
    t7data["ARGRADE_T72"].replace(np.nan, 0.0, inplace=True)
    t7data["MRGRADE_T72"].replace(np.nan, 0.0, inplace=True)
    t7data["ASGRADE_T72"].replace(np.nan, 0.0, inplace=True)
    t7data["MSGRADE_T72"].replace(np.nan, 0.0, inplace=True)

    t7data = create_heartfailure_variable(t7data)

    for i in range(1, 5):
        # ### Rename murmur variables ###
        # Murmur grades individual annotators
        t7data = t7data.rename(
            columns={
                f"MURMUR_{i}GRADENRAD_T72": f"murgrade{i}_sa",
                f"MURMUR_{i}GRADENRSA_T72": f"murgrade{i}_ad",
            }
        )
        t7data[f"murgrade{i}_sa"].replace(
            np.nan, 0.0, inplace=True
        )
        t7data[f"murgrade{i}_ad"].replace(
            np.nan, 0.0, inplace=True
        )
        # Agreed noise
        t7data = t7data.rename(
            columns={
                f"MURMUR_{i}NOISE_REF_T72": f"noise_agreed{i}",
            }
        )
        # ### Aggragate variables that summarizes murmur grades ###
        # Mean murmur grade
        t7data[f"murgrade{i}"] = t7data[[
            f"murgrade{i}_sa",
            f"murgrade{i}_ad"]].mean(axis=1)
        # Maximum murmur grade
        t7data[f"murgrade{i}_max"] = t7data[[
            f"murgrade{i}_sa",
            f"murgrade{i}_ad"]].max(axis=1)
        # Minimum murmur grade
        t7data[f"murgrade{i}_min"] = t7data[[
            f"murgrade{i}_sa",
            f"murgrade{i}_ad"]].max(axis=1)
        # Murmur presence (weak, systolic or diastolic) in position i
        t7data[f"mur{i}_presence"] = compare(t7data[f"murgrade{i}"], ">", 0.5)

    # Presence of murmur (grade>=1) in at least one position
    t7data["mur_presence"] = or_recursive([
        t7data["murgrade1"] >= 1,
        t7data["murgrade2"] >= 1,
        t7data["murgrade3"] >= 1,
        t7data["murgrade4"] >= 1,
    ])

    # Murmur grade aggragate values
    murgrade_columns = t7data[["murgrade1", "murgrade2",
                               "murgrade3", "murgrade4"]]
    t7data["murgrade_max"] = murgrade_columns.max(axis=1)
    t7data["murgrade_min"] = murgrade_columns.min(axis=1)
    t7data["murgrade_sum"] = murgrade_columns.sum(axis=1)

    # Rename clinical variables to more convenient names
    t7data = t7data.rename(
        columns={
            # General
            "UNIKT_LOPENR": "id",
            "AGE_T7": "age",
            "BMI_T7": "bmi",
            "SEX_T7": "sex",
            # Diseases and biometrics
            "DIABETES_T7": "diabetes",
            # Do you get breathless when ...
            "DYSPNEA_FAST_UPHILL_T7": "dyspnea_fast_uphill",  # ... walking rapidly up a moderate slope?
            "DYSPNEA_CALMLY_FLAT_T7": "dyspnea_flat",  # ... walking on a flat surface?
            "DYSPNOE_REST_T7": "dyspnea_rest",  # ... resting?
            "HIGH_BLOOD_PRESSURE_T7": "high_BP",  # Do you or have you had high blood pressure?
            "ANGINA_T7": "angina",  # Do you or have you had angina?
            "PO2_T72": "po2",  # Oxygen saturation
            "PULSESPIRO_T72": "heartrate",  # Heart rate prior to spirometry
            "CHEST_PAIN_NORMAL_T7": "chestpain_normal",  # Chest pain at normal pace flat surface?
            "CHEST_PAIN_FAST_T7": "chestpain_fast",
            # If you get chest pain while walking up uphill, do you: 1) stop, 2)
            # slow down, 3) carry on at same pace
            "CHEST_PAIN_ACTION_T7": "chestpain_action",
            "SMOKE_DAILY_Q2_T7": "smoke_daily",
            # Do you (lvl 2) or have you had (lvl 3) ...
            "HEART_ATTACK_T7": "heart_attack",  # ... a heart attack?
            "HEART_FAILURE_T7": "heart_failure",  # ... heart failure?
            # Echocardiogram
            "ARGRADE_T72": "AR_grade",
            "MRGRADE_T72": "MR_grade",
            "ASGRADE_T72": "AS_grade",
            "MSGRADE_T72": "MS_grade",
            "AVMEANPG_T72": "avpg_mean",
            "AVAVMAX_T72": "avarea_max",
            "LVEFBIPLANE_T72": "ejection_fraction",
        }
    )

    # Create clinical convenience variables
    # Dyspnea, currently or previously, walking on flat surface or at rest
    t7data["dyspnea_resting_or_walking"] = or_recursive(
        [t7data.dyspnea_flat, t7data.dyspnea_rest]
    )
    t7data["dyspnea_any"] = or_recursive(
        [t7data.dyspnea_flat, t7data.dyspnea_rest, t7data.dyspnea_fast_uphill]
    )
    # Angina or dyspnea (as defined above)
    t7data["angina_or_dyspnea"] = or_recursive(
        [t7data.dyspnea_resting_or_walking, t7data.angina]
    )
    # Chest pain, currently or previously, walking at normal pace on level
    # ground or during light activity (walking up stairs etc...)
    t7data["chest_pain_any"] = or_recursive([
        t7data.chestpain_normal,
        t7data.chestpain_fast,
    ])
    # Self reported high blood pressure, currently and previously
    t7data["high_BP_history"] = compare(
        t7data.high_BP, ">", 0)
    t7data["high_BP_current"] = compare(
        t7data.high_BP, ">", 0)

    # Define AS severity grade from the aortic valve pressure gradient (AVPG-mean)
    t7data = grade_aortic_stenosis(t7data)
    # Convention: clinical VHD is when grade>0 for stenosis and grade>2 for regurgitation
    t7data["AR_clinical"] = compare(t7data.AR_grade, ">=", 3)
    t7data["MR_clinical"] = compare(t7data.MR_grade, ">=", 3)
    t7data["AS_clinical"] = compare(t7data.AS_grade, ">=", 1)
    t7data["MS_clinical"] = compare(t7data.MS_grade, ">=", 1)

    t7data["VHD_clinical"] = or_recursive([t7data["AR_clinical"],
                                           t7data["MR_clinical"],
                                           t7data["AS_clinical"],
                                           t7data["MS_clinical"]])

    t7data["reduced_LVEF"] = compare(t7data["ejection_fraction"], "<", 40)

    t7data["avpg_mean_scaled"] = t7data["avpg_mean"] / 15
    t7data["avpg_mean_sqrt"] = t7data["avpg_mean"]**0.5 / 10**0.5
    t7data["ejection_fraction_div_25"] = t7data["ejection_fraction"]/25
    # Merge levels 0 and 1 into one level to produce DD-grades {0,1,2,3}
    t7data["DD_score_4lvl"] = recode_variable(
        t7data["DD_score_ny"],
        bin_edges=[0, 2, 3, 4, np.inf],
        bin_labels={"1": 0, "2": 1, "3": 2, "4": 3})
    t7data["DD_score_regraded"] = recode_variable(
        t7data["DD_score_ny"],
        bin_edges=[0, 1, 2, 3, 4, 5],
        bin_labels={0: 0,
                    1: 0.5,
                    2: 1,
                    3: 2,
                    4: 3}
    )
    return t7data


def create_heartfailure_variable(t7data):
    """Generates the heartfailure variable based on the syntaxt provided by
    Hasse in the 'Syntax hjertesvikt' document."""

    #### Diastolic Dysfunction ####
    esum = (t7data.TDIESEPT_T72 + t7data.TDIELAT_T72) / 2
    aver_E_e = t7data.MVEVMAX_T72 / esum
    Aver_Ee_over14 = recode_variable(
        aver_E_e,
        bin_edges=[0, 14, 100],
        bin_labels={0: 0, 1: 1},
    )
    height_m = t7data.HEIGHT_T7 / 100
    Velocity = or_recursive([
        compare(t7data.TDIESEPT_T72, "<", 0.07),
        compare(t7data.TDIELAT_T72, "<", 0.10),
    ])
    Velocity[
        np.logical_and(t7data.TDIESEPT_T72 >= 0.07, t7data.TDIELAT_T72 >= 0.10)
    ] = 0
    Tricusp_over28 = recode_variable(
        t7data["TRVMAX_T72"],
        bin_edges=[0, 2.8, np.inf],
        bin_labels={0: 0, 1: 1}
    ).astype(float)
    # Left atrial volume
    BSA = (0.20247 * (height_m ** 0.725)) * (t7data.WEIGHT_T7 ** 0.425)
    Lavi = t7data.LAESV_MOD_BP_T72/BSA
    LaLarge = compare(Lavi, ">", 34)
    # Diastolic dysfunction score
    t7data["DD_score_ny"] = Aver_Ee_over14 + Velocity + Tricusp_over28 + LaLarge
    # DD_score==1 --> persons with no DD. DD_score==2 --> probable DD.
    # DD_score>=3 --> definate DD.
    t7data["DD_score2"] = compare(t7data["DD_score_ny"], ">=", 3)

    #### Dyspnea ####
    t7data["HF_dyspne1"] = or_recursive([
        compare(t7data.DYSPNEA_FAST_UPHILL_T7, "==", 1),
        compare(t7data.M_MRC_T72, ">=", 1),
    ])
    t7data["HF_dyspne2"] = or_recursive([
        compare(t7data.DYSPNEA_CALMLY_FLAT_T7, "==", 1),
        compare(t7data.M_MRC_T72, ">=", 2),
    ])
    # NOTE: pro-BNP is not in our dataset, which is a hormone secreted due to
    # stretching of the heart, and is used to rule out or confirm heart failure.
    # Left ventricle mass

    #### Left Ventricular Hypertrophy ####
    LV_mass = 0.8 * 1.04 * (
        (t7data.LVIDD_T72+t7data.IVSDMMODE_T72+t7data.LWPWDMMODE_T72)**3 - t7data.LVIDD_T72**3) + 0.6
    # Left ventricle myocardial mass index height
    Body_areal_height_m = height_m ** 2.7
    t7data["Lvmmi_height"] = LV_mass / Body_areal_height_m
    # Left ventricle hypertrophy height
    LVH_height_men = and_custom(compare(t7data["Lvmmi_height"], ">", 50),
                         compare(t7data.SEX_T7, "==", 1))
    LVH_height_women = and_custom(compare(t7data["Lvmmi_height"], ">", 47),
                           compare(t7data.SEX_T7, "==", 0))
    t7data["LVH_height"] = or_recursive([LVH_height_men, LVH_height_women])

    #### Heart failure with reduced Ejection Fraction ####
    EF3delt_biplane = recode_variable(
        t7data["LVEFBIPLANE_T72"],
        bin_edges=[0, 40, 50, 100])
    # Heart Failure with reduced EF and Dyspnea
    t7data["HFrEF_sympt"] = and_recursive([
        compare(t7data["LVEFBIPLANE_T72"], "<", 40),
        compare(t7data["HF_dyspne1"], "==", 1),
    ])

    # Heart Failure with moderately reduced EF, criteria 2?
    t7data["HFmrEF_krit2"] = or_recursive([
        compare(t7data["LVH_height"], "==", 1),
        compare(LaLarge, "==", 1),
        compare(t7data["DD_score2"], "==", 1),
    ])

    return t7data


def recode_variable(
        numerical_array,
        bin_edges: list,
        bin_labels=None,
        return_float=True,
):
    """Takes numerical variable, assings labels based on which bin each value
    falls into, and returns variable with labels encoded as whole numbers.
    'bins' is a list that contains the edges defining the bins, for instance [0,
    0.2, 0.3, 1] assigns values in range 0 to 1 into 3 bins: [0,0.2), [0.2,0.3)
    and [0.3,1]. Optionally you can assign new labels. Default labels are 1, 2,
    3, etc."""
    # Assign elements to bins
    discretization = np.digitize(numerical_array, bins=bin_edges[1:-1]).astype(float)
    original_labels = np.unique(discretization)
    n_categories = len(original_labels)
    if bin_labels is None:
        bin_labels = {original_labels[i]: i + 1 for i in range(n_categories)}
    # Relabel using the dictionary to map from old to new values
    discretization = pd.DataFrame(
        {"var": discretization}
    ).replace({"var": bin_labels}).values.flatten()
    # Reinsert missing values
    idx_nan = np.isnan(numerical_array)
    if idx_nan.sum() > 0:
        discretization[np.isnan(numerical_array)] = np.nan

    return discretization


def grade_aortic_stenosis(t7data):
    """Create column with severity grades for aortic stenosis (AS) based on the
    aortic valve pressure gradient mean (AVPGmean). Cutoff values are 0, 15, 20,
    and 40 for defining absent, mild, moderate and severe AS."""
    idx_as_none = t7data.avpg_mean < 15
    idx_as_mild = np.logical_and(
        15 <= t7data.avpg_mean, t7data.avpg_mean < 20
    )
    idx_as_moderate = np.logical_and(
        20 <= t7data.avpg_mean, t7data.avpg_mean < 40
    )
    idx_as_severe = np.logical_and(
        40 <= t7data.avpg_mean, t7data.avpg_mean < np.inf
    )
    t7data.loc[idx_as_none, "AS_grade"] = 0.0
    t7data.loc[idx_as_mild, "AS_grade"] = 1.0
    t7data.loc[idx_as_moderate, "AS_grade"] = 2.0
    t7data.loc[idx_as_severe, "AS_grade"] = 3.0

    return t7data





