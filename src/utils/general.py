"""General utility functions."""

import os

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf

from utils.backend import AUDIO_DIR, OUTPUT_DIR, FeaturesPar


def get_subset_idx(x_superset, x_subset):
    """Takes a lists x and x_subset, and returns the indices for the elements
    in x_subset that are in x."""
    shape = np.shape(x_subset)
    if shape == ():
        x_subset = [x_subset]

    idx_list = []
    for _, element_i in enumerate(x_subset):
        idx_list.append(list(x_superset).index(element_i))

    return idx_list


def get_unique_elements(array: list):
    """Takes a list and returns a list with the unique elements."""
    return list(set(array))


def load_t7_audio(
        id_participant,
        audio_sr=FeaturesPar["audio_sr"],
        ausc_index=None,
):
    """Loads T7 audio files for specified auscultation areas associated with
    participant ID."""
    if ausc_index is None:
        ausc_index=[0, 1, 2, 3]

    n_audio = np.size(ausc_index)
    audio = [None] * n_audio

    if isinstance(ausc_index, int):
        ausc_index = [ausc_index]

    for i, ausc_ind_i in enumerate(ausc_index):
        audio_name = f"{id_participant}_hjertelyd_{ausc_ind_i+1}.wav"
        audio_path = os.path.join(AUDIO_DIR, audio_name)
        audio[i] = librosa.load(audio_path, sr=audio_sr)[0]

    if n_audio == 1:
        audio = audio[0]

    return audio


def get_audio_names(t7data):
    """Creates a list of lists with audio names (e.g.,
    '10003711_hjertelyd_1.wav') of the participants of T7."""
    audio_names = [None] * len(t7data)

    for i, id_participant in enumerate(t7data["id"]):
        audio_names[i] = []
        for pos in range(4):
            audio_names[i].append(f"{id_participant}_hjertelyd_{pos+1}.wav")

    return audio_names


def get_indices_for_id_list(id_all, id_subset):
    """Gets the indices for a list of id's in the T7 dataset."""
    indices = get_subset_idx(t7data["id"], id_list)
    return indices


def create_df_of_empty_lists(n_rows, n_cols, column_names=None):
    """Creates a dataframe consisting of empty lists to be used as a
    placeholder."""
    if column_names is None:
        column_names = np.arange(n_cols).astype(int)

    dataframe_placeholder = pd.DataFrame(
        index=np.arange(n_rows),
        columns=np.arange(n_cols),
    )
    for i in range(n_rows):
        for j in range(n_cols):
            dataframe_placeholder.iloc[i, j] = []
    
    dataframe_placeholder.columns = column_names
    
    return dataframe_placeholder


def and_recursive(statement_lists, return_bool=False):
    """Version of np.logical_or that takes list of numerical arrays with
    elements taking values 0, 1, or NaN, and maintains NaN values in positions
    where all arrays have missing values. statement_lists is a list of boolean
    arrays."""
    truth_tracker = and_custom(statement_lists[0], statement_lists[1])
    for i in range(1, len(statement_lists)):
        truth_tracker = and_custom(truth_tracker, statement_lists[i])
    if return_bool:
        truth_tracker = truth_tracker.astype(bool)
    return truth_tracker


def or_recursive(statement_lists, return_bool=False):
    """Version of np.logical_or that takes list of numerical arrays with
    elements taking values 0, 1, or NaN, and maintains NaN values in positions
    where all arrays have missing values. statement_lists is a list of boolean
    arrays."""
    truth_tracker = or_custom(statement_lists[0], statement_lists[1])
    for i in range(1, len(statement_lists)):
        truth_tracker = or_custom(truth_tracker, statement_lists[i])
    if return_bool:
        truth_tracker = truth_tracker.astype(bool)
    return truth_tracker


def or_custom(array_1, array_2, discard_when_any_nan=False):
    """Takes two numerical (also works with boolean arrays) arrays, with
    elements assumed to be either 0, 1 or np.nan, and returns an array with True
    when at least one element is to 1, False when both are 0, and np.nan when no
    definitive answer is impossible. Used instead of np.logical_and when it is
    desired to track of missing values."""
    # Find locations where both have missing values
    truth_array = np.zeros(len(array_1))
    # Find confirmed True and confirmed False
    idx_true = np.logical_or(array_1 == True, array_2 == True)
    idx_false = np.logical_and(array_1 == False, array_2 == False)
    # Find positions where at least one is missing
    idx_missing = np.logical_or(np.isnan(array_1), np.isnan(array_2))
    truth_array[idx_missing] = np.nan
    # Substitute nan in places where an answer can be inferred despite missing value
    truth_array[idx_true] = 1
    truth_array[idx_false] = 0
    if discard_when_any_nan:
        truth_array[idx_missing] = np.nan

    return truth_array


def and_custom(array_1, array_2, discard_when_any_nan=False):
    """Takes two numerical (also works with boolean arrays) arrays, with
    elements assumed to be either 0, 1 or np.nan, and returns an array with True
    when both elements are equal to 1, False when at least one is 0, and np.nan
    when no definitive answer is impossible. Used instead of np.logical_and when
    it is desired to track of missing values."""
    truth_array = np.zeros(len(array_1))
    # Find confirmed True and confirmed False
    idx_true = np.logical_and(array_1 == True, array_2 == True)
    idx_false = np.logical_or(array_1 == False, array_2 == False)
    # Find positions where at least one is missing
    idx_missing = np.logical_or(np.isnan(array_1), np.isnan(array_2))
    truth_array[idx_missing] = np.nan
    # Substitute nan in places where an answer is possible despite missing value
    truth_array[idx_true] = 1
    truth_array[idx_false] = 0
    if discard_when_any_nan:
        truth_array[idx_missing] = np.nan
    return truth_array


def extract_class_using_criteria(array, class_criteria=">= 1"):
    """Extracts a target class using a threshold and a relation that can be e.g.
    '>', '>=', '==', etc..."""
    if " " not in class_criteria:
        print("class criteria must contain space, e.g. '>= 1'")
    criteria_relation = class_criteria.split(" ")[0]
    class_threshold = float(class_criteria.split(" ")[1])
    class_array = compare(array, criteria_relation, class_threshold)
    return class_array


def compare(
        array,
        relation="==",
        comparison_value=1,
        preserve_nan=True,
):
    """Performs elementwise comparison for categorical array. Useful when array
    is ordinal array, in which case it desirable to compare order."""
    idx_missing = np.isnan(array)
    if relation == "==":
        comparison_outcome = (array == comparison_value)
    if relation == ">=":
        comparison_outcome = (array >= comparison_value)
    if relation == "<=":
        comparison_outcome = (array <= comparison_value)
    if relation == ">":
        comparison_outcome = (array > comparison_value)
    if relation == "<":
        comparison_outcome = (array < comparison_value)
    
    if tf.is_tensor(comparison_outcome):
        return comparison_outcome
    else:
        # Convert True/False to 0/1
        comparison_outcome = comparison_outcome.astype(float)
        if preserve_nan is True:
            comparison_outcome[idx_missing] = np.nan
        return comparison_outcome


def idx_to_time(idx_list, sampling_rate):
    """Converts a list of indices to time in seconds."""
    time_list = np.array(idx_list) / sampling_rate
    return time_list


def time_to_idx(time_list, sampling_rate):
    """Converts a list of indices to time in seconds."""
    idx_list = np.round(np.array(time_list) * sampling_rate).astype(int)
    return idx_list


def normalize_array(array):
    """Normalize an an array by subtracting the mean and dividing by the
    standard deviation (of all elements)."""
    array_normalized = (array - np.mean(array)) / np.std(array)
    return array_normalized


def drop_rows_on_condition(condition, dataframe):
    """Drops rows in dataframe that meet the specified condition. Example
    of condition: dataframe.column1 == 13.2."""
    dataframe.drop(dataframe[condition].index, inplace=True)
    dataframe.reset_index(inplace=True, drop=True)
    return dataframe


def save_model(model, model_name):
    """Save train model to output/models folder"""
    model_name_full_path = os.path.join(OUTPUT_DIR, "models", model_name)
    model.save(model_name_full_path)


def load_model(model_name):
    """Load trained model from output/models folder."""
    model_name_full_path = os.path.join(OUTPUT_DIR, "models", model_name)
    model = tf.keras.models.load_model(model_name_full_path)
    return model
