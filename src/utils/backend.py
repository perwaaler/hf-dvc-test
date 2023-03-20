"""Backend utilities for setting directory paths, setting hyper parameters, and
reading and writing files."""

import os
import pathlib
import pickle
import pandas as pd
import yaml


# Directory paths
ROOT_DIR = pathlib.Path(__file__).parents[3]
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "heartfailure/output")
AUDIO_DIR = os.path.join(ROOT_DIR, "data/heart-sounds/T7")

# Parameter paths
TABULAR_DATA_PATH = os.path.join(DATA_DIR, "tabular-data/T7_heartSounds_variables.csv")
PARAM_FOLDER_PATH =  os.path.join(ROOT_DIR, "heartfailure/src/params")
FEATURIZATION_PARAMS_PATH = os.path.join(PARAM_FOLDER_PATH, "featurization.yml")
TRAINING_PARAMS_PATH = os.path.join(PARAM_FOLDER_PATH, "training.yml")
PLOTTING_PARAMS_PATH = os.path.join(PARAM_FOLDER_PATH, "plotting.yml")


def dump_python_object_to_file(variable, file_path_relative):
    """Save python object. file_path_relative is file path relative to the
    output folder."""
    full_path = os.path.join(OUTPUT_DIR, file_path_relative)
    with open(full_path, 'wb') as file:
        pickle.dump(variable, file)
        print(f'Object successfully saved to "{file_path_relative}"')


def read_yaml_from_path(
        path: str,
) -> dict:
    """Read yaml file from `path`."""
    with open(path, mode="r", encoding="utf-8") as jfl:
        yaml_loaded = yaml.safe_load(jfl)
    return yaml_loaded


def read_tabular_data():
    """Read T7 tabular data from csv file."""
    t7data = pd.read_csv(
        TABULAR_DATA_PATH,
        encoding="ISO-8859-1",
        dtype={"OTHERECHOREFTX_T72": str,
               "OFINDINGS_DESCR_T72": str,
               "IMISSED_COMMENTS_T72": str
               },
        parse_dates=["ECHO_DATE_T72",
                     "ECHO_REFERRAL_COMMENT_T72",
                     "ECHO_TIME_T72",
                     "ECHO_CHKIN_DATE_T72"
                     ],
    )
    return t7data


def load_python_object(filepath_relative):
    """Loads a python object stored in the output folder."""
    full_path = os.path.join(OUTPUT_DIR, filepath_relative)
    with open(full_path, 'rb') as file:
        obj = pickle.load(file)
    return obj


# Load hyper parameters
FeaturesPar = read_yaml_from_path(FEATURIZATION_PARAMS_PATH)
TrainingPar = read_yaml_from_path(TRAINING_PARAMS_PATH)
PlotPar = read_yaml_from_path(PLOTTING_PARAMS_PATH)