"""Scripts for partitioning the dataset into training, validation or test
sets, or creating cross validation splits."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from utils.backend import TrainingPar
from utils.general import get_subset_idx


def get_train_test_split(
    t7data,
    test_size,
    id_list_full=None, 
    seed=None,
    stratification_columns=None,
    ):
    """Function that splits the T7 dataset (full set or subset) into a training
    set and test set. Returns indices (w.r.t. the full T/ dataset) for bpth
    sets."""
    # Set defaults
    if stratification_columns is None:
        stratification_columns = ["mur1_presence", "mur2_presence", "mur3_presence", "mur3_presence"]
    if seed is not None:
        seed = 1
    np.random.seed(seed)

    t7data_train, t7data_test = train_test_split(
        t7data,
        test_size=test_size,
        stratify=t7data[stratification_columns]
    )
    # Get lists of indices
    if id_list_full is None:
        idx_train = t7data_train.index.values.tolist()
        idx_test = t7data_test.index.values.tolist()
    else:
        idx_train = get_subset_idx(id_list_full, t7data_train.id)
        idx_test = get_subset_idx(id_list_full, t7data_test.id)

    return idx_train, idx_test
 

def get_stratified_cross_val_splits(
        t7data,
        n_folds=TrainingPar["n_folds"],
        stratification_columns=["mur_presence", "AS_clinical", "MS_clinical"],
        id_list=None,
        seed=1,
):
    """Creates n_folds training-validation splits, where splits are performed by
    row ID. Ensures rougly equal proportions of AS, MS, and presence of murmur
    (grade>=1 in one or more positions) in each split. Returns training and
    validation indices."""
    # Create dummy input data (required by 'skf.split')
    n_data = len(t7data)
    dummy_data = np.ones(n_data)
    # Split
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    classes = create_class_combination_column(t7data[stratification_columns])
    # Collect split indices
    cv_splits = [None] * n_folds
    for i, (idx_train, idx_valid) in enumerate(skf.split(dummy_data, classes)):
        id_train = t7data["id"][idx_train]
        id_valid = t7data["id"][idx_valid]
        if id_list is not None:
            # Get indices relative to the full dataset
            idx_train = get_subset_idx(id_list, id_train)
            idx_valid = get_subset_idx(id_list, id_valid)

        cv_splits[i] = {
            "idx_train": idx_train,
            "idx_valid": idx_valid,
        }

    return cv_splits


def create_class_combination_column(classes_df):
    """Concatenates the values in the specified columns ('class_columns') into
    strings that represent which classes a row of data belongs to. For instance,
    if class columns correspond to two classes, and if a row belongs to class 1
    but not class 2, then the corresponding string would be '10'."""
    class_membership = []
    for i in range(len(classes_df)):
        class_membership.append(
            ''.join(classes_df.loc[i].
                    astype(int).
                    astype(str))
        )
    class_membership = pd.DataFrame({"class_membership": class_membership}, dtype=str)
    return class_membership
