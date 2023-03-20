import numpy as np
import pandas as pd

from utils.general import create_df_of_empty_lists
from utils.backend import load_python_object, dump_python_object_to_file
from utils.featurization import extract_segment_borders, add_additional_input_features


def add_feature_rows_to_mfcc():
    """Adds rows to each MFCC array with information on time and auscultation
    position. Dumps the resulting dataframe with modified input arrays in the output folder."""

    # Load files
    segmentation_all = load_python_object("segmentation/segmentation")
    mfcc_arrays = load_python_object("rnn-input/rnn_input_arrays")
    # Create placeholder object
    rnn_inputs_mod = create_df_of_empty_lists(
        n_rows=2124,
        n_cols=4,
        column_names=["pos_1", "pos_2", "pos_3", "pos_4"])

    for row_i in range(2124):
        for col_j in range(4):
            segmentation_id_ij = segmentation_all.iloc[row_i, col_j]
            segment_borders_id_ij = extract_segment_borders(segmentation_id_ij)
            mfcc_arrays_id_ij = mfcc_arrays.iloc[row_i, col_j]

            for k_segment, borders in enumerate(segment_borders_id_ij):
                rnn_inputs_mod.iloc[row_i, col_j].append(
                    add_additional_input_features(
                        segment_borders_idx=borders,
                        mfcc_array=mfcc_arrays_id_ij[k_segment],
                        ausc_pos=col_j)
                )

    dump_python_object_to_file(rnn_inputs_mod,
                            "rnn-input/rnn_inputs_mod")


if __name__=="__main__":
    add_feature_rows_to_mfcc()

