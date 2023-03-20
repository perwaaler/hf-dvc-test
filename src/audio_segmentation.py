"""Performs joint segmentation of all the audio-files in the dataset. Output is
a dataframe (2124 rows and 4 columns) where each cell contains the segmentation
dictionary for the corresponding audio file. Output is stored in the outputs
folder."""

import logging

import numpy as np
import pandas as pd

from utils.backend import dump_python_object_to_file
from utils.prepare_data import prepare_t7data
from utils.general import get_audio_names

from segmentation.src.utils.segmentation import segment_audio_jointly


logging.getLogger("segmentation").setLevel(logging.ERROR)
LOGGER = logging.getLogger(__name__)


def segment_all_audiofiles(
        index_subset=None,
        write_to_file=True,
):
    """Segments all audio-files in the T7 dataset and saves the results in a
    dataframe in the output folder."""
    # Get tabular data
    t7data = prepare_t7data(index_subset)
    n_data = t7data.shape[0]
    # Get list of name of audio files
    audio_names = get_audio_names(t7data)
    # Create container dataframe to store segmentations
    segmentation_all = pd.DataFrame(
        index=np.arange(n_data),
        columns=["pos_1", "pos_2", "pos_3", "pos_4"]
    )

    # Iterate of participants and auscultation positions
    for i_row in range(n_data):
        seg_list = segment_audio_jointly(audio_names[i_row])
        for i_ausc in range(4):
            segmentation_all.iloc[i_row, i_ausc] = seg_list[i_ausc]
            LOGGER.info("%g cardiac intervals extracted", len(seg_list[i_ausc]))
        LOGGER.info("Segmentation participant %g complete", i_row)

    if write_to_file is True:
        dump_python_object_to_file(
            segmentation_all, file_name="segmentation/segmentation"
        )

    return segmentation_all


if __name__ == "__main__":
    # Get audio filename from command line
    assigned_states_preds = segment_all_audiofiles(np.arange(5))
