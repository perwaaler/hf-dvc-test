"""Computes MFCC array for each audio segment. Returns a dataframe (with
n_participant rows and n_audio_positions columns) where each cell contains the
MFCC arrays of the corresponding audio file (~6 arrays per audio). Assumes that
the output folder contains a file called 'segmentation' with segmentation
dataframe. Output is stored in outputs folder."""

import logging

import numpy as np
import pandas as pd

from utils.audio_processing import schmidt_spike_removal
from utils.backend import FeaturesPar
from utils.backend import dump_python_object_to_file
from utils.backend import load_python_object
from utils.prepare_data import prepare_t7data
from utils.featurization import extract_segment_borders
from utils.featurization import get_mfcc
from utils.featurization import resize_matrix_with_interpolation
from utils.general import load_t7_audio, normalize_array


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s',
)

LOGGER = logging.getLogger(__name__)


def get_feature_representation_of_segments(
        index_subset=None,
        write_to_file=True,
):
    """Computes MFCC arrays for the segments of each audio file (~6 segments per
    audio-file), which are the input sequences of the RNN. MFCC arrays are
    stored in a dataframe (2124 rows by 4 columns, with each element being a
    list of MFCC-arrays). Assumes that segmentation has been done, and that
    segmentation file is located in the output folder. If computing arrays for
    only a subset of participants, provide an index list for the corresponding
    rows."""
    # Load file with segmentations for all audio-files
    segmentations_all = load_python_object("segmentation")
    # Get cleaned dataset
    t7data = prepare_t7data(index_subset)
    n_data = len(t7data)
    # Create dataframe to be filled lists of MFCC arrays corresponding to audio
    # in each respective row and auscultation position
    rnn_input_arrays = pd.DataFrame(
        index=np.arange(n_data),
        columns=["pos_1", "pos_2", "pos_3", "pos_4"]
    )

    # Collect MFCC arrays for each audio file
    for i_row in range(n_data):
        for i_ausc in range(4):
            # Load audio file
            audio = load_t7_audio(
                id_participant=t7data["id"][i_row],
                ausc_index=i_ausc,
            )
            # Compute MFCC arrays for segments of the audio file
            rnn_input_arrays.iloc[i_row, i_ausc] = get_mfcc_arrays_of_audiofile(
                audio,
                segmentation=segmentations_all.iloc[i_row, i_ausc],
            )
        LOGGER.info("MFCC arrays extracted for participant %g", i_row)

    if write_to_file is True:
        dump_python_object_to_file(
            rnn_input_arrays, file_name="rnn-input/rnn_input_arrays"
        )

    return rnn_input_arrays


def get_mfcc_arrays_of_audiofile(
        audio,
        segmentation,
):
    """Takes segmentation data (dictionary with state label and interval
    endpoints) of a single audio-file, extracts audio segments, and computes
    MFCC arrays for each segment."""

    # Remove spikes
    audio = schmidt_spike_removal(audio)
    # Get borders for each segment (list index)
    segment_borders_idx = extract_segment_borders(segmentation)
    n_segments = np.shape(segment_borders_idx)[0]

    # Extract MFCC array for each segment
    input_units = [None] * n_segments
    for k in range(n_segments):
        # Extract k'th audio segment
        audio_segment = audio[np.arange(
            start=segment_borders_idx[k, 0],
            stop=segment_borders_idx[k, 1],
        )]
        # Get MFCC representation of segment
        mfcc_segment = get_mfcc(audio_segment, audio_sr=FeaturesPar["audio_sr"])
        # Normalize MFCC array
        mfcc_segment = normalize_array(mfcc_segment)
        # Resize to fixed shape using cubic interpolation
        mfcc_segment = resize_matrix_with_interpolation(
            mfcc_segment,
            new_shape=[
                FeaturesPar["n_mfcc"],
                FeaturesPar["n_rnn_input"]
            ]
        )
        input_units[k] = mfcc_segment

    return input_units


if __name__ == "__main__":
    # Get audio filename from command line
    assigned_states_preds = get_feature_representation_of_segments(np.arange(5))
