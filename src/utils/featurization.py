"""Contains scripts for extracting features from the audio data."""

import numpy as np
import librosa

from scipy import interpolate
from utils.backend import FeaturesPar
from utils.general import time_to_idx


def extract_segment_borders(segmentation, audio_sr=FeaturesPar["audio_sr"]):
    """Extract segment borders from a list of indices indicating where each
    carciac cycle starts (cycle_lines). Each segment consists of a fixed number
    of cardiac cycles, with n_cycles_per_seg cycles of overlap between each
    segment. Extracts segments untill it runs out of bounds, at which point it
    repeats the process after shifting the starting point once cycle to the
    right relative to the first run. Segment borders are stored as rows in a
    matrix."""

    # Get the starting point for each cardiac cycle
    cycle_borders_idx = get_borders_between_cardiac_cycles(segmentation)

    if audio_sr is not None:
        # Convert from time in seconds to index
        cycle_borders_idx = time_to_idx(cycle_borders_idx, audio_sr)

    # shorten name for convenience
    par = FeaturesPar
    # Set number of cycles to shift right for extraction of each new segment
    n_step_size = par["n_cycles_per_seg"] - par["n_cycles_overlap"]
    # Number of borders separating the segments
    n_seg_borders = len(cycle_borders_idx)
    # Number of cardiac cycles in the audio
    n_cycles_available = n_seg_borders - 1
    # Container object to be filled with the segment endpoints
    segment_borders = np.zeros([par["n_segments_per_audio"], 2], dtype=int)
    shift_and_restart = False
    k = 1

    if n_cycles_available < par["n_cycles_per_seg"]:
        # There are not enough carciac cycles. Only one segment will represent
        # the audio, but will not contain as many cycles as requested.
        segment_borders = np.array([cycle_borders_idx[[0, -1]]])
        c = 1
    else:
        for i in range(par["n_segments_per_audio"]):
            if not shift_and_restart:
                l = n_step_size * (k-1) + 1
                u = l + par["n_cycles_per_seg"]

                if u > n_seg_borders:
                    shift_and_restart = True
                    c = k - 1
                    k = 1
                else:
                    segment_borders[k-1, :] = cycle_borders_idx[[l-1, u-1]]
                    k = k + 1
            else:
                l = n_step_size * (k-1) + 2
                u = l + par["n_cycles_per_seg"]

                if u > n_seg_borders:
                    # Number of segments have been exceeded; can not extract
                    # number of segments requested
                    segment_borders = segment_borders[np.arange(c+k-1), :]
                    break
                segment_borders[c+k-1, :] = cycle_borders_idx[[l-1,u-1]]
                k = k + 1


    if not "c" in locals():
        segment_borders = segment_borders[np.arange(par["n_segments_per_audio"]), :]
    else:
        # Truncate in case more segments were extracted then were requested
        n_segments_extracted = min([par["n_segments_per_audio"], c+k-1])
        segment_borders = segment_borders[np.arange(n_segments_extracted)]

    return segment_borders


def get_borders_between_cardiac_cycles(segmentation: dict):
    """Takes a list of dictionaries with interval endpoints and state
    label and returns an array of points where a new cardiac cycle begins
    (defined as start of each S1)."""
    cycle_borders_idx = []
    for _, interval_dict in enumerate(segmentation):
        if interval_dict["label"] == "s1":
            cycle_borders_idx.append(interval_dict["start"])
    return np.array(cycle_borders_idx)


def get_mfcc(
        audio,
        audio_sr=FeaturesPar["audio_sr"],
        n_mfcc=FeaturesPar["n_mfcc"],
        win_length=FeaturesPar["window_len"],
        win_overlap=FeaturesPar["window_overlap"],
):
    """Calculates the Mel frequency cepstral coefficients used as input features
    in the time-frequency domain."""
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=audio_sr,
        n_mfcc=n_mfcc,
        win_length=time_to_idx(win_length, audio_sr),
        hop_length=time_to_idx(win_length - win_overlap, audio_sr),
    )
    return mfccs


def resize_matrix_with_interpolation(matrix, new_shape):
    """Takes a 2d-array 'matrix' and reshapes it to shape specified by
    'new_shape' using cubic interpolation. Used to standardize the lengths of
    the MFCC arrays, since RNNs do not perform well if sequences get to long."""
    # Create interpolation object
    x_grid_0 = np.linspace(0, 1, matrix.shape[0])
    y_grid_0 = np.linspace(0, 1, matrix.shape[1])
    interpolator = interpolate.interp2d(y_grid_0, x_grid_0, matrix, kind='cubic')
    # Evaluate at new grid corresponding to new size
    x_grid_1 = np.linspace(0, 1, new_shape[0])
    y_grid_1 = np.linspace(0, 1, new_shape[1])
    matrix_resized = interpolator(y_grid_1, x_grid_1)
    return matrix_resized


def add_additional_input_features(
        segment_borders_idx: list,
        mfcc_array,
        ausc_pos,
):
    """The MFCC arrays do not contain information on which position they come
    from or their length of time. Adds a row with time elapsed since start of
    segment as well as one-hot-encoded rows that indicate position from which
    the original audio file was collected."""
    
    # Create auscultation position dummy rows
    position = np.zeros([4, FeaturesPar["n_rnn_input"]])
    position[ausc_pos] = 1
    # Create row that tracks elapsed time
    segment_borders_sec = segment_borders_idx/FeaturesPar["audio_sr"]
    time_s = np.linspace(
        start=0, 
        stop=segment_borders_sec[1] - segment_borders_sec[0], 
        num=FeaturesPar["n_rnn_input"])
    # Append new rows to MFCC array
    additional_features = np.vstack([position, time_s])
    mfcc_with_added_features = np.vstack(
        [mfcc_array, additional_features]
    )
    
    return mfcc_with_added_features