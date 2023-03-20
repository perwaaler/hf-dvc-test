"""Contains functions for removing spikes in audio."""

from utils.backend import FeaturesPar
import numpy as np


def schmidt_spike_removal(
        audio,
        audio_sr=FeaturesPar["audio_sr"],
):
    """Perform Schmidt's spike removal as described in paper by Schmidt et
    al."""
    window_size = round(audio_sr / 2)
    # Find any samples outside of an integer number of windows
    trailing_samples = len(audio)
    # Reshape the signal into a number of windows
    upper_ind = -trailing_samples if trailing_samples > 0 else None
    sampleframes = np.reshape(
        audio[0:upper_ind],
        newshape=(window_size, -1),
        order="F",
    )
    # Find the MAAs:
    maas = np.max(abs(sampleframes), axis=0)

    # If there are still samples greater than 3 * the median value of the
    # MAAs, then remove those spikes
    while sum(maas > np.median(maas) * 3) > 0:
        # Find the window with the max MAA
        window_num = np.argmax(maas)
        # max_val = maas[window_num]

        # Find the postion of the spike within that window
        spike_position = np.argmax(np.abs(sampleframes[:, window_num]))
        # spike_val = sampleframes[spike_position, window_num]

        # Finding zero-crossings
        zero_crossings = np.append(
            abs(np.diff(np.sign(sampleframes[:, window_num]))) > 1,
            0,
        )

        # Find the start of the spike by finding the last zero crossing before
        # spike position. If that is empty, take the start of the window
        spike_start = np.argwhere(
            zero_crossings[0:spike_position],
        )
        if len(spike_start) == 0:
            spike_start = 0
        else:
            spike_start = spike_start[-1]

        # Find the end of the spike by finding the first zero crossing after
        # spike position. If that is empty, take the end of the window
        zero_crossings[0:spike_position] = 0
        spike_end = np.argwhere(zero_crossings)
        if len(spike_end) == 0:
            spike_end = window_size
        else:
            spike_end = min(window_size, spike_end[0])  # end of while loop

        # Set to Zero
        sampleframes[int(spike_start):int(spike_end) + 1, window_num] = 0.0001
        # Recalculate MAAs
        maas = np.max(abs(sampleframes), axis=0)

    despiked_audio = np.reshape(sampleframes, (-1, 1), order="F")
    # Add the trailing samples back to the signal
    # despiked_signal = [despiked_signal; audio[len(despiked_signal)+1:end]]
    despiked_audio = np.append(despiked_audio, audio[len(despiked_audio):])

    return despiked_audio
