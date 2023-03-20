import json
import yaml
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import io
from sklearn.metrics import roc_auc_score, roc_curve

from utils.backend import FeaturesPar, TrainingPar, PlotPar, OUTPUT_DIR, load_python_object
from utils.general import time_to_idx


def summarize_column(dataframe, n_significant=None):
    """Prints lots of summary statistics for columns in a dataframe."""
    if n_significant is None:
        n_significant = 2
    if isinstance(dataframe, np.ndarray):
        dataframe = pd.DataFrame(dataframe)

    if dataframe.ndim < 3:
        # if isinstance(dataframe.loc[0], np.bool_):
        #     dataframe = boolean_to_floats(dataframe)

        description = dataframe.describe().round(n_significant)
        unique_values = dataframe.drop_duplicates(
        ).values[:5].round(n_significant)
        counts = dataframe.value_counts().iloc[:5].round(n_significant)
        counts = pd.DataFrame({"count": counts}).round(n_significant)
        counts["frequencies"] = 100*counts.values.flatten() / \
            np.sum(counts.values).round(n_significant)
        n_missing = dataframe.isnull().sum()

        print(f"Summary metrics:\n{description}\n")
        print(f"Unique values (sample):\n{unique_values}\n")
        print(f"Counts and count frequencies for sample values:\n{counts}\n")
        print(f"N.o. missing: {n_missing}")
        print(f"N.o. not missing: {len(dataframe)-n_missing}")
    else:
        description = dataframe.describe().round(n_significant)
        n_missing = dataframe.isnull().sum()
        print(f"Summary metrics:\n{description}\n")
        print(f"The number of NaN values is: {n_missing}")


def boolean_to_floats(array):
    x_floats = np.zeros(len(array))
    x_floats[array] = 1.0
    array.loc[:] = x_floats
    return array

    
def get_matlab_array(mat_file):
    """Loads a matlab cell array and converts it to python list."""
    x = io.loadmat(mat_file + ".mat")
    x = x[mat_file]
    return x


def dump_dict_to_json(
        path: str,
        dump_dict: dict,
) -> None:
    """Save thr `dump_dict` to the json file under `path`."""
    # pylint: disable=invalid-name
    dirpath = os.path.dirname(path)
    if dirpath != '':
        os.makedirs(dirpath, exist_ok=True)
    with open(path, mode="w", encoding='utf-8') as f:
        json.dump(
            dump_dict,
            f,
            sort_keys=True,
            indent=4)


def dump_dict_to_yaml(
        path_relative: str,
        dump_dict: dict,
) -> None:
    """Save thr `dump_dict` to the json file under `path`."""
    # pylint: disable=invalid-name
    path = os.path.join(OUTPUT_DIR, path_relative)
    with open(path, mode="w", encoding='utf-8') as yaml_file:
        yaml.dump(
            dump_dict,
            yaml_file,
            sort_keys=True,
            indent=4)


################
### Plotting ###
################


def plot_cv_training_histories(
        history_filepath="training_history.py",
        figure_filename="training_history.png",
        write_figure_to_file=True,
):
    """Plots cross-validation training histories. If no arguments are provided
    it defaults to loading training history file with generic names
    'training_history.py' and writes the image to the
    figures/training-history/training_history.png folder."""
    history_list = load_python_object("training-history/cv/" + history_filepath)
        # history_list = load_python_object("training-history/cv/training_history.py")
    ax = plt.subplots(8, 2)[-1]
    for i, history in enumerate(history_list):
        plot_training_history(history=history,
                            metric_to_plot=TrainingPar["metric_to_plot"],
                            ax_plot=ax[i])
    if write_figure_to_file:
        plt.savefig(OUTPUT_DIR + "/figures/training-history/" + figure_filename)

    plt.show()
    


def get_spectrogram(audio, sr=FeaturesPar["audio_sr"]):
    """Computes the spectrogram of the audio."""
    spectrogram = librosa.stft(
        audio,
        window="hann",
        win_length=time_to_idx(25e-3, sr),
        hop_length=time_to_idx(25e-3 - 10e-3, sr),
        center=True,
    )

    return spectrogram


def plot_timeseries(audio, sr=FeaturesPar["audio_sr"]):
    """Plot a time-series with time in seconds on the x-axis."""
    n_samples = len(audio)
    t = n_samples / sr
    x = np.linspace(0, t, n_samples)

    plt.autoscale(enable=True, axis='x', tight=True)
    plt.plot(x, audio)
    plt.xlabel("time (s)")


def plot_spectrogram(x, sr=FeaturesPar["audio_sr"]):
    """Plots spectrogram"""

    if x.ndim > 1:
        # Spectrogram is provided
        spectrogram = x
        spect_sr = np.round((25e-3 - 10e-3)**-1).astype(int)
        audio_duration = spectrogram.shape[1] / spect_sr
    else:
        # Audio provided; compute the spectrogram
        audio = x
        spectrogram = get_spectrogram(audio, sr)
        audio_duration = len(audio) / sr
        print(f"audio duration is: {audio_duration}")

    frq = librosa.fft_frequencies(sr=sr, n_fft=2048)
    max_frq = 250
    plt.imshow(
        np.abs(spectrogram[frq < max_frq, :])**2,
        origin='lower',
        extent=[0, audio_duration, 0, max_frq],
        aspect='auto'
    )

    plt.xlabel("time [min]")
    plt.ylabel("frequency [Hz]")


def plot_segments(
    audio,
    segmentation,
    sr=FeaturesPar["audio_sr"],
    levels=[0.01, 0.02, 0.03, 0.04],
    plot_spect=False,
    plot_audio=False,
):
    """Plots cardiac states as coloured line segments together with the
    spectrogram of the audio."""

    if plot_spect and not plot_audio:
        # Plot audio takes priority over plot spectrogram
        plot_spectrogram(audio, sr=sr)
    if plot_audio:
        plot_timeseries(audio, sr=sr)

    state_map = {
        "s1": {"lvl": levels[0], "col": "k"},
        "systole": {"lvl": levels[1], "col": "r"},
        "s2": {"lvl": levels[2], "col": "k"},
        "diastole": {"lvl": levels[3], "col": "b"}
    }
    n_segments = len(segmentation)

    for i in range(n_segments):
        y = state_map[segmentation[i]["label"]]["lvl"]
        col = state_map[segmentation[i]["label"]]["col"]
        x1 = segmentation[i]["start"]
        x2 = segmentation[i]["end"]

        plt.plot(
            (x1, x2),
            (y,  y),
            col,
            linewidth=3,
        )

    plt.autoscale(enable=True, tight=True)


def get_auc_and_plot_roc(
        binary_target,
        scores,
        ax_plot=None,
        color="k",
):
    """Computes the AUC and, optionally, plots ROC curve."""
    fpr, tpr, thresholds = roc_curve(binary_target, scores)
    auc = roc_auc_score(y_true=binary_target, y_score=scores)
    if ax_plot is None:
        ax_plot = plt.subplots(1, 1)[-1]
    ax_plot.plot(fpr, tpr, c=color)
    ax_plot.set_title(f"AUC: {auc:.2}", size=PlotPar["sz_txt_roc"])

    return auc


def plot_training_results(scores_valid, df, history=None):
    """Plot ROC curvs for prediction of murmur (>= 1 and 2) and AS using murmur-grade-sum as predictor."""
    murgrade_array = df[
        ["murgrade1", "murgrade2", "murgrade3", "murgrade4"]
    ].values
    as_array = df[["AS_grade"]].values

    # Training plots and ROC curve
    if history is not None:
        training_loss = history.history["loss"]
        validation_loss = history.history["val_loss"]
        plt.plot(training_loss)
        plt.plot(validation_loss)
        plt.xlabel('loss [MSE]')

    # predicting murmur grade >= 1
    plt.figure()
    get_auc_and_plot_roc(
        murgrade_array.flatten() >= 1,
        scores_valid.flatten(),
        plot=True,
    )
    # predicting murmur grade >= 2
    get_auc_and_plot_roc(
        murgrade_array.flatten() >= 2,
        scores_valid.flatten(),
        plot=True,
    )

    # *** predicting AS with sum ***
    plt.figure()
    get_auc_and_plot_roc(
        as_array >= 1,
        np.sum(scores_valid, axis=1),
        plot=True,
        color="r",
    )

    plt.show()


def plot_training_history(
        history: dict,
        metric_to_plot: str=None,
        ax_plot=None
):
    """Plot training and validation history."""
    if not hasattr(history, "keys"):
        history = history.history

    if ax_plot is None:
        if metric_to_plot is None:
            _, ax_plot = plt.subplots()
            ax_plot = [ax_plot]
        else:
            _, ax_plot = plt.subplots(1, 2)

    if "loss" in history.keys():
        ax_plot[0].plot(history["loss"])
        ax_plot[0].set_title("Training loss")

    if metric_to_plot is not None:
        if metric_to_plot in history.keys():
            ax_plot[1].plot(history[metric_to_plot])
            ax_plot[1].set_title(metric_to_plot)
