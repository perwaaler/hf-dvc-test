"""Contains functions used for computing performance metrics."""

import numpy as np
import tensorflow as tf

from sklearn.metrics import roc_auc_score
from scipy.stats.stats import pearsonr


def mean(array, return_ci=False, axis=0):
    """Computes the mean of array and optionally returns 95% confidence
    interval. array can be vector or multi-dimensional array, in which case it
    computes means and confidence intervas across the specified axis. Confidence intervals are returned on the form: mean, lower, upper."""
    avg = np.mean(array, axis=axis)
    if return_ci is True:
        n_samples = np.shape(array)[axis]
        stdev_mean = np.std(array, axis=axis)/n_samples
        ci_lower = avg - stdev_mean*1.96
        ci_upper = avg + stdev_mean*1.96
        return avg, ci_lower, ci_upper
    return avg


def accuracy(y_true, y_pred):
    """Computes accuracy."""
    y_true = convert_to_np_array_and_squeeze(y_true)
    y_pred = convert_to_np_array_and_squeeze(y_pred)
    return np.mean(y_true==y_pred)


def sensitivity(y_true, y_pred):
    """Computes sensitivity."""
    # extract numpy arrays if tensors are provided
    y_true = convert_to_np_array_and_squeeze(y_true)
    y_pred = convert_to_np_array_and_squeeze(y_pred)
    sens = np.mean(y_pred[y_true])
    return sens


def specificity(y_true, y_pred):
    """Compute specificity."""
    # extract numpy arrays if tensors are provided
    y_true = convert_to_np_array_and_squeeze(y_true)
    y_pred = convert_to_np_array_and_squeeze(y_pred)
    sens = np.mean(y_pred[y_true])
    return sens


def mean_of_sn_and_sp(y_true, y_pred):
    """Compute the mean of sensitivity and specificity, which is similar to
    accuracy but assigns equal weights to each type of error."""
    sens = sensitivity(y_true, y_pred)
    spec = specificity(y_true, y_pred)
    return (sens + spec)/2


def convert_to_np_array_and_squeeze(array):
    """Checks if array is a tensor, and changes it to numpy array and flattens
    it if it is."""
    if tf.is_tensor(array):
        # convert to numpy array:
        array = array.numpy()
    # remove superfluous dimensions:
    array = tf.squeeze(array)
    return array


def get_auc(
        bin_target,
        scores,
):
    """Computes the ROC-AUC."""
    auc = roc_auc_score(y_true=bin_target, y_score=scores)
    return auc


def linear_correlation(x1, x2):
    """Compute the linear corellation between two vectors x1 and x2."""
    correlation = pearsonr(x1, x2)


def mae(y_test, y_test_pred):
    """Mean average error between target and prediction."""
    y_test = tf.squeeze(y_test)
    y_test_pred = tf.squeeze(y_test_pred)
    return tf.metrics.mae(y_test, y_test_pred).numpy()


def mae_relative(y_test, y_test_pred):
    """Mean absolute percentage error between target and prediction."""
    y_test = tf.squeeze(y_test)
    y_test_pred = tf.squeeze(y_test_pred)
    mape = tf.metrics.mean_absolute_percentage_error(
        y_test,
        y_test_pred
    )/100
    return mape.numpy()


def mse(y_test, y_test_pred):
    """Mean square error between target and prediction."""
    y_test = tf.squeeze(y_test)
    y_test_pred = tf.squeeze(y_test_pred)
    return tf.metrics.mse(y_test, y_test_pred).numpy()


def rmse(y_test, y_test_pred):
    """Root mean square error between target and prediction."""
    y_test = tf.squeeze(y_test)
    y_test_pred = tf.squeeze(y_test_pred)
    return tf.sqrt(tf.metrics.mse(y_test, y_test_pred)).numpy()
