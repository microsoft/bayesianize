import numpy as np


def calibration_curve(probabilities: np.ndarray, targets: np.ndarray, num_bins: int,
                      top_class_only: bool = True, equal_size_bins: bool = False, min_p: float = 0.0):
    """Calculates the calibration of a classifier (binary or multi-class). Specificially it takes
    predicted probability values, assigns them to a given number of bins (keeping either the width
    of the bins fixed or the number of predictions assigned to each bin) and then returns for each
    bin the mean predicted probability of the positive class occuring as well as the empirically observed
    frequency as per the targets. Additionally the relative size of each bin is returned. Note that
    all inputs are assumed to be well-specified, i.e. probabilities between 0 and 1 and, for multi-class
    targets, to sum to 1 across the final dimension.

    Using the default options top_class_only=True and equal_bin_size=False returns mean probabilities,
    bin_frequency and bin_weights values as used for the standard ECE forumlation, e.g. in
    http://openaccess.thecvf.com/content_CVPRW_2019/papers/Uncertainty%20and%20Robustness%20in%20Deep%20Visual%20Learning/Nixon_Measuring_Calibration_in_Deep_Learning_CVPRW_2019_paper.pdf
    top_class_only=False gives results for the Static Calibration Error, equal_size_bins=True the Adaptive Calibration
    Error (the paper does not specify whether to set top_class_only to True or False). Setting min_p > 0
    corresponds the Thresholded Adaptive Calibration Error. To calculate these calibration error, the outputs
    of this fucntion can directly be passed into expected_calibration_error function.

    Args:
        probabilities: Array containing probability predictions.
        targets: Array containing classification targets.
        num_bins: Number of bins for probability values.
        top_class_only: Whether to only use the maximum predicted probability for multi-class classification
            or all probabilities.
        equal_size_bins: Whether to have each bin an equal number of predictions assigned vs. equal width.
        min_p: Minimum threshold for the probabilities to count.
    Returns:
        bin_probability: Average predicted probability. NaN for empty bins.
        bin_frequency: Average observed true class frequency. NaN for empty bins.
        bin_weights: Relative size of each bin. Zero for empty bins.
    """

    if probabilities.ndim == targets.ndim + 1:
        # multi-class
        if top_class_only:
            # targets are converted to per-datapoint accuracies, i.e. checking whether or not the predicted
            # class was observed
            predictions = np.cast[targets.dtype](probabilities.argmax(-1))
            targets = targets == predictions
            probabilities = probabilities.max(-1)
        else:
            # convert the targets to one-hot encodings and flatten both those targets and the probabilities,
            # treating them as independent predictions for binary classification
            num_classes = probabilities.shape[-1]
            one_hot_targets = np.cast[targets.dtype](targets[..., np.newaxis] == np.arange(num_classes))
            targets = one_hot_targets.reshape(*targets.shape[:-1], -1)
            probabilities = probabilities.reshape(*probabilities.shape[:-2], -1)

    elif probabilities.ndim != targets.ndim:
        raise ValueError("Shapes of probabilities and targets do not match. "
                         "Must be either equal (binary classification) or probabilities "
                         "must have exactly one dimension more (multi-class).")
    else:
        # binary predictions, no pre-processing to do
        pass

    if equal_size_bins:
        quantiles = np.linspace(0, 1, num_bins + 1)
        bin_edges = np.quantile(probabilities, quantiles)
        # explicitly set upper and lower edge to be 0/1 
        bin_edges[0] = 0
        bin_edges[-1] = 1
    else:
        bin_edges = np.linspace(0, 1, num_bins + 1)

    # bin membership has to be checked with strict inequality to either the lower or upper
    # edge to avoid predictions exactly on a boundary to be included in multiple bins.
    # Therefore the exclusive boundary has to be slightly below or above the actual value
    # to avoid 0 or 1 predictions to not be assigned to any bin
    bin_edges[0] -= 1e-6
    lower = bin_edges[:-1]
    upper = bin_edges[1:]
    probabilities = probabilities.reshape(-1, 1)
    targets = targets.reshape(-1, 1)

    # set up masks for checking which bin probabilities fall into and whether they are above the minimum
    # threshold. I'm doing this by multiplication with those booleans rather than indexing in order to
    # allow for the code to be extensible for broadcasting
    bin_membership = (probabilities > lower) & (probabilities <= upper)
    exceeds_threshold = probabilities >= min_p

    bin_sizes = (bin_membership * exceeds_threshold).sum(-2)
    non_empty = bin_sizes > 0

    bin_probability = np.full(num_bins, np.nan)
    np.divide((probabilities * bin_membership * exceeds_threshold).sum(-2), bin_sizes,
              out=bin_probability, where=non_empty)

    bin_frequency = np.full(num_bins, np.nan)
    np.divide((targets * bin_membership * exceeds_threshold).sum(-2), bin_sizes,
              out=bin_frequency, where=non_empty)

    bin_weights = np.zeros(num_bins)
    np.divide(bin_sizes, bin_sizes.sum(), out=bin_weights, where=non_empty)

    return bin_probability, bin_frequency, bin_weights


def expected_calibration_error(mean_probability_predicted: np.ndarray, observed_frequency: np.ndarray,
                               bin_weights: np.ndarray):
    """Calculates the ECE, i.e. the average absolute difference between predicted probabilities and
    true observed frequencies for a classifier and its targets. Inputs are expected to be formatted
    as the return values from the calibration_curve method. NaNs in mean_probability_predicted and
    observed_frequency are ignored if the corresponding entry in bin_weights is 0."""
    idx = bin_weights > 0
    return np.sum(np.abs(mean_probability_predicted[idx] - observed_frequency[idx]) * bin_weights[idx])
