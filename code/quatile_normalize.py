import numpy as np
from scipy.stats import percentileofscore


def eQM_replace(ref_dataset, model_present, model_future):
    """
    For each model_future value, get its percentile on the CDF of model_present,
    then ust it to get a value from the model_present.
    returns: downscaled model_present and model_future
    """
    model_present_corrected = np.zeros(model_present.size)
    model_future_corrected = np.zeros(model_future.size)

    for ival, model_value in enumerate(model_present):
        percentile = percentileofscore(model_present, model_value)
        model_present_corrected[ival] = np.percentile(ref_dataset, percentile)

    for ival, model_value in enumerate(model_future):
        percentile = percentileofscore(model_present, model_value)
        model_future_corrected[ival] = np.percentile(ref_dataset, percentile)

    return model_present_corrected, model_future_corrected


def eQM_delta(ref_dataset, model_present, model_future):
    """
    Remove the biases for each quantile value taking the difference between
    ref_dataset and model_present at each percentile as a kind of systematic bias (delta)
    and add them to model_future at the same percentile.

    returns: downscaled model_present and model_future
    """

    model_present_corrected = np.zeros(model_present.size)
    model_future_corrected = np.zeros(model_future.size)

    for ival, model_value in enumerate(model_present):
        percentile = percentileofscore(model_present, model_value)
        model_present_corrected[ival] = np.percentile(ref_dataset, percentile)

    for ival, model_value in enumerate(model_future):
        percentile = percentileofscore(model_future, model_value)
        model_future_corrected[ival] = (
            model_value
            + np.percentile(ref_dataset, percentile)
            - np.percentile(model_present, percentile)
        )

    return model_present_corrected, model_future_corrected
