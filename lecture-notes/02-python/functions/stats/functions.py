# Imports
import numpy as np


# Custom function that calculates the Neyman statistic
def neyman_stat(array1, array2):
    """
    Calculate the Neyman statistic.

    Parameters
    ----------
    array1 : array-like
        Observed outcomes of the treatment group.
    array2 : array-like
        Observed outcomes of the control group.

    Returns
    -------
    float
        The Neyman statistic.
    """

    # Calculate means
    mean_array1 = np.mean(array1)
    mean_array2 = np.mean(array2)

    # Calculate the denominator
    sdev = np.sqrt(
        np.var(array1) / len(array1)
        + np.var(array2) / len(array2)
    )

    # Return the actual statistic
    return (mean_array1 - mean_array2) / sdev
