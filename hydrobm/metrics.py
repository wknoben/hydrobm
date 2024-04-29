import numpy as np


def nse(obs, sim):
    """Calculate Nash-Sutcliffe efficiency.

    Parameters
    ----------
    obs : array-like
        Observed values.
    sim : array-like
        Simulated values.

    Returns
    -------
    float
        Nash-Sutcliffe efficiency.
    """
    return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)


def rmse(obs, sim):
    """Calculate root mean square error.

    Parameters
    ----------
    obs : array-like
        Observed values.
    sim : array-like
        Simulated values.

    Returns
    -------
    float
        Root mean square error.
    """
    return np.sqrt(np.mean((obs - sim) ** 2))


def kge(obs, sim):
    """Calculate Kling-Gupta efficiency.

    Parameters
    ----------
    obs : array-like
        Observed values.
    sim : array-like
        Simulated values.

    Returns
    -------
    float
        Kling-Gupta efficiency.
    """
    # Account for the case where sim has zero variance
    if np.std(sim) == 0:
        r = 0
    else:
        r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
