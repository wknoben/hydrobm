import numpy as np


def filter_nan(obs, sim):
    """Select only non-NaN values from both observed and simulated data.

    Parameters
    ----------
    obs : array-like
        Observed values.
    sim : array-like
        Simulated values.

    Returns
    -------
    tuple
        Tuple of arrays with non-NaN values from both observed and simulated data.
    """

    # Catch the case where we have all NaNs.
    # Rare, but might happen for short timeseries and the lagged benchmarks.
    if np.all(np.isnan(obs)) or np.all(np.isnan(sim)):
        return np.nan, np.nan

    mask = ~np.isnan(obs) & ~np.isnan(sim)
    return obs[mask], sim[mask]


def nse(obs, sim, ignore_nan=True):
    """Calculate Nash-Sutcliffe efficiency.

    Parameters
    ----------
    obs : array-like
        Observed values.
    sim : array-like
        Simulated values.
    ignore_nan : bool, optional
        Flag to consider only non-NaN values. Default is True.

    Returns
    -------
    float
        Nash-Sutcliffe efficiency.
    """
    if ignore_nan:
        obs, sim = filter_nan(obs, sim)
    return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)


def mse(obs, sim, ignore_nan=True):
    """Calculate mean square error.

    Parameters
    ----------
    obs : array-like
        Observed values.
    sim : array-like
        Simulated values.
    ignore_nan : bool, optional
        Flag to consider only non-NaN values. Default is True.

    Returns
    -------
    float
        Mean square error.
    """
    if ignore_nan:
        obs, sim = filter_nan(obs, sim)
    return np.mean((obs - sim) ** 2)


def rmse(obs, sim, ignore_nan=True):
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
    if ignore_nan:
        obs, sim = filter_nan(obs, sim)
    return np.sqrt(np.mean((obs - sim) ** 2))


def kge(obs, sim, ignore_nan=True):
    """Calculate Kling-Gupta efficiency.

    Parameters
    ----------
    obs : array-like
        Observed values.
    sim : array-like
        Simulated values.
    ignore_nan : bool, optional
        Flag to consider only non-NaN values. Default is True.

    Returns
    -------
    float
        Kling-Gupta efficiency.
    """
    if ignore_nan:
        obs, sim = filter_nan(obs, sim)

    # Account for the case where obs has zero variance
    if (np.std(obs) == 0) or (np.std(sim) == 0):
        r = 0
    else:
        r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def calculate_metric(obs, sim, metric, ignore_nan=True):
    """Helper function to check metric existence and simplify loops.

    Parameters
    ----------
    obs : array-like
        Observed values.
    sim : array-like
        Simulated values.
    metric: str
        Name of the metric to calculate.
    ignore_nan : bool, optional
        Flag to consider only non-NaN values. Default is True.

    Returns
    -------
    float
        Metric score.
    """

    # List of currently implemented metrics
    metric_list = [
        "nse",
        "mse",
        "rmse",
        "kge",
    ]

    assert metric in metric_list, f"Requested metric {metric} not found."

    if metric == "nse":
        val = nse(obs, sim, ignore_nan=ignore_nan)

    elif metric == "mse":
        val = mse(obs, sim, ignore_nan=ignore_nan)

    elif metric == "rmse":
        val = rmse(obs, sim, ignore_nan=ignore_nan)

    elif metric == "kge":
        val = kge(obs, sim, ignore_nan=ignore_nan)

    return val
