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
    # For some reason np.isnan gets really cranky if you feed it floats.
    # This is a quick hack until I can figure out why this happens.
    obs = np.float64(obs)
    sim = np.float64(sim)
    # One possible solution appears to be to go into the benchmark definitions
    # and replace all the temporary pd.NA values with something like -999 and
    # enforce dtype=np.float64 when creating the dataframe. I'm reluctant to
    # put this in right now because it's not fully clear to me why this issue
    # doesn't crop up during the tests, and only when running the example code.
    # To reproduce, run the camels_01022500_minimal.nc example with metric 'nse'
    # and benchmark 'mean_daily_flow', and comment out the two lines above.

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
