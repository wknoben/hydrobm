import numpy as np

from .metrics import mse


# Basic optimization routine for lagged precipitation benchmark
def optimize_lag(scaled_precip, streamflow, max_lag=30):
    """Optimize the lag for a lagged precipitation benchmark model.

    Parameters
    ----------
    scaled_precip : pandas Series
        Scaled precipitation data.
    streamflow : pandas Series
        Streamflow data.
    max_lag : int, optional
        Maximum lag to consider. Default is 30.

    Returns
    -------
    best_lag : int
        Best lag value.
    best_mse : float
        Best mean squared error value.

    Notes
    -----
    Equally good as scipy.optimize.minimize_scalar with added rounding, but
    much slower. Keeping as a record of the attempt in case we want to revisit
    the optimization part of adjusted_lagged_precipitation_benchmark().
    """

    # Initialize the best lag and MSE
    best_lag = 0
    best_mse = np.inf

    # Loop over all possible lags
    for lag in range(0, max_lag):
        # Shift the precipitation data by the lag
        shifted_precip = scaled_precip.shift(lag)

        # Calculate the MSE for the shifted precipitation
        mse_val = mse(shifted_precip, streamflow)

        # Update the best lag and MSE
        if mse_val < best_mse:
            best_lag = lag
            best_mse = mse_val

    return best_lag, best_mse


# Basic optimization routine for lagged, smoothed precipitation benchmark
def optimize_aspb(scaled_precip, streamflow, max_lag=30, max_window=90):
    """Optimize the lag for a lagged smoothed precipitation benchmark model.

    Parameters
    ----------
    scaled_precip : pandas Series
        Scaled precipitation data.
    streamflow : pandas Series
        Streamflow data.
    max_lag : int, optional
        Maximum lag to consider. Default is 30.
    max_window: int, optional
        Maximum smoothing window length to consider. Default is 90.

    Returns
    -------
    best_lag : int
        Best lag value.
    best_window: int
        Best window value.
    best_mse : float
        Best mean squared error value.

    Notes
    -----
    Implemented because scipy.optimize.minimize seems to struggle with
    integer-only solutions, and searching docs is hard.
    """

    # Initialize an array to store MSE scores
    all_mse = np.full([max_lag, max_window], np.inf)

    # Loop over all possible lags
    for lag in range(0, max_lag):
        for window in range(1, max_window):  # window=0 is undefined
            # Shift the precipitation data by the lag and smooth
            aspb = scaled_precip.shift(lag).rolling(window=window).mean()

            # Calculate the MSE for the shifted precipitation
            mse_val = mse(aspb, streamflow)
            if not np.isnan(mse_val):  # might get NaN if lag/window > time series
                all_mse[lag, window] = mse_val

    # Find the lowest MSE. If multiple, select smallest lag and window.
    best_lag, best_window = np.unravel_index(all_mse.argmin(), all_mse.shape)
    best_mse = all_mse[best_lag, best_window]

    return best_lag, best_window, best_mse


# Basic snow accumulation and melt model
def rain_to_melt(
    data, precipitation="precipitation", temperature="temperature", snow_and_melt_temp=0.0, snow_and_melt_rate=3.0
):
    """Calculate snow accumulation and melt based on temperature thresholds.

    Parameters
    ----------
    data : pandas DataFrame
        Input data containing precipitation and temperature columns.
    precipitation : str, optional
        Name of the precipitation column in the input data. Default is 'precipitation'.
    temperature : str, optional'
        Name of the temperature column in the input data. Default is 'temperature'.
    snow_and_melt_temp : float, optional
        Temperature threshold for snow accumulation and melt. Default is 0.0 [C].
    snow_and_melt_rate : float, optional
        Snow melt rate if temperature above threshold. Default is 3.0 [mm/hour/degree C].

    Returns
    -------
    data : pandas DataFrame
        Input data with additional columns for snow depth and rain plus melt.
    """

    # Docs: 3 degrees C is a conservative estimate (see e.g.: https://tc.copernicus.org/articles/17/211/2023/)

    # Check that melt rate is not negative
    if snow_and_melt_rate < 0:
        raise ValueError(f"Snow melt rate must be non-negative. Currently set to: {snow_and_melt_rate}.")

    # Run a really simple time-stepping scheme to account for snow accumulation and melt.
    # We'll deal with the time step implicitly, simply assuming that delta t = 1.
    # We can get away with Explicit Euler (Snew = Sold + snowfall - snowmelt) because fall
    # and melt are mutually exclusive: no problems with ad-hoc operator splitting here.
    snow_depth = []
    rain_plus_melt = []
    Sold = 0
    for _, row in data.iterrows():
        # Determine snowfall or melt
        if row[temperature] > snow_and_melt_temp:
            melt = np.min([Sold, snow_and_melt_rate * (row[temperature] - snow_and_melt_temp)])
            rain = row[precipitation]
            snow = 0
        else:
            melt = 0
            rain = 0
            snow = row[precipitation]

        # Update the snow pack
        Snew = Sold + snow - melt

        # Retain the values
        snow_depth.append(Snew)
        rain_plus_melt.append(rain + melt)

        # Prepare for the next time step
        Sold = Snew

    # Outputs
    data["snow_depth"] = snow_depth
    data["rain_plus_melt"] = rain_plus_melt

    return data
