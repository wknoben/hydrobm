import numpy as np
from scipy.optimize import Bounds, minimize, minimize_scalar  # differential_evolution,

from .metrics import mse

# --- Wrapper functions for adjusted precipitation benchmark
# and adjusted smoothed precipitation benchmark optimization


# Wrapper for adjusted precipitation benchmark optimization
def optimize_apb(scaled_precip, streamflow, method, max_lag=30):
    """Wrapper function around adjusted precipitation benchmark model optimization functions.

    Parameters
    ----------
    scaled_precip : pandas Series
        Scaled precipitation data.
    streamflow : pandas Series
        Streamflow data.
    method : str
        Optimization method to use. Currently supports "brute_force" and "minimize".
    max_lag : int, optional
        Maximum lag to consider. Default is 30.

    Returns
    -------
    best_lag : int
        Best lag value.
    best_mse : float
        Best mean squared error value.
    """

    # Check that the method is valid
    if method not in ["brute_force", "minimize"]:  # , "differential_evolution"]:
        raise ValueError(f"Invalid optimization method specified for optimize_lag: {method}.")

    # Brute force optimization
    if method == "brute_force":
        best_lag, best_mse = brute_force_apb(scaled_precip, streamflow, max_lag)

    # Minimize scalar optimization
    elif method == "minimize":
        best_lag, best_mse = minimize_scalar_apb(scaled_precip, streamflow, max_lag)

    # # Differential evolution optimization
    # elif method == "differential_evolution":
    #     best_lag, best_mse = differential_evolution_apb(scaled_precip, streamflow, max_lag)

    return best_lag, best_mse


# Wrapper for adjusted smoothed precipitation benchmark optimization
def optimize_aspb(scaled_precip, streamflow, method, max_lag=30, max_window=90):
    """Wrapper function around adjusted smoothed precipitation benchmark model optimization functions.

    Parameters
    ----------
    scaled_precip : pandas Series
        Scaled precipitation data.
    streamflow : pandas Series
        Streamflow data.
    method : str
        Optimization method to use. Currently supports "brute_force" and "minimize".
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
    """

    # Check that the method is valid
    if method not in ["brute_force", "minimize"]:
        raise ValueError(f"Invalid optimization method specified for optimize_lag: {method}.")

    # Brute force optimization
    if method == "brute_force":
        best_lag, best_window, best_mse = brute_force_aspb(scaled_precip, streamflow, max_lag, max_window)

    # Minimize scalar optimization
    elif method == "minimize":
        best_lag, best_window, best_mse = minimize_aspb(scaled_precip, streamflow, max_lag, max_window)

    return best_lag, best_window, best_mse


# --- Optimization functions for adjusted precipitation benchmark (APB)


# Basic brute force optimization routine for adjusted precipitation benchmark
def brute_force_apb(scaled_precip, streamflow, max_lag=30):
    """Optimize the lag for the adjusted precipitation benchmark model using brute force.

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


# minimize_scalar optimization routine for adjusted precipitation benchmark
def minimize_scalar_apb(scaled_precip, streamflow, max_lag=30):
    """Optimize the lag for the adjusted precipitation benchmark model using scipy.optimize.minimize_scalar.

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
    scipy.optimize.minimize_scalar is not designed for use with integer-only solutions. Here we
    use the round function to enforce integer solutions. This seems to work for simple test cases,
    but results for real data may vary. User caution is advised. Use brute force optimization if
    100% accurate solutions are required.
    """

    # Define the optimization function
    def mse_apb(lag):
        lag = round(lag)  # ensures integer
        apb = scaled_precip.shift(lag)
        return mse(apb, streamflow)

    # Run the optimization
    bounds = (0, max_lag - 1)  # minimize_scalar only accepts bounds as a tuple, not as Bounds class
    res = minimize_scalar(mse_apb, bounds=bounds, method="bounded")

    # Extract the best lag and MSE
    best_lag = round(res.x)
    best_mse = res.fun

    return best_lag, best_mse


# # differential_evolution optimization routine for adjusted precipitation benchmark
# def differential_evolution_apb(scaled_precip, streamflow, max_lag=30):
#     # Define the optimization function
#     def mse_apb(lag):
#         apb = scaled_precip.shift(lag)  # lag as integer enforced elsewhere
#         return mse(apb, streamflow)

#     # Custom integer mutation function
#     def integer_mutation(xk, **kwargs):
#         xk_new = np.round(xk).astype(int)
#         return xk_new

#     # Run the optimization
#     bounds = Bounds(0, max_lag - 1)  # Bounds([lower], [upper])
#     bounds = [(0, max_lag - 1)]
#     res = differential_evolution(
#         mse,
#         bounds,
#         args=(obs,),
#         strategy="best1bin",
#         mutation=(0.5, 1),
#         recombination=0.7,
#         tol=0.01,
#         seed=42,
#         workers=1,
#         updating="deferred",
#         disp=True,
#         polish=False,
#         init="random",
#         callback=integer_mutation,
#     )

#     return "Not implemented yet"


# --- Optimization functions for adjusted smoothed precipitation benchmark (ASPB)


# Basic brute force optimization routine for adjusted smoothed precipitation benchmark
def brute_force_aspb(scaled_precip, streamflow, max_lag=30, max_window=90):
    """Optimize the lag and window for adjusted smoothed precipitation benchmark model using brute force.

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

    # Find the lowest MSE. If multiple, by default selects the smallest lag and window.
    best_lag, best_window = np.unravel_index(all_mse.argmin(), all_mse.shape)
    best_mse = all_mse[best_lag, best_window]

    return best_lag, best_window, best_mse


# minimize optimization routine for adjusted precipitation benchmark
def minimize_aspb(scaled_precip, streamflow, max_lag=30, max_window=90, method="Powell"):
    """Optimize the lag and window for the ASPB model using scipy.optimize.minimize.

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
    method: str, optional
        Optimization method to use. Default is 'Powell'. See scipy.optimize.minimize for more options.

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
    scipy.optimize.minimize is not designed for use with integer-only solutions. Here we
    use the round function to enforce integer solutions. The 'Powell' optimization method
    seems to return appropriate lag and window values in simple test cases, but results
    for real data may vary. User caution is advised. Use brute force optimization if 100%
    accurate solutions are required.
    """

    # Define the optimization function
    def mse_aspb(params):
        lag, window = params
        lag = round(lag)  # ensures integer
        window = round(window)

        # Calculate the adjusted smoothed precipitation benchmark
        aspb = scaled_precip.shift(lag).rolling(window=window).mean()
        return mse(aspb, streamflow)

    # Run the optimization
    init = [0, 1]  # initial guess for lag and window
    bounds = Bounds([0, 1], [max_lag - 1, max_window - 1])
    res = minimize(mse_aspb, init, bounds=bounds, method=method)

    # Extract the best lag, window, and MSE
    best_lag, best_window = res.x.round()
    best_mse = res.fun

    return int(best_lag), int(best_window), best_mse


# --- Snow accumulation and melt model


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
        Snow melt rate if temperature above threshold. Default is 3.0 [mm/timestep/degree C].

    Returns
    -------
    data : pandas DataFrame
        Input data with additional columns for snow depth and rain plus melt.

    Notes
    -----
    The default values for snow_and_melt_temp and snow_and_melt_rate are given in units of
    degrees Celsius and millimeters per time step per degree Celsius, respectively. These
    are not used in the code however, as the function is designed to work with any units.

    For example, providing the input data in Kelvin and setting snow_and_melt_temp to 273.15
    will work as expected. Similarly, if the input precipitation data is not in millimeters,
    simply providing the snow_and_melt_rate in those same units will yield the correct output.
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
