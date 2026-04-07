import numpy as np
import pandas as pd
from scipy.optimize import Bounds, minimize, minimize_scalar  # differential_evolution,

from .metrics import kge, mse

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


# --- Eckhardt Baseflow utility functions


# Parameter estimation for Eckhardt
def estimate_eckhardt_parameters(streamflow, precip, precip_window_days=3, precip_threshold=0.1):
    """
    Estimate both recession coefficient (k) and maximum baseflow index (BFI_max) which are required for
    baseflow separation as outlined by Eckhardt (2005).

    This function combines recession analysis to estimate k with the backward filter
    method from Collischonn & Fan (2013) to estimate BFI_max. Automatically detects
    the timestep from the data and adjusts time window accordingly.

    Parameters
    ----------
    streamflow : pandas Series
        Observed streamflow with DatetimeIndex.
    precip : pandas Series
        Precipitation data with DatetimeIndex.
    precip_window_days : float, optional
        Number of DAYS to check for precipitation when identifying recessions.
        Automatically converted to appropriate number of timesteps based on data frequency.
        Default is 3 days.
    precip_threshold : float, optional
        Precipitation threshold in same units as precip data (e.g., mm/day or kg/m²/s).
        Default is 0.1.

    Returns
    -------
    k : float
        Recession coefficient estimated from recession periods (for native timestep).
    BFI_max : float
        Maximum baseflow index estimated using backward filter method.

    References
    ----------
    Eckhardt, K. (2005). How to construct recursive digital filters for baseflow separation.
    Hydrological Processes, 19(2), 507-515.

    Collischonn, W., & Fan, F. M. (2013). Defining parameters for Eckhardt's digital baseflow filter.
    Hydrological Processes, 27(18), 2614–2622. https://doi.org/10.1002/hyp.9391
    """
    # Detect timestep
    time_diff = streamflow.index.to_series().diff()
    median_timestep = time_diff.median()

    if isinstance(median_timestep, pd.Timedelta):
        delta_t_days = median_timestep.total_seconds() / 86400
    else:
        delta_t_days = float(median_timestep)

    # Convert precipitation window from days to number of periods
    precip_window_periods = max(1, int(np.ceil(precip_window_days / delta_t_days)))

    # ===========================
    # Step 1: Estimate k from recession periods
    # ===========================
    k_values = []

    # Identify periods with no precipitation
    no_precip = precip.rolling(window=precip_window_periods, min_periods=1).sum() < precip_threshold

    # Flow is declining
    declining = streamflow.diff() < 0

    # Combined condition: recession period
    recession_mask = no_precip & declining

    # Calculate k for recession days: k = Q[t] / Q[t-1]
    for t in range(1, len(streamflow)):
        if recession_mask.iloc[t] and streamflow.iloc[t] > 0 and streamflow.iloc[t - 1] > 0:
            k_t = streamflow.iloc[t] / streamflow.iloc[t - 1]
            k_values.append(k_t)

    # Calculate the final value of k
    k = np.median(k_values)

    # ===========================
    # Step 2: Estimate BFI_max using backward filter
    # ===========================

    Q_values = streamflow.values
    n = len(Q_values)

    # Initialize backward baseflow array
    baseflow_backward = np.zeros(n)

    # Start from the end: assume last timestep is all baseflow
    baseflow_backward[-1] = Q_values[-1]

    # Apply backward filter: b'[t] = b'[t+1] / k
    for t in range(n - 2, -1, -1):  # Go backward from second-to-last to first
        if Q_values[t] > 0:
            baseflow_backward[t] = baseflow_backward[t + 1] / k
            # Baseflow cannot exceed total flow
            baseflow_backward[t] = min(baseflow_backward[t], Q_values[t])
        else:
            baseflow_backward[t] = 0

    # BFI_max is the ratio of total baseflow to total streamflow (Equation 11)
    BFI_max = np.sum(baseflow_backward) / np.sum(Q_values)

    return k, BFI_max


# Calculation function for Eckhardt
def eckhardt_filter(Q, BFI_max, k):
    """
    Eckhardt two-parameter digital filter for baseflow separation.

    The Eckhardt filter was found to be the best of 9 evaluated baseflow
    separation methods in Xie et al. (2020), showing superior performance
    across diverse catchment conditions.

    Parameters
    ----------
    Q : pandas Series or numpy array
        Streamflow time series.
    BFI_max : float
        Maximum baseflow index.
    k : float
        Recession constant.

    Returns
    -------
    baseflow : pandas Series or numpy array
        Separated baseflow component.

    References
    ----------
    Eckhardt, K. (2005). How to construct recursive digital filters for baseflow separation.
    Hydrological Processes, 19(2), 507-515.

    Xie, J., Liu, X., Wang, K., Yang, T., Liang, K., & Liu, C. (2020). Evaluation of typical
    methods for baseflow separation in the contiguous United States. Journal of Hydrology, 583,
    124628. https://doi.org/10.1016/j.jhydrol.2020.124628

    """

    # Check input type and extract values
    is_series = isinstance(Q, pd.Series)
    if is_series:
        Q_values = Q.values
        index = Q.index
    else:
        Q_values = np.asarray(Q)

    baseflow = np.zeros(len(Q_values))
    baseflow[0] = Q_values[0] * BFI_max  # Initialize with fraction of first flow

    # Apply Eckhardt filter
    for t in range(1, len(Q_values)):
        if Q_values[t] > 0:
            baseflow[t] = ((1 - BFI_max) * k * baseflow[t - 1] + (1 - k) * BFI_max * Q_values[t]) / (
                1 - k * BFI_max
            )
            # Baseflow cannot exceed total flow
            baseflow[t] = min(baseflow[t], Q_values[t])
        else:
            baseflow[t] = 0

    # Return in same format as input
    if is_series:
        return pd.Series(baseflow, index=index)
    else:
        return baseflow


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


# --- BME calculation functions


def bme_nse(q_obs, q_sim, q_bm, cal_mask, val_mask=None):
    """Calculate NSE-based Benchmark Efficiency (BME) for cal and val periods. The formulation
    can be found in Seibert (2001) and Schaefli and Gupta (2007).

    BME = 1 - sum((q_obs - q_sim)^2) / sum((q_obs - q_bm)^2)

    Parameters
    ----------
    q_obs : pandas Series
        Observed streamflow.
    q_sim : pandas Series
        Simulated streamflow.
    q_bm : pandas Series
        Benchmark streamflow.
    cal_mask : pandas Series
        Boolean mask for the calibration period.
    val_mask : pandas Series, optional
        Boolean mask for the validation period. Default is None (no val score returned).

    Returns
    -------
    cal_score : float
        NSE-based BME score for the calibration period.
    val_score : float
        NSE-based BME score for the validation period. NaN if no val_mask specified.


    References
    ----------
    Seibert, J. (2001). On the need for benchmarks in hydrological modelling.
    Hydrological Processes, 15(6), 1063–1064. https://doi.org/10.1002/hyp.446

    Schaefli, B., & Gupta, H. V. (2007). Do Nash values have value?
    Hydrological Processes, 21(15), 2075–2080. https://doi.org/10.1002/hyp.6825


    """

    def _nse_bme_score(mask):
        obs = q_obs[mask].values
        sim = q_sim[mask].values
        bm = q_bm[mask].values

        # filter nan for 3 arrays
        if np.all(np.isnan(obs)) or np.all(np.isnan(sim)) or np.all(np.isnan(bm)):
            return np.nan
        nan_mask = ~np.isnan(obs) & ~np.isnan(sim) & ~np.isnan(bm)
        obs, sim, bm = obs[nan_mask], sim[nan_mask], bm[nan_mask]

        denominator = np.sum((obs - bm) ** 2)
        if denominator == 0:
            return np.nan
        return 1 - np.sum((obs - sim) ** 2) / denominator

    cal_score = _nse_bme_score(cal_mask)
    val_score = _nse_bme_score(val_mask) if val_mask is not None else np.nan
    return cal_score, val_score


def bme_kge(q_obs, q_sim, q_bm, cal_mask, val_mask=None):
    """Calculate KGE-based Benchmark Model Efficiency (KGE skill score) for cal and val periods.
    This skill score formulation can be found in Knoben et al. (2019) among others.

    KGE_skill = (KGE_model - KGE_benchmark) / (1 - KGE_benchmark)

    Parameters
    ----------
    q_obs : pandas Series
        Observed streamflow.
    q_sim : pandas Series
        Simulated streamflow.
    q_bm : pandas Series
        Benchmark streamflow.
    cal_mask : pandas Series
        Boolean mask for the calibration period.
    val_mask : pandas Series, optional
        Boolean mask for the validation period. Default is None (no val score returned).

    Returns
    -------
    cal_score : float
        KGE skill score for the calibration period.
    val_score : float
        KGE skill score for the validation period. NaN if no val_mask specified.

    References
    ----------
    Knoben, W. J. M., Freer, J. E., & Woods, R. A. (2019). Technical note: Inherent benchmark or not? Comparing
    Nash–Sutcliffe and Kling–Gupta efficiency scores. Hydrology and Earth System Sciences, 23(10), 4323–4331.
    https://doi.org/10.5194/hess-23-4323-2019

    """

    def _bme_kge_score(mask):
        obs = q_obs[mask].values
        sim = q_sim[mask].values
        bm = q_bm[mask].values
        kge_model = kge(obs, sim)
        kge_benchmark = kge(obs, bm)
        if np.isclose(kge_benchmark, 1.0):
            return np.nan
        return (kge_model - kge_benchmark) / (1 - kge_benchmark)

    cal_score = _bme_kge_score(cal_mask)
    val_score = _bme_kge_score(val_mask) if val_mask is not None else np.nan
    return cal_score, val_score
