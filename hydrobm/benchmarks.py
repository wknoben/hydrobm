import numpy as np
import pandas as pd

from .metrics import calculate_metric
from .utils import optimize_apb, optimize_aspb

# from scipy.optimize import Bounds, minimize, minimize_scalar


# --- benchmark definitions ---
# --- Benchmarks relying on streamflow only ---
# These benchmarks all calculate a typical flow value/regime using
# data from the calculation period and use that as a predictor for
# all timesteps in the whole dataframe.


def bm_mean_flow(data, cal_mask, streamflow="streamflow"):
    """Calculate the mean flow over the calculation period and
    use that as a predictor for all timesteps in the whole dataframe.

    Parameters
    ----------
    data : pandas DataFrame
        Input data containing streamflow column.
        cal_mask : pandas Series
        Boolean mask for the calculation period.
    streamflow : str, optional
        Name of the streamflow column in the input data. Default is ['streamflow'].

    Returns
    -------
    bm_vals: float
        Mean flow value for the calculation period.
    qbm : pandas DataFrame
        Benchmark flow time series for the mean flow benchmark model.
    """

    bm_vals = data[streamflow].loc[cal_mask].mean()  # Returns a single value
    qbm = pd.DataFrame(
        {"bm_mean_flow": bm_vals}, index=data.index
    )  # Use single value as predictor at all time steps
    return bm_vals, qbm


def bm_median_flow(data, cal_mask, streamflow="streamflow"):
    """Calculate the median flow over the calculation period and
    use that as a predictor for all timesteps in the whole dataframe.

    Parameters
    ----------
    data : pandas DataFrame
        Input data containing streamflow column.
    cal_mask : pandas Series
        Boolean mask for the calculation period.
    streamflow : str, optional
        Name of the streamflow column in the input data. Default is ['streamflow'].

    Returns
    -------
    bm_vals: float
        Median flow value for the calculation period.
    qbm : pandas DataFrame
        Benchmark flow time series for the median flow benchmark model.
    """

    bm_vals = data[streamflow].loc[cal_mask].median()
    qbm = pd.DataFrame({"bm_median_flow": bm_vals}, index=data.index)
    return bm_vals, qbm


def bm_annual_mean_flow(data, cal_mask, streamflow="streamflow"):
    """Calculate the annual mean flow over the calculation period and
    use that as a predictor for each year in the calculation period.

    Parameters
    ----------
    data : pandas DataFrame
        Input data containing streamflow column.
    cal_mask : pandas Series
        Boolean mask for the calculation period.
    streamflow : str, optional
        Name of the streamflow column in the input data. Default is ['streamflow'].

    Returns
    -------
    bm_vals: pandas DataSeries
        Annual mean flow values for the calculation period.
    qbm : pandas DataFrame
        Benchmark flow time series for the annual mean flow benchmark model.

    Notes
    -----
    This benchmark cannot be used to predict unseen data, because the
    years don't repeat. This function will return a ebnchmark time series
    that has the same length as the input data, but only the calculation
    period will have values. The rest will be NaNs.
    """
    cal_set = data[streamflow].loc[cal_mask]
    bm_vals = cal_set.groupby(cal_set.index.year).mean()  # Returns one value per year

    # Initialize an empty dataframe for the full time series, even if we can only
    # calculate this particular benchmark for the calculation period. That way we
    # can at least guarantee that the output has the right shape. Evaluation period
    # will be all NaNs.
    qbm = pd.DataFrame({"bm_annual_mean_flow": np.nan}, index=data.index)
    for year in bm_vals.index:
        qbm.loc[qbm.index.year == year, "bm_annual_mean_flow"] = bm_vals[bm_vals.index == year].values
    return bm_vals, qbm


def bm_annual_median_flow(data, cal_mask, streamflow="streamflow"):
    """Calculate the annual median flow over the calculation period and
    use that as a predictor for each year in the calculation period.

    Parameters
    ----------
    data : pandas DataFrame
        Input data containing streamflow column.
    cal_mask : pandas Series
        Boolean mask for the calculation period.
    streamflow : str, optional
        Name of the streamflow column in the input data. Default is ['streamflow'].

    Returns
    -------
    bm_vals: pandas DataSeries
        Annual median flow values for the calculation period.
    qbm : pandas DataFrame
        Benchmark flow time series for the annual median flow benchmark model.

    Notes
    -----
    This benchmark cannot be used to predict unseen data, because the
    years don't repeat. This function will return a ebnchmark time series
    that has the same length as the input data, but only the calculation
    period will have values. The rest will be NaNs.
    """

    cal_set = data[streamflow].loc[cal_mask]
    bm_vals = cal_set.groupby(cal_set.index.year).median()

    # Initialize an empty dataframe for the full time series, even if we can only
    # calculate this particular benchmark for the calculation period. That way we
    # can at least guarantee that the output has the right shape. Evaluation period
    # will be all NaNs.
    qbm = pd.DataFrame({"bm_annual_median_flow": np.nan}, index=data.index)
    for year in bm_vals.index:
        qbm.loc[qbm.index.year == year, "bm_annual_median_flow"] = bm_vals[bm_vals.index == year].values
    return bm_vals, qbm


def bm_monthly_mean_flow(data, cal_mask, streamflow="streamflow"):
    """Calculate the monthly mean flow over the calculation period and
    use that as a predictor for each month in the whole dataframe.

    Parameters
    ----------
    data : pandas DataFrame
        Input data containing streamflow column.
    cal_mask : pandas Series
        Boolean mask for the calculation period.
    streamflow : str, optional
        Name of the streamflow column in the input data. Default is ['streamflow'].

    Returns
    -------
    bm_vals: pandas DataSeries
        Monthly mean flow values for the calculation period.
    qbm : pandas DataFrame
        Benchmark flow time series for the monthly mean flow benchmark model.
    """

    cal_set = data[streamflow].loc[cal_mask]
    bm_vals = cal_set.groupby(cal_set.index.month).mean()  # Returns one value per month in the index
    qbm = pd.DataFrame({"bm_monthly_mean_flow": np.nan}, index=data.index)
    for month in qbm.index.month.unique():
        if month in bm_vals.index:  # takes care of cases where for some reason we have no cal data for this month
            qbm.loc[qbm.index.month == month, "bm_monthly_mean_flow"] = bm_vals[bm_vals.index == month].values
    return bm_vals, qbm


def bm_monthly_median_flow(data, cal_mask, streamflow="streamflow"):
    """Calculate the monthly median flow over the calculation period and
    use that as a predictor for each month in the whole dataframe.

    Parameters
    ----------
    data : pandas DataFrame
        Input data containing streamflow column.
    cal_mask : pandas Series
        Boolean mask for the calculation period.
    streamflow : str, optional
        Name of the streamflow column in the input data. Default is ['streamflow'].

    Returns
    -------
    bm_vals: pandas DataSeries
        Monthly median flow values for the calculation period.
    qbm : pandas DataFrame
        Benchmark flow time series for the monthly median flow benchmark model.
    """

    cal_set = data[streamflow].loc[cal_mask]
    bm_vals = cal_set.groupby(cal_set.index.month).median()
    qbm = pd.DataFrame({"bm_monthly_median_flow": np.nan}, index=data.index)
    for month in qbm.index.month.unique():
        if month in bm_vals.index:  # takes care of cases where for some reason we have no cal data for this month
            qbm.loc[qbm.index.month == month, "bm_monthly_median_flow"] = bm_vals[bm_vals.index == month].values
    return bm_vals, qbm


def bm_daily_mean_flow(data, cal_mask, streamflow="streamflow"):
    """Calculate the daily mean flow over the calculation period and
    use that as a predictor for each day in the whole dataframe.

    Parameters
    ----------
    data : pandas DataFrame
        Input data containing streamflow column.
    cal_mask : pandas Series
        Boolean mask for the calculation period.
    streamflow : str, optional
        Name of the streamflow column in the input data. Default is ['streamflow'].

    Returns
    -------
    bm_vals: pandas DataSeries
        Daily mean flow values for the calculation period.
    qbm : pandas DataFrame
        Benchmark flow time series for the daily mean flow benchmark model.
    """

    # < TO DO > Better DoY creation, so that dates have consistent numbers. Trial version below.
    # create a day-of-year index with consistent values for each day whether leap year or not
    # data['mmdd'] = data.index.strftime('%m') + data.index.strftime('%d')
    cal_set = data[streamflow].loc[cal_mask]
    bm_vals = cal_set.groupby(cal_set.index.dayofyear).mean()  # Returns one value per day-of-year
    qbm = pd.DataFrame({"bm_daily_mean_flow": np.nan}, index=data.index)
    for doy in qbm.index.dayofyear.unique():
        if doy in bm_vals.index:  # takes care of cases where for some reason we have no cal data for this day
            qbm.loc[qbm.index.dayofyear == doy, "bm_daily_mean_flow"] = bm_vals[bm_vals.index == doy].values
    return bm_vals, qbm


def bm_daily_median_flow(data, cal_mask, streamflow="streamflow"):
    """Calculate the daily median flow over the calculation period and
    use that as a predictor for each day in the whole dataframe.

    Parameters
    ----------
    data : pandas DataFrame
        Input data containing streamflow column.
    cal_mask : pandas Series
        Boolean mask for the calculation period.
    streamflow : str, optional
        Name of the streamflow column in the input data. Default is ['streamflow'].

    Returns
    -------
    bm_vals: pandas DataSeries
        Daily median flow values for the calculation period.
    qbm : pandas DataFrame
        Benchmark flow time series for the daily median flow benchmark model.
    """

    cal_set = data[streamflow].loc[cal_mask]
    bm_vals = cal_set.groupby(cal_set.index.dayofyear).median()
    qbm = pd.DataFrame({"bm_daily_median_flow": np.nan}, index=data.index)
    for doy in qbm.index.dayofyear.unique():
        if doy in bm_vals.index:  # takes care of cases where for some reason we have no cal data for this day
            qbm.loc[qbm.index.dayofyear == doy, "bm_daily_median_flow"] = bm_vals[bm_vals.index == doy].values
    return bm_vals, qbm


# --- Benchmarks relying on precipitation and streamflow ---


def bm_rainfall_runoff_ratio_to_all(data, cal_mask, precipitation="precipitation", streamflow="streamflow"):
    """Calculate the long-term rainfall-runoff ratio over the calculation period and
    use that as a predictor of runoff using precipitation totals from the calculation
    period and non-calculation period respectively.

    Parameters
    ----------
    data : pandas DataFrame
        Input data containing precipitation and streamflow columns.
    cal_mask : pandas Series
        Boolean mask for the calculation period.
    precipitation : str, optional
        Name of the precipitation column in the input data. Default is ['precipitation'].
    streamflow : str, optional
        Name of the streamflow column in the input data. Default is ['streamflow'].

    Returns
    -------
    bm_vals: float
        Rainfall-runoff ratio value for the calculation period.
    qbm : pandas DataFrame
        Benchmark flow time series for the rainfall-runoff ratio (RRR) benchmark model.
        Computed as long-term RRR during calculation period multiplied by long-term
        mean precipitation from calculation and non-calculation periods.

    Notes
    -----
    Effectively the same as the mean flow benchmark.
    """

    cal_set = data.loc[cal_mask]
    bm_vals = cal_set[streamflow].sum() / cal_set[precipitation].sum()  # single rainfall-runoff ratio
    qbm = pd.DataFrame({"bm_rainfall_runoff_ratio_to_all": np.nan}, index=data.index)
    qbm.loc[cal_mask] = bm_vals * data[precipitation].loc[cal_mask].mean()
    if len(cal_set.index) != len(
        data.index
    ):  # This prevents an error for the case where the calculation period is the whole dataset
        qbm.loc[~cal_mask] = bm_vals * data[precipitation].loc[~cal_mask].mean()
    return bm_vals, qbm


def bm_rainfall_runoff_ratio_to_annual(data, cal_mask, precipitation="precipitation", streamflow="streamflow"):
    """Calculate the long-term rainfall-runoff ratio over the calculation period and
    use that as a predictor of runoff-from-precipitation for each year in the whole dataframe.

    Parameters
    ----------
    data : pandas DataFrame
        Input data containing precipitation and streamflow columns.
    cal_mask : pandas Series
        Boolean mask for the calculation period.
    precipitation : str, optional
        Name of the precipitation column in the input data. Default is ['precipitation'].
    streamflow : str, optional
        Name of the streamflow column in the input data. Default is ['streamflow'].

    Returns
    -------
    bm_vals: pandas DataSeries
        Rainfall-runoff ratio value for the calculation period.
    qbm : pandas DataFrame
        Benchmark flow time series for the annual rainfall-runoff ratio (RRR) benchmark model.
        Computed as long-term RRR multiplied by annual mean precipitation.
    """

    cal_set = data.loc[cal_mask]
    bm_vals = cal_set[streamflow].sum() / cal_set[precipitation].sum()  # single rainfall-runoff ratio
    qbm = pd.DataFrame({"bm_rainfall_runoff_ratio_to_annual": np.nan}, index=data.index)
    for year in qbm.index.year.unique():
        mean_annual_precip = data[precipitation].loc[data.index.year == year].mean()
        qbm.loc[qbm.index.year == year, "bm_rainfall_runoff_ratio_to_annual"] = bm_vals * mean_annual_precip
    return bm_vals, qbm


def bm_rainfall_runoff_ratio_to_monthly(data, cal_mask, precipitation="precipitation", streamflow="streamflow"):
    """Calculate the long-term rainfall-runoff ratio over the calculation period and
    use that as a predictor of runoff-from-precipitation for each month in the whole dataframe.

    Parameters
    ----------
    data : pandas DataFrame
        Input data containing precipitation and streamflow columns.
    cal_mask : pandas Series
        Boolean mask for the calculation period.
    precipitation : str, optional
        Name of the precipitation column in the input data. Default is ['precipitation'].
    streamflow : str, optional
        Name of the streamflow column in the input data. Default is ['streamflow'].

    Returns
    -------
    bm_vals: pandas DataSeries
        Rainfall-runoff ratio value for the calculation period.
    qbm : pandas DataFrame
        Benchmark flow time series for the monthly rainfall-runoff ratio (RRR) benchmark model.
        Computed as long-term RRR multiplied by monthly mean precipitation.
    """

    cal_set = data.loc[cal_mask]
    bm_vals = cal_set[streamflow].sum() / cal_set[precipitation].sum()  # single rainfall-runoff ratio
    qbm = pd.DataFrame({"bm_rainfall_runoff_ratio_to_monthly": np.nan}, index=data.index)
    for year in qbm.index.year.unique():  # for each year
        for month in qbm.loc[
            qbm.index.year == year
        ].index.month.unique():  # for each month we have in the index for that year (takes care of mising months)
            mean_monthly_precip = (
                data[precipitation].loc[(data.index.year == year) & (data.index.month == month)].mean()
            )
            qbm.loc[
                (qbm.index.year == year) & (qbm.index.month == month), "bm_rainfall_runoff_ratio_to_monthly"
            ] = (bm_vals * mean_monthly_precip)

    return bm_vals, qbm


def bm_rainfall_runoff_ratio_to_daily(data, cal_mask, precipitation="precipitation", streamflow="streamflow"):
    """Calculate the long-term rainfall-runoff ratio over the calculation period and
    use that as a predictor of runoff-from-precipitation for each day in the whole dataframe.

    Parameters
    ----------
    data : pandas DataFrame
        Input data containing precipitation and streamflow columns.
    cal_mask : pandas Series
        Boolean mask for the calculation period.
    precipitation : str, optional
        Name of the precipitation column in the input data. Default is ['precipitation'].
    streamflow : str, optional
        Name of the streamflow column in the input data. Default is ['streamflow'].

    Returns
    -------
    bm_vals: pandas DataSeries
        Rainfall-runoff ratio value for the calculation period.
    qbm : pandas DataFrame
        Benchmark flow time series for the daily rainfall-runoff ratio (RRR) benchmark model.
        Computed as long-term RRR multiplied by daily mean precipitation.
    """

    cal_set = data.loc[cal_mask]
    bm_vals = cal_set[streamflow].sum() / cal_set[precipitation].sum()  # single rainfall-runoff ratio
    qbm = pd.DataFrame({"bm_rainfall_runoff_ratio_to_daily": np.nan}, index=data.index)
    for year in qbm.index.year.unique():  # for each year
        for doy in qbm.loc[
            qbm.index.year == year
        ].index.dayofyear.unique():  # for each DoY in index for this year (takes care of mising days)
            this_day = (data.index.year == year) & (data.index.dayofyear == doy)
            mean_daily_precip = data[precipitation].loc[this_day].mean()
            qbm.loc[this_day, "bm_rainfall_runoff_ratio_to_daily"] = bm_vals * mean_daily_precip

    return bm_vals, qbm


def bm_rainfall_runoff_ratio_to_timestep(data, cal_mask, precipitation="precipitation", streamflow="streamflow"):
    """Calculate the long-term rainfall-runoff ratio over the calculation period and
    use that as a predictor of runoff-from-precipitation for each timestep in the whole dataframe.

    Parameters
    ----------
    data : pandas DataFrame
        Input data containing precipitation and streamflow columns.
    cal_mask : pandas Series
        Boolean mask for the calculation period.
    precipitation : str, optional
        Name of the precipitation column in the input data. Default is ['precipitation'].
    streamflow : str, optional
        Name of the streamflow column in the input data. Default is ['streamflow'].

    Returns
    -------
    bm_vals: float
        Rainfall-runoff ratio value for the calculation period.
    qbm : pandas DataFrame
        Benchmark flow time series for the rainfall-runoff ratio (RRR) benchmark model.
        Computed as long-term RRR multiplied by precipitation at each timestep.
    """

    cal_set = data.loc[cal_mask]
    bm_vals = cal_set[streamflow].sum() / cal_set[precipitation].sum()  # single rainfall-runoff ratio
    qbm = pd.DataFrame({"bm_rainfall_runoff_ratio_to_timestep": bm_vals * data[precipitation]}, index=data.index)
    return bm_vals, qbm


def monthly_rainfall_runoff_ratio_to_monthly(
    data, cal_mask, precipitation="precipitation", streamflow="streamflow"
):
    """Calculate the mean monthly rainfall-runoff ratio over the calculation period and
    use that as a predictor of runoff-from-precipitation for each month in the whole dataframe.

    Parameters
    ----------
    data : pandas DataFrame
        Input data containing precipitation and streamflow columns.
    cal_mask : pandas Series
        Boolean mask for the calculation period.
    precipitation : str, optional
        Name of the precipitation column in the input data. Default is ['precipitation'].
    streamflow : str, optional
        Name of the streamflow column in the input data. Default is ['streamflow'].

    Returns
    -------
    bm_vals: pandas DataSeries
        Monthly rainfall-runoff ratio values for the calculation period.
    qbm : pandas DataFrame
        Benchmark flow time series for the monthly rainfall-runoff ratio (RRR) benchmark model.
        Computed as mean monthly RRR multiplied by monthly mean precipitation.
    """

    cal_set = data.loc[cal_mask]
    monthly_mean_q = cal_set[streamflow].groupby(cal_set.index.month).mean()
    monthly_mean_p = cal_set[precipitation].groupby(cal_set.index.month).mean()
    bm_vals = monthly_mean_q / monthly_mean_p  # (at most) 12 rainfall-runoff ratios
    bm_vals = bm_vals.reindex(range(1, 13))  # fill missing months with NaN, does nothing if already 12
    qbm = pd.DataFrame({"bm_monthly_rainfall_runoff_ratio_to_monthly": np.nan}, index=data.index)
    for year in qbm.index.year.unique():  # for each year
        for month in qbm.loc[
            qbm.index.year == year
        ].index.month.unique():  # for each month we have in the index for that year (takes care of mising months)
            this_month = (data.index.year == year) & (data.index.month == month)
            mean_monthly_precip = data[precipitation].loc[this_month].mean()
            qbm.loc[this_month, "bm_monthly_rainfall_runoff_ratio_to_monthly"] = (
                bm_vals.loc[month] * mean_monthly_precip
            )
    return bm_vals, qbm


def monthly_rainfall_runoff_ratio_to_daily(data, cal_mask, precipitation="precipitation", streamflow="streamflow"):
    """Calculate the mean monthly rainfall-runoff ratio over the calculation period and
    use that as a predictor of runoff-from-precipitation for each day in the whole dataframe.

    Parameters
    ----------
    data : pandas DataFrame
        Input data containing precipitation and streamflow columns.
    cal_mask : pandas Series
        Boolean mask for the calculation period.
    precipitation : str, optional
        Name of the precipitation column in the input data. Default is ['precipitation'].
    streamflow : str, optional
        Name of the streamflow column in the input data. Default is ['streamflow'].

    Returns
    -------
    bm_vals: pandas DataSeries
        Monthly rainfall-runoff ratio values for the calculation period.
    qbm : pandas DataFrame
        Benchmark flow time series for the monthly rainfall-runoff ratio (RRR) benchmark model.
        Computed as mean monthly RRR multiplied by daily mean precipitation.
    """

    cal_set = data.loc[cal_mask]
    monthly_mean_q = cal_set[streamflow].groupby(cal_set.index.month).mean()
    monthly_mean_p = cal_set[precipitation].groupby(cal_set.index.month).mean()
    bm_vals = monthly_mean_q / monthly_mean_p  # (at most) 12 rainfall-runoff ratios
    bm_vals = bm_vals.reindex(range(1, 13))  # fill missing months with NaN, does nothing if already 12
    qbm = pd.DataFrame({"bm_monthly_rainfall_runoff_ratio_to_daily": np.nan}, index=data.index)
    for year in qbm.index.year.unique():  # for each year
        for doy in qbm.loc[
            qbm.index.year == year
        ].index.dayofyear.unique():  # for each DoY in index for this year (takes care of mising days)
            this_day = (data.index.year == year) & (data.index.dayofyear == doy)
            mean_daily_precip = data[precipitation].loc[this_day].mean()
            month = data[precipitation].loc[this_day].index.month[0]
            qbm.loc[this_day, "bm_monthly_rainfall_runoff_ratio_to_daily"] = bm_vals.loc[month] * mean_daily_precip
    return bm_vals, qbm


def monthly_rainfall_runoff_ratio_to_timestep(
    data, cal_mask, precipitation="precipitation", streamflow="streamflow"
):
    """Calculate the mean monthly rainfall-runoff ratio over the calculation period and
    use that as a predictor of runoff-from-precipitation for each timestep in the whole dataframe.

    Parameters
    ----------
    data : pandas DataFrame
        Input data containing precipitation and streamflow columns.
    cal_mask : pandas Series
        Boolean mask for the calculation period.
    precipitation : str, optional
        Name of the precipitation column in the input data. Default is ['precipitation'].
    streamflow : str, optional
        Name of the streamflow column in the input data. Default is ['streamflow'].

    Returns
    -------
    bm_vals: pandas DataSeries
        Monthly rainfall-runoff ratio values for the calculation period.
    qbm : pandas DataFrame
        Benchmark flow time series for the monthly rainfall-runoff ratio (RRR) benchmark model.
        Computed as mean monthly RRR multiplied by precipitation at each timestep.
    """

    cal_set = data.loc[cal_mask]
    monthly_mean_q = cal_set[streamflow].groupby(cal_set.index.month).mean()
    monthly_mean_p = cal_set[precipitation].groupby(cal_set.index.month).mean()
    bm_vals = monthly_mean_q / monthly_mean_p  # (at most) 12 rainfall-runoff ratios
    bm_vals = bm_vals.reindex(range(1, 13))  # fill missing months with NaN, does nothing if already 12
    qbm = pd.DataFrame(
        {
            "bm_monthly_rainfall_runoff_ratio_to_timestep": bm_vals.loc[data.index.month].values
            * data[precipitation].values
        },
        index=data.index,
    )

    return bm_vals, qbm


def scaled_precipitation_benchmark(data, cal_mask, precipitation="precipitation", streamflow="streamflow"):
    """Calculate the scaled precipitation benchmark model as a predictor
    of runoff-from-precipitation for each timestep in the whole dataframe.

    Parameters
    ----------
    data : pandas DataFrame
        Input data containing precipitation and streamflow columns.
    cal_mask : pandas Series
        Boolean mask for the calculation period.
    precipitation : str, optional
        Name of the precipitation column in the input data. Default is ['precipitation'].
    streamflow : str, optional
        Name of the streamflow column in the input data. Default is ['streamflow'].

    Returns
    -------
    bm_vals: float
        Rainfall-runoff ratio value for the calculation period.
    qbm : pandas DataFrame
        Benchmark flow time series for the scaled precipitation benchmark model.
        Computed as long-term RRR multiplied by precipitation at each timestep.

    Notes
    -----
    This benchmark is effectively the same as the rainfall-runoff ratio to
    timestep benchmark, though Schaefli & Gupta (2007) apply this with daily
    data only.

    References
    ----------
    Schaefli, B. and Gupta, H.V. (2007), Do Nash values have value?.
    Hydrol. Process., 21: 2075-2080. https://doi.org/10.1002/hyp.6825
    """

    bm_vals, qbm = bm_rainfall_runoff_ratio_to_timestep(data, cal_mask, precipitation, streamflow)
    qbm = qbm.rename(
        columns={"bm_rainfall_runoff_ratio_to_timestep": "bm_scaled_precipitation_benchmark"}
    )  # Rename column to match function name
    return bm_vals, qbm


def adjusted_precipitation_benchmark(
    data, cal_mask, precipitation="precipitation", streamflow="streamflow", optimization_method="brute_force"
):
    """Calculate the adjusted precipitation benchmark model as a predictor
    of runoff-from-precipitation for each timestep in the whole dataframe.

    Parameters
    ----------
    data : pandas DataFrame
        Input data containing precipitation and streamflow columns.
    cal_mask : pandas Series
        Boolean mask for the calculation period.
    precipitation : str, optional
        Name of the precipitation column in the input data. Default is ['precipitation'].
    streamflow : str, optional
        Name of the streamflow column in the input data. Default is ['streamflow'].
    optimize_method : str, optional
        Optimization method to use. Default is ['brute_force']. See optimize_apb() for further options.

    Returns
    -------
    bm_vals: tuple
        Rainfall-runoff ratio value for the calculation period and the optimized lag value.
    qbm : pandas DataFrame
        Benchmark flow time series for the adjusted precipitation benchmark model.
        Computed as long-term RRR multiplied by precipitation at each timestep,
        lagged for a number of timesteps that minimizes MSE (Schaefli & Gupta, 2007).

    References
    ----------
    Schaefli, B. and Gupta, H.V. (2007), Do Nash values have value?.
    Hydrol. Process., 21: 2075-2080. https://doi.org/10.1002/hyp.6825
    """

    # Calculate the rainfall-scaling
    cal_set = data.loc[cal_mask]
    bm_vals = cal_set[streamflow].sum() / cal_set[precipitation].sum()  # single rainfall-runoff ratio

    # Find the maximum lag value to avoid shifting the data too much (1 year max seems reasonable)
    max_lag = data.groupby(data.index.year).size().iloc[0]  # group by year and get the size of the first group

    # Optimize the lag by minimizing MSE between observed and predicted streamflow
    lag, _ = optimize_apb(
        bm_vals * cal_set[precipitation], cal_set[streamflow], optimization_method, max_lag=max_lag
    )

    # Calculate the adjusted precipitation benchmark
    qbm = pd.DataFrame(
        {"bm_adjusted_precipitation_benchmark": bm_vals * data[precipitation].shift(lag)}, index=data.index
    )

    # Add the lag as an output
    bm_vals = (bm_vals, lag)

    return bm_vals, qbm


def adjusted_smoothed_precipitation_benchmark(
    data, cal_mask, precipitation="precipitation", streamflow="streamflow", optimization_method="brute_force"
):
    """Calculate the adjusted smoothed precipitation benchmark model as a predictor
    of runoff-from-precipitation for each timestep in the whole dataframe.

    Parameters
    ----------
    data : pandas DataFrame
        Input data containing precipitation and streamflow columns.
    cal_mask : pandas Series
        Boolean mask for the calculation period.
    precipitation : str, optional
        Name of the precipitation column in the input data. Default is ['precipitation'].
    streamflow : str, optional
        Name of the streamflow column in the input data. Default is ['streamflow'].
    optimization_method : str, optional
        Optimization method to use. Default is ['brute_force']. See optimize_aspb() for further options.

    Returns
    -------
    bm_vals: tuple
        Rainfall-runoff ratio value for the calculation period and the optimized lag and window values.
    qbm : pandas DataFrame
        Benchmark flow time series for the adjusted smoothed precipitation benchmark model.
        Computed as long-term RRR multiplied by precipitation at each timestep,
        lagged for a number of timesteps and smoothed with a moving average filter (Schaefli & Gupta, 2007).

    References
    ----------
    Schaefli, B. and Gupta, H.V. (2007), Do Nash values have value?.
    Hydrol. Process., 21: 2075-2080. https://doi.org/10.1002/hyp.6825
    """

    # Calculate the rainfall-scaling
    cal_set = data.loc[cal_mask]
    bm_vals = cal_set[streamflow].sum() / cal_set[precipitation].sum()  # single rainfall-runoff ratio

    # Find the maximum lag value to avoid shifting the data too much (1 year max seems reasonable)
    timesteps_per_year = (
        data.groupby(data.index.year).size().iloc[0]
    )  # group by year and get the size of the first group

    # Optimize the lag and window by minimizing MSE between observed and predicted streamflow
    lag, window, _ = optimize_aspb(
        bm_vals * cal_set[precipitation],
        cal_set[streamflow],
        optimization_method,
        max_lag=timesteps_per_year - 1,
        max_window=timesteps_per_year - 1,
    )

    # Calculate the adjusted smoothed precipitation benchmark
    qbm = pd.DataFrame(
        {
            "bm_adjusted_smoothed_precipitation_benchmark": bm_vals
            * data[precipitation].shift(lag).rolling(window=window).mean()
        },
        index=data.index,
    )

    # Add the lag and window as an output
    bm_vals = (bm_vals, lag, window)

    return bm_vals, qbm

    """
        import numpy as np
        from scipy.optimize import differential_evolution

        # Example function fn. Replace this with your actual function.
        def fn(lag, window):
            # Example implementation (replace with actual function logic)
            return np.sin(lag) + np.cos(window)

        # Observations (replace with your actual observations)
        obs = np.array([1.0, 2.0, 3.0, 4.0])

        # Define the MSE function
        def mse(params, obs):
            lag, window = params
            predictions = fn(lag, window)
            return np.mean((predictions - obs) ** 2)

        # Custom integer mutation function
        def integer_mutation(xk, **kwargs):
            xk_new = np.round(xk).astype(int)
            return xk_new

        # Bounds for lag and window (set appropriate bounds)
        bounds = [(0, 10), (0, 10)]  # Example bounds; adjust as needed

        # Perform the optimization
        result = differential_evolution(mse,
                                        bounds,
                                        args=(obs,),
                                        strategy='best1bin',
                                        mutation=(0.5, 1),
                                        recombination=0.7,
                                        tol=0.01,
                                        seed=42,
                                        workers=1,
                                        updating='deferred',
                                        disp=True,
                                        polish=False,
                                        init='random',
                                        callback=integer_mutation)

        # Output the result
        optimal_lag = int(result.x[0])
        optimal_window = int(result.x[1])
    """


# --- Benchmark creation and evaluation ---


def create_bm(
    data,
    benchmark,
    cal_mask,
    precipitation="precipitation",
    streamflow="streamflow",
    optimization_method="brute_force",
):
    """Helper function to call the correct benchmark model function;
    makes looping over benchmark models easier.

    Parameters
    ----------
    data : pandas DataFrame
        Input data containing precipitation and streamflow columns.
    benchmark : str
        Benchmark model to calculate.
    cal_mask : pandas Series
        Boolean mask for the calculation period.
    precipitation : str, optional
        Name of the precipitation column in the input data. Default is ['precipitation'].
    streamflow : str, optional
        Name of the streamflow column in the input data. Default is ['streamflow'].
    optimization_method : str, optional
        Optimization method to create adjusted (snoothed) precipitation benchmark. Default is ['brute_force'].

    Returns
    -------
    bm_values: pandas DataSeries
        Benchmark values for the given benchmark model.
    qbm : pandas DataFrame
        Benchmark flow time series for the given benchmark model.
    """

    # List of currently implemented benchmark models
    bm_list = [
        # Streamflow benchmarks
        "mean_flow",
        "median_flow",
        "annual_mean_flow",
        "annual_median_flow",
        "monthly_mean_flow",
        "monthly_median_flow",
        "daily_mean_flow",
        "daily_median_flow",
        # Long-term rainfall-runoff ratio benchmarks
        "rainfall_runoff_ratio_to_all",
        "rainfall_runoff_ratio_to_annual",
        "rainfall_runoff_ratio_to_monthly",
        "rainfall_runoff_ratio_to_daily",
        "rainfall_runoff_ratio_to_timestep",
        # Short-term rainfall-runoff ratio benchmarks
        "monthly_rainfall_runoff_ratio_to_monthly",
        "monthly_rainfall_runoff_ratio_to_daily",
        "monthly_rainfall_runoff_ratio_to_timestep",
        # Schaefli & Gupta (2007) benchmarks
        "scaled_precipitation_benchmark",  # equivalent to "rainfall_runoff_ratio_to_daily"
        "adjusted_precipitation_benchmark",
        "adjusted_smoothed_precipitation_benchmark",
    ]
    assert benchmark in bm_list, f"Requested benchmark {benchmark} not found."

    # --- Benchmarks relying on streamflow only

    if benchmark == "mean_flow":
        bm_vals, qbm = bm_mean_flow(data, cal_mask, streamflow=streamflow)

    elif benchmark == "median_flow":
        bm_vals, qbm = bm_median_flow(data, cal_mask, streamflow=streamflow)

    elif benchmark == "annual_mean_flow":
        print(
            f"WARNING: the {benchmark} benchmark cannot be used to predict unseen data. See docstring for details."
        )
        bm_vals, qbm = bm_annual_mean_flow(data, cal_mask, streamflow=streamflow)

    elif benchmark == "annual_median_flow":
        print(
            f"WARNING: the {benchmark} benchmark cannot be used to predict unseen data. See docstring for details."
        )
        bm_vals, qbm = bm_annual_median_flow(data, cal_mask, streamflow=streamflow)

    elif benchmark == "monthly_mean_flow":
        bm_vals, qbm = bm_monthly_mean_flow(data, cal_mask, streamflow=streamflow)

    elif benchmark == "monthly_median_flow":
        bm_vals, qbm = bm_monthly_median_flow(data, cal_mask, streamflow=streamflow)

    elif benchmark == "daily_mean_flow":
        bm_vals, qbm = bm_daily_mean_flow(data, cal_mask, streamflow=streamflow)

    elif benchmark == "daily_median_flow":
        bm_vals, qbm = bm_daily_median_flow(data, cal_mask, streamflow=streamflow)

    # --- Benchmarks relying on precipitation and streamflow

    elif benchmark == "rainfall_runoff_ratio_to_all":
        bm_vals, qbm = bm_rainfall_runoff_ratio_to_all(
            data, cal_mask, precipitation=precipitation, streamflow=streamflow
        )

    # Also hard to use for prediction, but could be useful for comparison
    elif benchmark == "rainfall_runoff_ratio_to_annual":
        bm_vals, qbm = bm_rainfall_runoff_ratio_to_annual(
            data, cal_mask, precipitation=precipitation, streamflow=streamflow
        )

    elif benchmark == "rainfall_runoff_ratio_to_monthly":
        bm_vals, qbm = bm_rainfall_runoff_ratio_to_monthly(
            data, cal_mask, precipitation=precipitation, streamflow=streamflow
        )

    # Equivalent to Schaefli & Gupta's (2007) simple benchmark
    # (adjusted precipitation, no lag and smoothing)
    elif benchmark == "rainfall_runoff_ratio_to_daily":
        bm_vals, qbm = bm_rainfall_runoff_ratio_to_daily(
            data, cal_mask, precipitation=precipitation, streamflow=streamflow
        )

    elif benchmark == "rainfall_runoff_ratio_to_timestep":
        bm_vals, qbm = bm_rainfall_runoff_ratio_to_timestep(
            data, cal_mask, precipitation=precipitation, streamflow=streamflow
        )

    elif benchmark == "monthly_rainfall_runoff_ratio_to_monthly":
        bm_vals, qbm = monthly_rainfall_runoff_ratio_to_monthly(
            data, cal_mask, precipitation=precipitation, streamflow=streamflow
        )

    elif benchmark == "monthly_rainfall_runoff_ratio_to_daily":
        bm_vals, qbm = monthly_rainfall_runoff_ratio_to_daily(
            data, cal_mask, precipitation=precipitation, streamflow=streamflow
        )

    elif benchmark == "monthly_rainfall_runoff_ratio_to_timestep":
        bm_vals, qbm = monthly_rainfall_runoff_ratio_to_timestep(
            data, cal_mask, precipitation=precipitation, streamflow=streamflow
        )

    # --- Schaefli & Gupta (2007) benchmarks

    elif benchmark == "scaled_precipitation_benchmark":
        bm_vals, qbm = scaled_precipitation_benchmark(
            data, cal_mask, precipitation=precipitation, streamflow=streamflow
        )

    elif benchmark == "adjusted_precipitation_benchmark":
        bm_vals, qbm = adjusted_precipitation_benchmark(
            data,
            cal_mask,
            precipitation=precipitation,
            streamflow=streamflow,
            optimization_method=optimization_method,
        )

    elif benchmark == "adjusted_smoothed_precipitation_benchmark":
        bm_vals, qbm = adjusted_smoothed_precipitation_benchmark(
            data,
            cal_mask,
            precipitation=precipitation,
            streamflow=streamflow,
            optimization_method=optimization_method,
        )

    # End of benchmark definitions
    # Return the benchmark values and benchmark time series for the calculation period
    return bm_vals, qbm


def evaluate_bm(data, benchmark_flow, metric, cal_mask, val_mask=None, streamflow="streamflow", ignore_nan=True):
    """Helper function to calculate calculation and evaluation metric scores for a given
    set of observations and benchmark flows.

    Parameters
    ----------
    data : pandas DataFrame
        Input data containing streamflow observation column.
    benchmark_flow : pandas DataFrame
        Benchmark flow time series as returned by one of the benchmark model functions.
    metric : str
        Name of the metric to calculate. See hydrobm/metrics for a list.
    cal_mask : pandas Series
        Boolean mask for the calculation period.
    val_mask : pandas Series, optional
        Boolean mask for the evaluation period. Default is None (no evaluation score returned).
    streamflow : str, optional
        Name of the streamflow column in the input data. Default is ['streamflow'].
    ignore_nan : bool, optional
        Flag to consider only non-NaN values. Default is True.

    Returns
    -------
    cal_score: float
        Metric score for the calculation period.
    val_score: float
        Metric score for the evaluation period. NaN if no val_mask specified.
    """

    # Catch

    # Compute the metric for the calculation period
    cal_obs = data[streamflow].loc[cal_mask]
    cal_sim = benchmark_flow.loc[cal_mask]  # should have only one column
    assert (
        cal_obs.index == cal_sim.index
    ).all(), "Time index mismatch in metric calculation for calculation period"
    cal_score = calculate_metric(cal_obs.values.flatten(), cal_sim.values.flatten(), metric, ignore_nan=ignore_nan)

    # Calculate the evaluation score if a mask is provided
    val_score = np.nan
    if val_mask is not None:
        val_obs = data[streamflow].loc[val_mask]
        val_sim = benchmark_flow.loc[val_mask]
        assert (
            val_obs.index == val_sim.index
        ).all(), "Time index mismatch in metric calculation for evaluation period"
        val_score = calculate_metric(
            val_obs.values.flatten(), val_sim.values.flatten(), metric, ignore_nan=ignore_nan
        )

    return cal_score, val_score
