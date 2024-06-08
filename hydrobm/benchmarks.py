import pandas as pd

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
    years don't repeat.
    """
    cal_set = data[streamflow].loc[cal_mask]
    bm_vals = cal_set.groupby(cal_set.index.year).mean()  # Returns one value per year
    qbm = pd.DataFrame(
        {"bm_annual_mean_flow": pd.NA}, index=cal_set.index
    )  # Initialize an empty dataframe with the right number of time steps
    for year in qbm.index.year.unique():  # TO DO: check if there is a cleaner way to do this
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
    This benchmark  cannot be used to predict unseen data, because the
    years don't repeat.
    """

    cal_set = data[streamflow].loc[cal_mask]
    bm_vals = cal_set.groupby(cal_set.index.year).median()
    qbm = pd.DataFrame({"bm_annual_median_flow": pd.NA}, index=cal_set.index)
    for year in qbm.index.year.unique():
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

    bm_vals = (
        data[streamflow].loc[cal_mask].groupby(data.index.month).mean()
    )  # Returns one value per month in the index
    qbm = pd.DataFrame({"bm_monthly_mean_flow": pd.NA}, index=data.index)
    for month in qbm.index.month.unique():
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

    bm_vals = data[streamflow].loc[cal_mask].groupby(data.index.month).median()
    qbm = pd.DataFrame({"bm_monthly_median_flow": pd.NA}, index=data.index)
    for month in qbm.index.month.unique():
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
    bm_vals = (
        data[streamflow].loc[cal_mask].groupby(data.index.dayofyear).mean()
    )  # Returns one value per day-of-year
    qbm = pd.DataFrame({"bm_daily_mean_flow": pd.NA}, index=data.index)
    for doy in qbm.index.dayofyear.unique():
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

    bm_vals = data[streamflow].loc[cal_mask].groupby(data.index.dayofyear).median()
    qbm = pd.DataFrame({"bm_daily_median_flow": pd.NA}, index=data.index)
    for doy in qbm.index.dayofyear.unique():
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
    qbm = pd.DataFrame({"bm_rainfall_runoff_ratio_to_all": pd.NA}, index=data.index)
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
        Annual rainfall-runoff ratio values for the calculation period.
    qbm : pandas DataFrame
        Benchmark flow time series for the annual rainfall-runoff ratio (RRR) benchmark model.
        Computed as long-term RRR multiplied by annual mean precipitation.
    """

    cal_set = data.loc[cal_mask]
    bm_vals = cal_set[streamflow].sum() / cal_set[precipitation].sum()  # single rainfall-runoff ratio
    qbm = pd.DataFrame({"bm_rainfall_runoff_ratio_to_annual": pd.NA}, index=data.index)
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
        Monthly rainfall-runoff ratio values for the calculation period.
    qbm : pandas DataFrame
        Benchmark flow time series for the monthly rainfall-runoff ratio (RRR) benchmark model.
        Computed as long-term RRR multiplied by monthly mean precipitation.
    """

    cal_set = data.loc[cal_mask]
    bm_vals = cal_set[streamflow].sum() / cal_set[precipitation].sum()  # single rainfall-runoff ratio
    qbm = pd.DataFrame({"bm_rainfall_runoff_ratio_to_monthly": pd.NA}, index=data.index)
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


# --- Benchmark creation and evaluation ---


def create_bm(data, benchmark, cal_mask, precipitation="precipitation", streamflow="streamflow"):
    """Helper function to call the correct benchmark model function

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

    Returns
    -------
    bm_values: pandas DataSeries
        Benchmark values for the given benchmark model.
    qbm : pandas DataFrame
        Benchmark flow time series for the given benchmark model.
    """

    # < TO DO >: Update/complete the list
    bm_list = [
        "mean_flow",
        "median_flow",
        "annual_mean_flow",
        "annual_median_flow",
        "monthly_mean_flow",
        "monthly_median_flow",
        "daily_mean_flow",
        "daily_median_flow",
        "rainfall_runoff_ratio_to_all",
        "rainfall_runoff_ratio_to_annual",
        "rainfall_runoff_ratio_to_monthly",
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

    # End of benchmark definitions
    # Return the benchmark values and benchmark time series for the calculation period
    return bm_vals, qbm


def evaluate_bm(data, benchmark_flow, metric, cal_mask, val_mask):
    cal_score = 0
    val_score = 0
    return cal_score, val_score
