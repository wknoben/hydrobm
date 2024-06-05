import pandas as pd


def create_bm(data, benchmark, cal_mask, precipitation="precipitation", streamflow="streamflow"):
    """Create a benchmark flow for a given benchmark model

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
    ]
    assert benchmark in bm_list, f"Requested benchmark {benchmark} not found."

    # Calculate the requested benchmark
    if benchmark == "mean_flow":
        bm_vals = data[streamflow].loc[cal_mask].mean()  # Returns a single value
        qbm = pd.DataFrame(
            {"bm_mean_flow": bm_vals}, index=data.index
        )  # Use single value as predictor at all time steps

    elif benchmark == "median_flow":
        bm_vals = data[streamflow].loc[cal_mask].median()
        qbm = pd.DataFrame({"bm_median_flow": bm_vals}, index=data.index)

    # Note that we cannot actually use the annual mean flows for prediction on unseen data
    # because the years don't repeat, but perhaps someone has a use for it anyway.
    elif benchmark == "annual_mean_flow":
        bm_vals = data[streamflow].loc[cal_mask].groupby(data.index.year).mean()  # Returns one value per year
        qbm = pd.DataFrame(
            {"bm_annual_mean_flow": pd.NA}, index=data.index
        )  # Initialize an empty dataframe with the right number of time steps
        for year in qbm.index.year.unique():  # TO DO: check if there is a cleaner way to do this
            qbm.loc[qbm.index.year == year, "bm_annual_mean_flow"] = bm_vals[bm_vals.index == year].values

    elif benchmark == "annual_median_flow":
        bm_vals = data[streamflow].loc[cal_mask].groupby(data.index.year).median()
        qbm = pd.DataFrame({"bm_annual_median_flow": pd.NA}, index=data.index)
        for year in qbm.index.year.unique():
            qbm.loc[qbm.index.year == year, "bm_annual_median_flow"] = bm_vals[bm_vals.index == year].values

    elif benchmark == "monthly_mean_flow":
        bm_vals = (
            data[streamflow].loc[cal_mask].groupby(data.index.month).mean()
        )  # Returns one value per month in the index
        qbm = pd.DataFrame({"bm_monthly_mean_flow": pd.NA}, index=data.index)
        for month in qbm.index.month.unique():
            qbm.loc[qbm.index.month == month, "bm_monthly_mean_flow"] = bm_vals[bm_vals.index == month].values

    elif benchmark == "monthly_median_flow":
        bm_vals = data[streamflow].loc[cal_mask].groupby(data.index.month).median()
        qbm = pd.DataFrame({"bm_monthly_median_flow": pd.NA}, index=data.index)
        for month in qbm.index.month.unique():
            qbm.loc[qbm.index.month == month, "bm_monthly_median_flow"] = bm_vals[bm_vals.index == month].values

    elif benchmark == "daily_mean_flow":
        # < TO DO > Better DoY creation, so that dates have consistent numbers. Trial version below.
        # create a day-of-year index with consistent values for each day whether leap year or not
        # data['mmdd'] = data.index.strftime('%m') + data.index.strftime('%d')
        bm_vals = (
            data[streamflow].loc[cal_mask].groupby(data.index.dayofyear).mean()
        )  # Returns onee value per day-of-year
        qbm = pd.DataFrame({"bm_daily_mean_flow": pd.NA}, index=data.index)
        for doy in qbm.index.dayofyear.unique():
            qbm.loc[qbm.index.dayofyear == doy, "bm_daily_mean_flow"] = bm_vals[bm_vals.index == doy].values

    elif benchmark == "daily_median_flow":
        bm_vals = data[streamflow].loc[cal_mask].groupby(data.index.dayofyear).median()
        qbm = pd.DataFrame({"bm_daily_median_flow": pd.NA}, index=data.index)
        for doy in qbm.index.dayofyear.unique():
            qbm.loc[qbm.index.dayofyear == doy, "bm_daily_median_flow"] = bm_vals[bm_vals.index == doy].values

    # End of benchmark definitions
    # Return the benchmark values and benchmark time series for the calculation period
    return bm_vals, qbm


def evaluate_bm(data, benchmark_flow, metric, cal_mask, val_mask):
    cal_score = 0
    val_score = 0
    return cal_score, val_score
