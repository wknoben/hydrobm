import pandas as pd
import xarray as xr

from .benchmarks import create_bm, evaluate_bm
from .utils import rain_to_melt


# Main function to calculate metric scores for a given set of benchmark models
def calc_bm(
    data,
    # Time period selection
    cal_mask,
    val_mask=[],
    # Variable names in 'data'
    precipitation="precipitation",
    streamflow="streamflow",
    # Benchmark choices
    benchmarks=["daily_mean_flow"],
    metrics=["rmse"],
    optimization_method="brute_force",
    # Snow model inputs
    calc_snowmelt=False,
    temperature="temperature",
    snowmelt_threshold=0.0,
    snowmelt_rate=3.0,
):
    """Calculate benchmark model scores for a given set of benchmark models and metrics.

    Parameters
    ----------
    data : pandas DataFrame or xarray Dataset
        Input data containing precipitation and streamflow columns.
    cal_mask : pandas Series
        Boolean mask for the calculation period.
    val_mask : pandas Series, optional
        Boolean mask for the validation period. Default is [] (no validation scores returned).
    precipitation : str, optional
        Name of the precipitation column in the input data. Default is 'precipitation'.
    streamflow : str, optional
        Name of the streamflow column in the input data. Default is 'streamflow'.
    benchmarks : list, optional
        List of benchmark models to calculate. Default is ['daily_mean_flow'].
    metrics : list, optional
        List of metrics to calculate. Default is ['rmse'].
    optimization_method : str, optional
        Optimization method to use for benchmark model calibration. Default is 'brute_force'.
    calc_snowmelt : bool, optional
        Flag to run a basic snow accumulation and melt model. Default is False.
    temperature : str, optional
        Name of the temperature column in the input data. Default is 'temperature'.
    snowmelt_threshold : float, optional
        Threshold temperature for snowmelt calculation. Default is 0.0 [C].

    Returns
    -------
    benchmark_flows : pandas DataFrame
        DataFrame containing benchmark flows for each benchmark model.
    metrics : dict
        Dictionary containing metric scores for each benchmark model.
    """

    # Input handling: if xarray dataset is provided, convert to DataFrame
    if isinstance(data, xr.Dataset):
        data = data.to_dataframe()

    # Input check: data is a pandas DataFrame.
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame or xarray Dataset")

    # Input check: precipitation and streamflow columns are present in the DataFrame
    if precipitation not in data.columns:
        raise ValueError(f"Precipitation column {precipitation} not found in the input data")
    if streamflow not in data.columns:
        raise ValueError(f"Streamflow column {streamflow} not found in the input data")

    # Input check: DataFrame has a datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Input data must have a datetime index")

    # Input check: calibration and validation masks same length as the data index
    if len(cal_mask) != len(data.index):
        raise ValueError("Benchmark calculation mask does not match length of data index")
    if len(val_mask) != len(data.index):
        raise ValueError("benchmark evaluation mask does not match length of data index")

    # Run a basic snow model if requested
    if calc_snowmelt:
        # Input check: temperature column is present in input data
        if temperature not in data.columns:
            raise ValueError(f"Temperature column {temperature} not found in the input data")
        # Calculate snowmelt
        data = rain_to_melt(
            data,
            precipitation=precipitation,
            temperature=temperature,
            snow_and_melt_temp=snowmelt_threshold,
            snow_and_melt_rate=snowmelt_rate,
        )
        # Update the precipitation variable to instead use rain_plus_melt
        precipitation = "rain_plus_melt"

    # First create the benchmark flows as a one-off
    benchmark_flow_list = (
        []
    )  # list to store DataFrames of benchmark flows, merged later if multiple benchmarks are requested
    for benchmark in benchmarks:
        _, qbm = create_bm(
            data,
            benchmark,
            cal_mask,
            precipitation=precipitation,
            streamflow=streamflow,
            optimization_method=optimization_method,
        )  # Create the benchmark flow for calibration period
        benchmark_flow_list.append(qbm)

    # Then loop over the metrics to calculate scores for each benchmark flow
    results = {"benchmarks": benchmarks}  # dictionary to store metric scores; tracks the benchmark models used
    for metric in metrics:
        cal_scores = []
        val_scores = []
        for benchmark_flow in benchmark_flow_list:
            [cal_score, val_score] = evaluate_bm(
                data, benchmark_flow, metric, cal_mask, val_mask=val_mask, streamflow=streamflow
            )
            cal_scores.append(cal_score)
            val_scores.append(val_score)
        results.update({metric + "_cal": cal_scores, metric + "_val": val_scores})

    return pd.concat(benchmark_flow_list, axis=1), results
