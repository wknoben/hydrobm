# from hydrobm.benchmarks import create_bm # manual import from top-level folder only
import numpy as np
import pandas as pd
import pytest

from ..benchmarks import create_bm, evaluate_bm


def create_sines(period=2, mean_p=2, mean_q=1, var_p=1, var_q=1, offset_p=0, offset_q=0):
    """Create two sine curve time series for testing (precip, flow)."""
    hour_per_year = 365 * 24  # 365 days/year * 24 hours/day
    n_steps = period * hour_per_year
    dates = pd.date_range(
        "2001-01-01", periods=n_steps, freq="H"
    )  # Start in 2001 so we avoid the leap year in 2000
    # Sine curve parameters
    data_p = mean_p + var_p * np.sin((np.arange(n_steps) - offset_p) / hour_per_year * (2 * np.pi))
    data_q = mean_q + var_q * np.sin((np.arange(n_steps) - offset_q) / hour_per_year * (2 * np.pi))
    # DataFrame for benchmark calculation inputs
    data = pd.DataFrame(
        {"precipitation": data_p, "streamflow": data_q},
        index=dates,
    )
    return data


def test_mean_flow():
    # Get the testing data
    mean_q = 1
    data = create_sines(mean_q=mean_q)

    # T1: 1 year: should return mean_q values for every time step
    cal_mask = data.index.year == 2001
    bm_v, bm_t = create_bm(data, "mean_flow", cal_mask)
    assert bm_v == mean_q, "Failed T1a."
    assert (bm_t["bm_mean_flow"] == mean_q).all(), "Failed mean flow T1b."

    # T2: Should return values > mean_q for the first half a year of data
    cal_mask = (data.index.year == 2001) & (data.index.month < 7)
    bm_v, bm_t = create_bm(data, "mean_flow", cal_mask)
    assert bm_v > mean_q, "Failed T2a."
    assert (bm_t["bm_mean_flow"] > mean_q).all(), "Failed mean flow T2b."

    # T3: Should return vales < mean_q for the second half a year of data
    cal_mask = (data.index.year == 2001) & (data.index.month >= 7)
    bm_v, bm_t = create_bm(data, "mean_flow", cal_mask)
    assert bm_v < mean_q, "Failed T3a."
    assert (bm_t["bm_mean_flow"] < mean_q).all(), "Failed mean flow T3b."


def test_annual_mean_flow():
    # Get the testing data
    mean_q = 1
    data = create_sines(mean_q=mean_q)

    # Double the mean flow for year 2
    year2_mask = data.index.year == 2002
    data.loc[year2_mask, "streamflow"] *= 2

    # T1: should return all mean_q values for year 1, and mean_q * 2 values for year 2
    cal_mask = data.index  # all data
    bm_v, bm_t = create_bm(data, "annual_mean_flow", cal_mask)
    assert (bm_v == [mean_q, 2 * mean_q]).all(), "Failed annual mean flow T1a."
    assert (bm_t[bm_t.index.year == 2001]["bm_annual_mean_flow"] == mean_q).all(), "Failed annual mean flow T1b."
    assert (
        bm_t[bm_t.index.year == 2002]["bm_annual_mean_flow"] == 2 * mean_q
    ).all(), "Failed annual mean flow T1c."


def test_monthly_mean_flow():
    # Get the testing data
    data = create_sines()

    # T1: should return different values for every month, but the same values within each month
    cal_mask = data.index  # all data
    bm_v, bm_t = create_bm(data, "monthly_mean_flow", cal_mask)
    assert len(bm_v.unique()) == 12, "Failed monthly mean flow T1a."
    assert all(bm_t.groupby(bm_t.index.month).nunique() == 1), "Failed monthly mean flow T1b."

    # T2: should return different values for every month, but the same values within each month
    cal_mask = data.index.year == 2001  # all data
    bm_v, bm_t = create_bm(data, "monthly_mean_flow", cal_mask)
    assert len(bm_v.unique()) == 12, "Failed monthly mean flow T2a."
    assert all(bm_t.groupby(bm_t.index.month).nunique() == 1), "Failed monthly mean flow T2b."


def test_daily_mean_flow():
    # Get the testing data
    data = create_sines()

    # T1: should return different values for every day, but the same values within each day
    cal_mask = data.index  # all data
    bm_v, bm_t = create_bm(data, "daily_mean_flow", cal_mask)
    assert len(bm_v.unique()) == 365, "Failed daily mean flow T1a."
    assert all(bm_t.groupby(bm_t.index.dayofyear).nunique() == 1), "Failed daily mean flow T1b."

    # T2: should return different values for every day, but the same values within each day
    cal_mask = data.index.year == 2001  # all data
    bm_v, bm_t = create_bm(data, "daily_mean_flow", cal_mask)
    assert len(bm_v.unique()) == 365, "Failed daily mean flow T2a."
    assert all(bm_t.groupby(bm_t.index.dayofyear).nunique() == 1), "Failed daily mean flow T2b."


def test_rainfall_runoff_ratio_to_all():
    # Get the testing data
    data = create_sines(period=3)

    # Test 1 year to see if it works at all
    # T1a: should return 0.5 for the given sine curves
    # T1b: should return 1.0 for all timesteps
    cal_mask = data.index.year == 2001
    bm_v, bm_t = create_bm(data, "rainfall_runoff_ratio_to_all", cal_mask)
    assert np.isclose(bm_v, 0.5), "Failed rainfall-runoff ratio to all T1a."
    assert (
        np.isclose(bm_t["bm_rainfall_runoff_ratio_to_all"].astype("float"), 1.0)
    ).all(), "Failed rainfall-runoff ratio to all T1b."

    # Test all data to see if no non-cal data is handled correctly
    # T2a: should return 0.5 for the given sine curves
    # T2b: should return 1.0 for all timesteps
    cal_mask = data.index
    bm_v, bm_t = create_bm(data, "rainfall_runoff_ratio_to_all", cal_mask)
    assert np.isclose(bm_v, 0.5), "Failed rainfall-runoff ratio to all T2a."
    assert (
        np.isclose(bm_t["bm_rainfall_runoff_ratio_to_all"].astype("float"), 1.0)
    ).all(), "Failed rainfall-runoff ratio to all T2b."

    # Test 1 year with different P during year 2 to check predictive capability
    # T3a: should return 0.5 for the given sine curves
    # T3b: should return 1.0 for all timesteps in cal_mask
    # T4b: should return 2.0 for all timesteps in ~cal_mask
    cal_mask = data.index.year == 2001
    data["precipitation"].loc[~cal_mask] = data["precipitation"].loc[~cal_mask] * 2
    bm_v, bm_t = create_bm(data, "rainfall_runoff_ratio_to_all", cal_mask)
    assert np.isclose(bm_v, 0.5), "Failed rainfall-runoff ratio T3a."
    assert (
        np.isclose(bm_t["bm_rainfall_runoff_ratio_to_all"].loc[cal_mask].astype("float"), 1.0)
    ).all(), "Failed rainfall-runoff ratio to all T3b."
    assert (
        np.isclose(bm_t["bm_rainfall_runoff_ratio_to_all"].loc[~cal_mask].astype("float"), 2.0)
    ).all(), "Failed rainfall-runoff ratio to all T3c."


def test_rainfall_runoff_ratio_to_annual():
    # Get the testing data
    data = create_sines(period=3)

    # Test 1 year to see if it works at all
    # T1a: should return 0.5 for the given sine curves
    # T1b: should return 1.0 for all timesteps
    cal_mask = data.index.year == 2001
    bm_v, bm_t = create_bm(data, "rainfall_runoff_ratio_to_annual", cal_mask)
    assert np.isclose(bm_v, 0.5), "Failed rainfall-runoff ratio to annual T1a."
    assert (
        np.isclose(bm_t["bm_rainfall_runoff_ratio_to_annual"].astype("float"), 1.0)
    ).all(), "Failed rainfall-runoff ratio to annual T1b."

    # Test all data to see if no non-cal data is handled correctly
    # T2a: should return 0.5 for the given sine curves
    # T2b: should return 1.0 for all timesteps
    cal_mask = data.index
    bm_v, bm_t = create_bm(data, "rainfall_runoff_ratio_to_annual", cal_mask)
    assert np.isclose(bm_v, 0.5), "Failed rainfall-runoff ratio to annual T2a."
    assert (
        np.isclose(bm_t["bm_rainfall_runoff_ratio_to_annual"].astype("float"), 1.0)
    ).all(), "Failed rainfall-runoff ratio to annual T2b."

    # Test 1 year with different P during year 2 to check predictive capability
    # T3a: should return 0.5 for the given sine curves
    # T3b: should return 1.0 for all timesteps in cal_mask
    # T4b: should return 2.0 for all timesteps in year 2002
    # T4b: should return 3.0 for all timesteps in year 2003
    cal_mask = data.index.year == 2001
    data["precipitation"].loc[data.index.year == 2002] = data["precipitation"].loc[data.index.year == 2002] * 2
    data["precipitation"].loc[data.index.year == 2003] = data["precipitation"].loc[data.index.year == 2003] * 3
    bm_v, bm_t = create_bm(data, "rainfall_runoff_ratio_to_annual", cal_mask)
    assert np.isclose(bm_v, 0.5), "Failed rainfall-runoff ratio to annual T3a."
    assert (
        np.isclose(bm_t["bm_rainfall_runoff_ratio_to_annual"].loc[cal_mask].astype("float"), 1.0)
    ).all(), "Failed rainfall-runoff ratio to annual T3b."
    assert (
        np.isclose(bm_t["bm_rainfall_runoff_ratio_to_annual"].loc[data.index.year == 2002].astype("float"), 2.0)
    ).all(), "Failed rainfall-runoff ratio to annual T3c."
    assert (
        np.isclose(bm_t["bm_rainfall_runoff_ratio_to_annual"].loc[data.index.year == 2003].astype("float"), 3.0)
    ).all(), "Failed rainfall-runoff ratio to annual T3d."


def test_rainfall_runoff_ratio_to_monthly():
    # Test 1: 1 year of data should result in 12 unique values but the same values in each month
    data = create_sines(period=1)
    cal_mask = data.index
    bm_v, bm_t = create_bm(data, "rainfall_runoff_ratio_to_monthly", cal_mask)
    assert np.isclose(bm_v, 0.5), "Failed rainfall-runoff ratio to monthly T1a."
    assert (
        len(bm_t["bm_rainfall_runoff_ratio_to_monthly"].unique()) == 12
    ), "Failed rainfall-runoff ratio to monthly T1b."
    assert all(
        bm_t["bm_rainfall_runoff_ratio_to_monthly"].groupby(bm_t.index.month).nunique() == 1
    ), "Failed rainfall-runoff ratio to monthly T1c."

    # Test 2: increase precipitation for years 2 and 3, should result in 36 unique values
    data = create_sines(period=3)
    cal_mask = data.index.year == 2001
    data["precipitation"].loc[data.index.year == 2002] = data["precipitation"].loc[data.index.year == 2002] * 2
    data["precipitation"].loc[data.index.year == 2003] = data["precipitation"].loc[data.index.year == 2003] * 3
    bm_v, bm_t = create_bm(data, "rainfall_runoff_ratio_to_monthly", cal_mask)
    assert (
        len(bm_t["bm_rainfall_runoff_ratio_to_monthly"].unique()) == 36
    ), "Failed rainfall-runoff ratio to monthly T2a."
    assert all(
        bm_t["bm_rainfall_runoff_ratio_to_monthly"].groupby(bm_t.index.month).nunique() == 3
    ), "Failed rainfall-runoff ratio to monthly T2b."


def test_rainfall_runoff_ratio_to_daily():
    # Test 1: 1 year of data should result in 365 unique values but the same values in each day
    data = create_sines(period=1)
    cal_mask = data.index
    bm_v, bm_t = create_bm(data, "rainfall_runoff_ratio_to_daily", cal_mask)
    assert np.isclose(bm_v, 0.5), "Failed rainfall-runoff ratio to daily T1a."
    assert (
        len(bm_t["bm_rainfall_runoff_ratio_to_daily"].unique()) == 365
    ), "Failed rainfall-runoff ratio to daily T1b."
    assert all(
        bm_t["bm_rainfall_runoff_ratio_to_daily"].groupby(bm_t.index.dayofyear).nunique() == 1
    ), "Failed rainfall-runoff ratio to daily T1c."

    # Test 2: increase precipitation for years 2 and 3, should result in 1095 unique values
    data = create_sines(period=3)
    cal_mask = data.index.year == 2001
    data["precipitation"].loc[data.index.year == 2002] = data["precipitation"].loc[data.index.year == 2002] * 2
    data["precipitation"].loc[data.index.year == 2003] = data["precipitation"].loc[data.index.year == 2003] * 3
    bm_v, bm_t = create_bm(data, "rainfall_runoff_ratio_to_daily", cal_mask)
    assert (
        len(bm_t["bm_rainfall_runoff_ratio_to_daily"].unique()) == 1095
    ), "Failed rainfall-runoff ratio to daily T2a."
    assert all(
        bm_t["bm_rainfall_runoff_ratio_to_daily"].groupby(bm_t.index.dayofyear).nunique() == 3
    ), "Failed rainfall-runoff ratio to daily T2b."


def test_rainfall_runoff_ratio_to_timestep():
    # Test 1: 1 year of data should have an overall ratio of 0.5,
    # as well as a per-day ratio of 0.5
    data = create_sines(period=1)
    cal_mask = data.index
    bm_v, bm_t = create_bm(data, "rainfall_runoff_ratio_to_timestep", cal_mask)
    assert np.isclose(bm_v, 0.5), "Failed rainfall-runoff ratio to timestep T1a."
    assert np.isclose(
        (bm_t["bm_rainfall_runoff_ratio_to_timestep"] / data["precipitation"]).values, 0.5
    ).all(), "Failed rainfall-runoff ratio to timestep T1b."


def test_monthly_rainfall_runoff_ratio_to_monthly():
    # Test 1: 1 year of data should have 12 benchmark values in bm_v,
    # and at most 12 unique values in bm_t, as well as 1 unique value per month
    data = create_sines(period=1)
    cal_mask = data.index
    bm_v, bm_t = create_bm(data, "monthly_rainfall_runoff_ratio_to_monthly", cal_mask)
    assert len(bm_v) == 12, "Failed monthly rainfall-runoff ratio to monthly T1a."
    assert (
        len(bm_t["bm_monthly_rainfall_runoff_ratio_to_monthly"].unique()) <= 12
    ), "Failed monthly rainfall-runoff ratio to monthly T1b."
    assert all(
        bm_t.groupby(bm_t.index.month).nunique() == 1
    ), "Failed monthly rainfall-runoff ratio to monthly T1c."

    # Test 2: increase precipitation for years 2 and 3, should result in at most 36 unique values,
    # as well as 3 unique values per month (one for each year)
    data = create_sines(period=3)
    cal_mask = data.index.year == 2001
    data["precipitation"].loc[data.index.year == 2002] = data["precipitation"].loc[data.index.year == 2002] * 2
    data["precipitation"].loc[data.index.year == 2003] = data["precipitation"].loc[data.index.year == 2003] * 3
    bm_v, bm_t = create_bm(data, "monthly_rainfall_runoff_ratio_to_monthly", cal_mask)
    assert len(bm_v) == 12, "Failed monthly rainfall-runoff ratio to monthly T2a."
    assert (
        len(bm_t["bm_monthly_rainfall_runoff_ratio_to_monthly"].unique()) <= 36
    ), "Failed monthly rainfall-runoff ratio to monthly T2b."
    assert all(
        bm_t.groupby(bm_t.index.month).nunique() == 3
    ), "Failed monthly rainfall-runoff ratio to monthly T2c."


def test_monthly_rainfall_runoff_ratio_to_daily():
    # Test 1: 1 year of data should have 12 benchmark values in bm_v,
    # and at most 365 unique values in bm_t, as well as 1 unique value per day
    data = create_sines(period=1)
    cal_mask = data.index
    bm_v, bm_t = create_bm(data, "monthly_rainfall_runoff_ratio_to_daily", cal_mask)
    assert len(bm_v) == 12, "Failed monthly rainfall-runoff ratio to daily T1a."
    assert (
        len(bm_t["bm_monthly_rainfall_runoff_ratio_to_daily"].unique()) <= 365
    ), "Failed monthly rainfall-runoff ratio to daily T1b."
    assert all(
        bm_t.groupby(bm_t.index.dayofyear).nunique() == 1
    ), "Failed monthly rainfall-runoff ratio to daily T1c."

    # Test 2: increase precipitation for years 2 and 3, should result in at most 36 unique values,
    # as well as 3 unique values per month (one for each year)
    data = create_sines(period=3)
    cal_mask = data.index.year == 2001
    data["precipitation"].loc[data.index.year == 2002] = data["precipitation"].loc[data.index.year == 2002] * 2
    data["precipitation"].loc[data.index.year == 2003] = data["precipitation"].loc[data.index.year == 2003] * 3
    bm_v, bm_t = create_bm(data, "monthly_rainfall_runoff_ratio_to_daily", cal_mask)
    assert len(bm_v) == 12, "Failed monthly rainfall-runoff ratio to daily T2a."
    assert (
        len(bm_t["bm_monthly_rainfall_runoff_ratio_to_daily"].unique()) <= 365 * 3
    ), "Failed monthly rainfall-runoff ratio to daily T2b."
    assert all(
        bm_t.groupby(bm_t.index.dayofyear).nunique() == 3
    ), "Failed monthly rainfall-runoff ratio to daily T2c."


def test_monthly_rainfall_runoff_ratio_to_timestep():
    # Test 1: 1 year of data should have 12 benchmark values in bm_v, and 24 unique values on most days
    # We will get fewer than 24 unique values after 3 and 9 months
    # < TO DO > write a test that doesn't use sine curves so we avoid the "values are too close to equal" issue
    data = create_sines(period=1, mean_p=10, var_p=10)
    cal_mask = data.index
    bm_v, bm_t = create_bm(data, "monthly_rainfall_runoff_ratio_to_timestep", cal_mask)
    assert len(bm_v) == 12, "Failed monthly rainfall-runoff ratio to timestep T1a."
    assert (
        bm_t["bm_monthly_rainfall_runoff_ratio_to_timestep"].groupby(bm_t.index.dayofyear).nunique() >= 12
    ).all(), "Failed monthly rainfall-runoff ratio to timestep T1b."
    assert (
        int(bm_t["bm_monthly_rainfall_runoff_ratio_to_timestep"].groupby(bm_t.index.dayofyear).nunique().median())
        == 24
    ), "Failed monthly rainfall-runoff ratio to timestep T1c."


def test_scaled_precipitation_benchmark():
    # Test 1: 1 year of data should have an overall ratio of 0.5,
    # as well as a per-day ratio of 0.5
    data = create_sines(period=1)
    cal_mask = data.index
    bm_v, bm_t = create_bm(data, "scaled_precipitation_benchmark", cal_mask)
    assert np.isclose(bm_v, 0.5), "Failed scaled precipitation benchmark T1a."
    assert np.isclose(
        (bm_t["bm_scaled_precipitation_benchmark"] / data["precipitation"]).values, 0.5
    ).all(), "Failed scaled precipitation benchmark T1b."


def test_adjusted_precipitation_benchmark():
    # Test 1: check if we find the known optimum lag (2)
    dates = pd.date_range("2001-01-01", periods=5, freq="D")
    data = pd.DataFrame({"precipitation": [2, 0, 0, 0, 0], "streamflow": [0, 0, 1, 0, 0]}, index=dates)
    expected_output = pd.DataFrame({"bm_adjusted_precipitation_benchmark": [np.nan, np.nan, 1, 0, 0]}, index=dates)
    cal_mask = data.index
    bm_v, bm_t = create_bm(data, "adjusted_precipitation_benchmark", cal_mask)
    assert np.isclose(bm_v[0], 0.5), "Failed adjusted precipitation benchmark T1a."
    assert np.isclose(bm_v[1], 2), "Failed adjusted precipitation benchmark T1b."
    pd.testing.assert_frame_equal(bm_t, expected_output, check_dtype=False)


def test_adjusted_smoothed_precipitation_benchmark():
    # Test 1: check if we find the known optimum lag and smoothing (2,3)
    dates = pd.date_range("2001-01-01", periods=10, freq="D")
    data = pd.DataFrame(
        {"precipitation": [0, 0, 6, 0, 0, 0, 0, 0, 0, 0], "streamflow": [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]},
        index=dates,
    )
    expected_output = pd.DataFrame(
        {"bm_adjusted_smoothed_precipitation_benchmark": [np.nan, np.nan, np.nan, np.nan, 1, 1, 1, 0, 0, 0]},
        index=dates,
    )
    cal_mask = data.index
    bm_v, bm_t = create_bm(data, "adjusted_smoothed_precipitation_benchmark", cal_mask)
    assert np.isclose(bm_v[0], 0.5), "Failed adjusted smoothed precipitation benchmark T1a."
    assert np.isclose(bm_v[1], 2), "Failed adjusted smoothed precipitation benchmark T1b."
    assert np.isclose(bm_v[2], 3), "Failed adjusted smoothed precipitation benchmark T1c."
    pd.testing.assert_frame_equal(bm_t, expected_output, check_dtype=False)


def test_evaluate_bm():
    # We know the benchmarks work, so we don't need a ton of tests here

    # Get some simple testing data and split 50/50
    dates = pd.date_range("2001-01-01", periods=6, freq="D")
    data = pd.DataFrame({"precipitation": [0, 4, 6, 0, 1, 0], "streamflow": [0, 2, 3, 0, 1, 0]}, index=dates)
    cal_mask = data.index.day < 4
    val_mask = ~cal_mask

    # Create a benchmark
    _, qbm = create_bm(data, "scaled_precipitation_benchmark", cal_mask)

    # Calculate MSE
    cal_mse, val_mse = evaluate_bm(data, qbm, "mse", cal_mask, val_mask=val_mask)

    assert np.isclose(cal_mse, 0)
    assert np.isclose(val_mse, np.mean(np.array([0, 0.5, 0]) ** 2))


if __name__ == "__main__":
    pytest.main([__file__])
