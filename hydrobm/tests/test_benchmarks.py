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
        "2001-01-01", periods=n_steps, freq="h"
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
    cal_mask = pd.Series(True, index=data.index)  # all data
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


def test_eckhardt_baseflow():
    # ========================================================================
    # Generate synthetic data with known recession coefficient
    # ========================================================================

    # Initialize timeseries
    dates = pd.date_range("2001-01-01", periods=365 * 2, freq="D")
    n_steps = len(dates)

    k_true = 0.96  # Known daily recession coefficient

    # Generate baseflow and quickflow components
    baseflow_true = np.zeros(n_steps)
    quickflow = np.zeros(n_steps)

    baseflow_true[0] = 80.0
    quickflow[0] = 20.0

    # Every 30 days: rain event
    for i in range(1, n_steps):
        if i % 30 == 0:
            baseflow_true[i] = baseflow_true[i - 1] * k_true + 40.0
            quickflow[i] = 60.0
        else:
            baseflow_true[i] = baseflow_true[i - 1] * k_true
            quickflow[i] = quickflow[i - 1] * 0.5

    # Create streamflow and precipitation timeseries
    streamflow = baseflow_true + quickflow
    precipitation = np.zeros(n_steps)
    precipitation[::30] = 20.0

    data = pd.DataFrame({"precipitation": precipitation, "streamflow": streamflow}, index=dates)

    # Calculate true BFI from our synthetic components
    BFI_true = baseflow_true.sum() / streamflow.sum()

    # ========================================================================
    # Test 1: Verify k estimation and BFI_max
    # T1a: k should match true k (0.96)
    # T1b: BFI_max should match true BFI
    # ========================================================================
    cal_mask = data.index.year == 2001
    bm_v, bm_t = create_bm(data, "eckhardt_baseflow", cal_mask)

    k_estimated, BFI_max_estimated = bm_v

    assert np.isclose(
        k_estimated,
        k_true,
        atol=0.0005,  # Tolerance between +- 0.05 % for synthetic data
    ), f"Failed Eckhardt T1a: k={k_estimated:.4f} should be close to {k_true}"

    assert np.isclose(
        BFI_max_estimated,
        BFI_true,
        atol=0.005,  # Tolearance between +- 0.5%, estimated iteratively so a looser tolerance is appropriate
    ), f"Failed Eckhardt T1b: BFI_max={BFI_max_estimated:.4f} should be close to {BFI_true:.4f}"

    # ========================================================================
    # Test 2: Daily climatology pattern
    # T2a: Should have at most 365 unique values (one per day-of-year)
    # T2b: Same day-of-year should have same value across years
    # ========================================================================
    assert bm_t["bm_eckhardt_baseflow"].nunique() <= 365, "Failed Eckhardt T2a: should have ≤365 unique values"

    assert (
        bm_t.groupby(bm_t.index.dayofyear)["bm_eckhardt_baseflow"].nunique() == 1
    ).all(), "Failed Eckhardt T2b: each day-of-year should have 1 unique value"

    # ========================================================================
    # Test 3: Baseflow constraints
    # T3a: Baseflow must be non-negative
    # T3b: Baseflow should be less than daily mean flow.
    # ========================================================================
    assert (bm_t["bm_eckhardt_baseflow"] >= 0).all(), "Failed Eckhardt T3a: baseflow must be >= 0"

    # Compare to daily mean flow benchmark
    _, bm_t_daily = create_bm(data, "daily_mean_flow", cal_mask)

    assert (
        bm_t["bm_eckhardt_baseflow"] <= bm_t_daily["bm_daily_mean_flow"]
    ).all(), "Failed Eckhardt T3b: baseflow should be <= daily mean flow"


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
    # T3c: should return 2.0 for all timesteps in ~cal_mask
    cal_mask = data.index.year == 2001
    data.loc[~cal_mask, "precipitation"] = data.loc[~cal_mask, "precipitation"] * 2
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
    # T3c: should return 2.0 for all timesteps in year 2002
    # T3d: should return 3.0 for all timesteps in year 2003
    cal_mask = data.index.year == 2001
    data.loc[data.index.year == 2002, "precipitation"] = data.loc[data.index.year == 2002, "precipitation"] * 2
    data.loc[data.index.year == 2003, "precipitation"] = data.loc[data.index.year == 2003, "precipitation"] * 3
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
    data.loc[data.index.year == 2002, "precipitation"] = data.loc[data.index.year == 2002, "precipitation"] * 2
    data.loc[data.index.year == 2003, "precipitation"] = data.loc[data.index.year == 2003, "precipitation"] * 3
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
    data.loc[data.index.year == 2002, "precipitation"] = data.loc[data.index.year == 2002, "precipitation"] * 2
    data.loc[data.index.year == 2003, "precipitation"] = data.loc[data.index.year == 2003, "precipitation"] * 3
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
    data.loc[data.index.year == 2002, "precipitation"] = data.loc[data.index.year == 2002, "precipitation"] * 2
    data.loc[data.index.year == 2003, "precipitation"] = data.loc[data.index.year == 2003, "precipitation"] * 3
    bm_v, bm_t = create_bm(data, "monthly_rainfall_runoff_ratio_to_monthly", cal_mask)
    assert len(bm_v) == 12, "Failed monthly rainfall-runoff ratio to monthly T2a."
    assert (
        len(bm_t["bm_monthly_rainfall_runoff_ratio_to_monthly"].unique()) <= 36
    ), "Failed monthly rainfall-runoff ratio to monthly T2b."
    assert all(
        bm_t.groupby(bm_t.index.month).nunique() == 3
    ), "Failed monthly rainfall-runoff ratio to monthly T2c."

    # Test 3: set precipitation values for January  to 0, should result in no NaN in timeseries,
    # and benchmark flows for January being 0
    data = create_sines(period=2)
    data.loc[data.index.month == 1, "precipitation"] = 0.0
    cal_mask = data.index
    bm_v, bm_t = create_bm(data, "monthly_rainfall_runoff_ratio_to_monthly", cal_mask)
    assert (
        not bm_t["bm_monthly_rainfall_runoff_ratio_to_monthly"].isna().any()
    ), "Failed monthly rainfall-runoff ratio to monthly T3a: found NaN values"
    assert (
        bm_t.loc[bm_t.index.month == 1, "bm_monthly_rainfall_runoff_ratio_to_monthly"] == 0
    ).all(), "Failed monthly rainfall-runoff ratio to monthly T3b: not all January values are 0"


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
    data.loc[data.index.year == 2002, "precipitation"] = data.loc[data.index.year == 2002, "precipitation"] * 2
    data.loc[data.index.year == 2003, "precipitation"] = data.loc[data.index.year == 2003, "precipitation"] * 3
    bm_v, bm_t = create_bm(data, "monthly_rainfall_runoff_ratio_to_daily", cal_mask)
    assert len(bm_v) == 12, "Failed monthly rainfall-runoff ratio to daily T2a."
    assert (
        len(bm_t["bm_monthly_rainfall_runoff_ratio_to_daily"].unique()) <= 365 * 3
    ), "Failed monthly rainfall-runoff ratio to daily T2b."
    assert all(
        bm_t.groupby(bm_t.index.dayofyear).nunique() == 3
    ), "Failed monthly rainfall-runoff ratio to daily T2c."

    # Test 3: set precipitation values for January  to 0, should result in no NaN in timeseries,
    # and benchmark flows for January being 0
    data = create_sines(period=2)
    data.loc[data.index.month == 1, "precipitation"] = 0.0
    cal_mask = data.index
    bm_v, bm_t = create_bm(data, "monthly_rainfall_runoff_ratio_to_monthly", cal_mask)
    assert (
        not bm_t["bm_monthly_rainfall_runoff_ratio_to_monthly"].isna().any()
    ), "Failed monthly rainfall-runoff ratio to monthly T3a: found NaN values"
    assert (
        bm_t.loc[bm_t.index.month == 1, "bm_monthly_rainfall_runoff_ratio_to_monthly"] == 0
    ).all(), "Failed monthly rainfall-runoff ratio to monthly T3b: not all January values are 0"


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

    # Test 3: set precipitation values for January  to 0, should result in no NaN in timeseries,
    # and benchmark flows for January being 0
    data = create_sines(period=2)
    data.loc[data.index.month == 1, "precipitation"] = 0.0
    cal_mask = data.index
    bm_v, bm_t = create_bm(data, "monthly_rainfall_runoff_ratio_to_monthly", cal_mask)
    assert (
        not bm_t["bm_monthly_rainfall_runoff_ratio_to_monthly"].isna().any()
    ), "Failed monthly rainfall-runoff ratio to monthly T3a: found NaN values"
    assert (
        bm_t.loc[bm_t.index.month == 1, "bm_monthly_rainfall_runoff_ratio_to_monthly"] == 0
    ).all(), "Failed monthly rainfall-runoff ratio to monthly T3b: not all January values are 0"


def test_annual_scaled_daily_mean_flow():
    # Get the testing data (6 years total) with varying precipitation
    # Start from Dec 2000 to get an incomplete year at the beginning
    data = create_sines(period=6, mean_p=2, mean_q=1, var_p=1, var_q=1, offset_p=1000, offset_q=0)

    # Add December 2000 (incomplete year) at the beginning
    dec_2000_dates = pd.date_range("2000-12-01", "2000-12-31", freq="h")
    dec_2000_data = pd.DataFrame(
        {
            "precipitation": np.random.uniform(1, 3, len(dec_2000_dates)),
            "streamflow": np.random.uniform(0.5, 1.5, len(dec_2000_dates)),
        },
        index=dec_2000_dates,
    )
    data = pd.concat([dec_2000_data, data])

    # Modify precipitation in 2003 to create known variation
    data.loc[data.index.year == 2003, "precipitation"] = data.loc[data.index.year == 2003, "precipitation"] * 3

    # Detect timestep for threshold calculation
    time_diffs = data.index.to_series().diff().median()
    timesteps_per_day = pd.Timedelta("1 day") / time_diffs
    complete_year_threshold = int(365 * timesteps_per_day)

    # Use first 3 years as calibration (2001-2003, excluding incomplete 2000)
    cal_mask = data.index.year.isin([2001, 2002, 2003])

    # Run the benchmark
    bm_v, bm_t = create_bm(
        data, "annual_scaled_daily_mean_flow", cal_mask, precipitation="precipitation", streamflow="streamflow"
    )

    # T1: should return 7 scaling factors (one per year: 2000-2006)
    assert len(bm_v) == 7, "Failed annual scaled T1: should have 7 scaling factors (one per year)"

    # T2: Pattern within one year should match daily mean flow scaled by that year's factor
    # Calculate daily mean flow from calibration
    cal_set = data["streamflow"].loc[cal_mask]
    daily_mean_flow = cal_set.groupby(cal_set.index.dayofyear).mean()

    # Check pattern for year 2001
    year = 2001
    year_mask = data.index.year == year
    year_bm = bm_t.loc[year_mask, "bm_annual_scaled_daily_mean_flow"]
    year_doy = data.index[year_mask].dayofyear

    expected_pattern = daily_mean_flow[year_doy].values * bm_v[year]
    np.testing.assert_allclose(
        year_bm.values,
        expected_pattern,
        rtol=1e-10,
        err_msg="Failed annual scaled T2: daily pattern not scaled correctly",
    )

    # T3: Verify scaling factors are calculated correctly from precipitation
    # Calculate annual precipitation totals
    annual_precip = data["precipitation"].groupby(data.index.year).sum()

    # Calculate mean annual precipitation from calibration period (complete years only)
    cal_precip = data["precipitation"].loc[cal_mask]
    annual_precip_cal = cal_precip.groupby(cal_precip.index.year).sum()
    timesteps_per_year_cal = cal_precip.groupby(cal_precip.index.year).count()
    complete_years_cal = timesteps_per_year_cal >= complete_year_threshold
    mean_annual_precip_cal = annual_precip_cal[complete_years_cal].mean()

    # Check each year's scaling factor
    for year in data.index.year.unique():
        # Check if year is complete (using timestep-aware threshold)
        timesteps_in_year = data.loc[data.index.year == year, "precipitation"].count()

        if timesteps_in_year >= complete_year_threshold:
            # Complete year - should be scaled by precip ratio
            expected_scaling = annual_precip[year] / mean_annual_precip_cal
        else:
            # Incomplete year - should be 1.0
            expected_scaling = 1.0

        np.testing.assert_allclose(
            bm_v[year],
            expected_scaling,
            rtol=1e-10,
            err_msg=f"Failed annual scaled T3 for year {year}: scaling factor incorrect",
        )

    print("All annual scaled daily mean flow tests passed!")


def test_monthly_scaled_daily_mean_flow():
    # Get the testing data (6 years total) with varying precipitation
    data = create_sines(period=6, mean_p=2, mean_q=1, var_p=1, var_q=1, offset_p=1000, offset_q=0)

    # Modify precipitation in specific month to create known variation
    # Triple precipitation in March 2003
    march_2003_mask = (data.index.year == 2003) & (data.index.month == 3)
    data.loc[march_2003_mask, "precipitation"] = data.loc[march_2003_mask, "precipitation"] * 3

    # Use first 3 years as calibration
    cal_mask = data.index.year.isin([2001, 2002, 2003])

    # Run the benchmark
    bm_v, bm_t = create_bm(
        data, "monthly_scaled_daily_mean_flow", cal_mask, precipitation="precipitation", streamflow="streamflow"
    )

    # T1: should return scaling factors for each year-month combination (6 years × 12 months = 72)
    expected_n_months = 6 * 12
    assert (
        len(bm_v) == expected_n_months
    ), f"Failed monthly scaled T1: should have {expected_n_months} scaling factors, got {len(bm_v)}"

    # T2: Pattern within one month should match daily mean flow scaled by that month's factor
    # Calculate daily mean flow from calibration
    cal_set = data["streamflow"].loc[cal_mask]
    daily_mean_flow = cal_set.groupby(cal_set.index.dayofyear).mean()

    # Check pattern for March 2001
    year, month = 2001, 3
    month_mask = (data.index.year == year) & (data.index.month == month)
    month_bm = bm_t.loc[month_mask, "bm_monthly_scaled_daily_mean_flow"]
    month_doy = data.index[month_mask].dayofyear

    expected_pattern = daily_mean_flow[month_doy].values * bm_v[(year, month)]
    np.testing.assert_allclose(
        month_bm.values,
        expected_pattern,
        rtol=1e-10,
        err_msg="Failed monthly scaled T2: daily pattern not scaled correctly",
    )

    # T3: Verify scaling factors are calculated correctly from precipitation
    # Calculate monthly precipitation totals for all year-months
    monthly_precip = data["precipitation"].groupby([data.index.year, data.index.month]).sum()

    # Calculate mean monthly precipitation from calibration period
    cal_precip = data["precipitation"].loc[cal_mask]
    monthly_totals_cal = cal_precip.groupby([cal_precip.index.year, cal_precip.index.month]).sum()
    mean_monthly_precip_cal = monthly_totals_cal.groupby(level=1).mean()  # Average by month (1-12)

    # Check each year-month's scaling factor
    for year, month in bm_v.index:
        if month in mean_monthly_precip_cal.index:
            if mean_monthly_precip_cal[month] > 0:
                expected_scaling = monthly_precip[(year, month)] / mean_monthly_precip_cal[month]
            else:
                expected_scaling = 1.0

            np.testing.assert_allclose(
                bm_v[(year, month)],
                expected_scaling,
                rtol=1e-10,
                err_msg=f"Failed monthly scaled T3 for {year}-{month}: scaling factor incorrect",
            )

    print("All monthly scaled daily mean flow tests passed!")


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
