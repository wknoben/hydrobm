# from hydrobm.benchmarks import create_bm # manual import from top-level folder only
import numpy as np
import pandas as pd
import pytest

from ..benchmarks import create_bm


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


def test_daily_mean_flow():
    # Get the testing data
    data = create_sines()

    # T1: should return different values for every day, but the same values within each day
    cal_mask = data.index  # all data
    bm_v, bm_t = create_bm(data, "daily_mean_flow", cal_mask)
    assert len(bm_v.unique()) == 365, "Failed daily mean flow T1a."
    assert all(bm_t.groupby(bm_t.index.dayofyear).nunique() == 1), "Failed daily mean flow T1b."


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


if __name__ == "__main__":
    pytest.main([__file__])
