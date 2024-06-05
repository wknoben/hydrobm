# from hydrobm.benchmarks import create_bm # manual import from top-level folder only
import numpy as np
import pandas as pd
import pytest

from ..benchmarks import create_bm

# Create three sine curve time series for testing (precip, flow, temp)
hour_per_year = 365 * 24  # 365 days/year * 24 hours/day
period = 2 * hour_per_year  # 2 years
dates = pd.date_range("2001-01-01", periods=period, freq="H")  # Start in 2001 so we avoid the leap year in 2000


# Sine curve parameters
mean_p = 2
mean_q = 1
var_p = 1  # less than mean, so we get some rain on all days so we can test snow
var_q = 1  # same as mean, so flow drops to zero but not below
offset_p = 0  # for completeness
offset_q = 0
data_p = mean_p + var_p * np.sin((np.arange(period) - offset_p) / hour_per_year * (2 * np.pi))
data_q = mean_q + var_q * np.sin((np.arange(period) - offset_q) / hour_per_year * (2 * np.pi))


# DataFrame for benchmark calculation inputs
data = pd.DataFrame(
    {"precipitation": data_p, "streamflow": data_q},
    index=dates,
)


def test_mean_flow():
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
    # T1: should return different values for every month, but the same values within each month
    cal_mask = data.index  # all data
    bm_v, bm_t = create_bm(data, "monthly_mean_flow", cal_mask)
    assert len(bm_v.unique()) == 12, "Failed monthly mean flow T1a."
    assert all(bm_t.groupby(bm_t.index.month).nunique() == 1), "Failed monthly mean flow T1b."


def test_daily_mean_flow():
    # T1: should return different values for every day, but the same values within each day
    cal_mask = data.index  # all data
    bm_v, bm_t = create_bm(data, "daily_mean_flow", cal_mask)
    assert len(bm_v.unique()) == 365, "Failed daily mean flow T1a."
    assert all(bm_t.groupby(bm_t.index.dayofyear).nunique() == 1), "Failed daily mean flow T1b."


if __name__ == "__main__":
    pytest.main([__file__])
