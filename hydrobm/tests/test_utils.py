import numpy as np
import pandas as pd
import pytest

from ..utils import bme_kge, bme_nse, rain_to_melt


def test_rain_to_melt():
    # Test case 1: No precipitation, temperature below snow_and_melt_temp
    data = pd.DataFrame({"precipitation": [0, 0, 0], "temperature": [-1, -2, -3]})
    expected_output = pd.DataFrame(
        {
            "precipitation": [0, 0, 0],
            "temperature": [-1, -2, -3],
            "snow_depth": [0, 0, 0],
            "rain_plus_melt": [0, 0, 0],
        }
    )
    pd.testing.assert_frame_equal(rain_to_melt(data), expected_output, check_dtype=False)

    # Test case 2: No precipitation, temperature above snow_and_melt_temp
    data = pd.DataFrame({"precipitation": [0, 0, 0], "temperature": [1, 2, 3]})
    expected_output = pd.DataFrame(
        {
            "precipitation": [0, 0, 0],
            "temperature": [1, 2, 3],
            "snow_depth": [0.0, 0.0, 0.0],
            "rain_plus_melt": [0.0, 0.0, 0.0],
        }
    )
    # assert rain_to_melt(data).equals(expected_output)
    pd.testing.assert_frame_equal(rain_to_melt(data), expected_output, check_dtype=False)

    # Test case 3: Precipitation and temperature below snow_and_melt_temp
    data = pd.DataFrame({"precipitation": [1, 2, 3], "temperature": [-1, -2, -3]})
    expected_output = pd.DataFrame(
        {
            "precipitation": [1, 2, 3],
            "temperature": [-1, -2, -3],
            "snow_depth": [1, 3, 6],
            "rain_plus_melt": [0, 0, 0],
        }
    )
    pd.testing.assert_frame_equal(rain_to_melt(data), expected_output, check_dtype=False)

    # Test case 4: Precipitation and temperature above snow_and_melt_temp
    data = pd.DataFrame({"precipitation": [1, 2, 3], "temperature": [1, 2, 3]})
    expected_output = pd.DataFrame(
        {
            "precipitation": [1, 2, 3],
            "temperature": [1, 2, 3],
            "snow_depth": [0, 0, 0],
            "rain_plus_melt": [1, 2, 3],
        }
    )
    pd.testing.assert_frame_equal(rain_to_melt(data), expected_output, check_dtype=False)

    # Test case 5: Precipitation and temperature mix
    data = pd.DataFrame({"precipitation": [1, 2, 3], "temperature": [-1, 1, 3]})
    expected_output = pd.DataFrame(
        {
            "precipitation": [1, 2, 3],
            "temperature": [-1, 1, 3],
            "snow_depth": [1, 0, 0],
            "rain_plus_melt": [0, 3, 3],
        }
    )
    pd.testing.assert_frame_equal(rain_to_melt(data), expected_output, check_dtype=False)

    # Test case 6: Large precipitation and temperature mix
    data = pd.DataFrame({"precipitation": [10, 20, 30], "temperature": [-10, 0, 10]})
    expected_output = pd.DataFrame(
        {
            "precipitation": [10, 20, 30],
            "temperature": [-10, 0, 10],
            "snow_depth": [10, 30, 0],
            "rain_plus_melt": [0, 0, 60],
        }
    )
    pd.testing.assert_frame_equal(rain_to_melt(data), expected_output, check_dtype=False)

    # Test case 7: Empty input data
    data = pd.DataFrame({"precipitation": [], "temperature": []})
    expected_output = pd.DataFrame(
        {"precipitation": [], "temperature": [], "snow_depth": [], "rain_plus_melt": []}
    )
    pd.testing.assert_frame_equal(rain_to_melt(data), expected_output, check_dtype=False)

    # Test case 8: Random input data
    data = pd.DataFrame({"precipitation": [1, 2, 3, 4, 5], "temperature": [0, 1, 2, 3, 4]})
    expected_output = pd.DataFrame(
        {
            "precipitation": [1, 2, 3, 4, 5],
            "temperature": [0, 1, 2, 3, 4],
            "snow_depth": [1, 0, 0, 0, 0],
            "rain_plus_melt": [0, 3, 3, 4, 5],
        }
    )
    pd.testing.assert_frame_equal(rain_to_melt(data), expected_output, check_dtype=False)

    # Test case 9: Change rain/snow threshold
    data = pd.DataFrame({"precipitation": [1, 2, 3], "temperature": [1, 1, 3]})
    expected_output = pd.DataFrame(
        {
            "precipitation": [1, 2, 3],
            "temperature": [1, 1, 3],
            "snow_depth": [1, 3, 0],
            "rain_plus_melt": [0, 0, 6],
        }
    )
    pd.testing.assert_frame_equal(rain_to_melt(data, snow_and_melt_temp=2.0), expected_output, check_dtype=False)

    # Test case 10: Change snow melt rate
    data = pd.DataFrame({"precipitation": [1, 2, 3], "temperature": [-1, -1, 3]})
    expected_output = pd.DataFrame(
        {
            "precipitation": [1, 2, 3],
            "temperature": [-1, -1, 3],
            "snow_depth": [1, 3, 3],
            "rain_plus_melt": [0, 0, 3],
        }
    )
    pd.testing.assert_frame_equal(rain_to_melt(data, snow_and_melt_rate=0.0), expected_output, check_dtype=False)


def test_bme_nse():
    # Test case 1: Perfect simulation — BME should be 1.0
    q_obs = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    q_sim = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    q_bm = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0])
    cal_mask = pd.Series([True, True, True, True, True])
    cal_score, val_score = bme_nse(q_obs, q_sim, q_bm, cal_mask)
    assert cal_score == pytest.approx(1.0)
    assert np.isnan(val_score)

    # Test case 2: Simulation equals benchmark — BME should be 0.0
    q_obs = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    q_sim = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0])
    q_bm = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0])
    cal_mask = pd.Series([True, True, True, True, True])
    cal_score, val_score = bme_nse(q_obs, q_sim, q_bm, cal_mask)
    assert cal_score == pytest.approx(0.0)
    assert np.isnan(val_score)

    # Test case 3: Perfect benchmark (denominator = 0) — BME should be nan
    q_obs = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    q_sim = pd.Series([1.1, 2.1, 3.1, 4.1, 5.1])
    q_bm = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    cal_mask = pd.Series([True, True, True, True, True])
    cal_score, val_score = bme_nse(q_obs, q_sim, q_bm, cal_mask)
    assert np.isnan(cal_score)
    assert np.isnan(val_score)

    # Test case 4: No val_mask provided — val_score should be nan
    q_obs = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    q_sim = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    q_bm = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0])
    cal_mask = pd.Series([True, True, True, True, True])
    _, val_score = bme_nse(q_obs, q_sim, q_bm, cal_mask)
    assert np.isnan(val_score)

    # Test case 5: Cal and val masks — scores computed independently
    # Cal: perfect simulation -> BME = 1.0, Val: sim equals benchmark -> BME = 0.0
    q_obs = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    q_sim = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    q_bm = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    cal_mask = pd.Series([True, True, True, True, True, False, False, False, False, False])
    val_mask = pd.Series([False, False, False, False, False, True, True, True, True, True])
    cal_score, val_score = bme_nse(q_obs, q_sim, q_bm, cal_mask, val_mask=val_mask)
    assert cal_score == pytest.approx(1.0)
    assert val_score == pytest.approx(0.0)

    # Test case 6: NaN in series — affected timestep should be dropped
    # q_obs is NaN at index 2, q_sim has 999 there — if NaN handling breaks, score would be corrupted
    # After dropping index 2: obs=[1,2,4,5], sim=[1,2,4,5], bm=[2,2,2,2] -> BME = 1.0
    q_obs = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    q_sim = pd.Series([1.0, 2.0, 999.0, 4.0, 5.0])
    q_bm = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0])
    cal_mask = pd.Series([True, True, True, True, True])
    cal_score, _ = bme_nse(q_obs, q_sim, q_bm, cal_mask)
    assert cal_score == pytest.approx(1.0)

    # Test case 7: All NaN — should return nan
    q_obs = pd.Series([np.nan, np.nan, np.nan])
    q_sim = pd.Series([1.0, 2.0, 3.0])
    q_bm = pd.Series([1.0, 2.0, 3.0])
    cal_mask = pd.Series([True, True, True])
    cal_score, _ = bme_nse(q_obs, q_sim, q_bm, cal_mask)
    assert np.isnan(cal_score)

    # Test case 8: Returns a tuple of two values
    q_obs = pd.Series([1.0, 2.0, 3.0])
    q_sim = pd.Series([1.0, 2.0, 3.0])
    q_bm = pd.Series([2.0, 2.0, 2.0])
    cal_mask = pd.Series([True, True, True])
    result = bme_nse(q_obs, q_sim, q_bm, cal_mask)
    assert isinstance(result, tuple)
    assert len(result) == 2

    # Test case 9: Cal mask selects correct timesteps
    # Cal covers first 3 timesteps where sim == obs -> BME = 1.0
    # Last 2 timesteps have sim=10 — if incorrectly included, BME would be << 1.0
    q_obs = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    q_sim = pd.Series([1.0, 2.0, 3.0, 10.0, 10.0])
    q_bm = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0])
    cal_mask = pd.Series([True, True, True, False, False])
    cal_score, _ = bme_nse(q_obs, q_sim, q_bm, cal_mask)
    assert cal_score == pytest.approx(1.0)


def test_bme_kge():
    # Test case 1: Perfect simulation — BME should be 1.0
    # KGE_model = 1.0, so (1 - KGE_bm) / (1 - KGE_bm) = 1.0
    q_obs = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    q_sim = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    q_bm = pd.Series([3.0, 3.0, 3.0, 3.0, 3.0])  # mean flow benchmark
    cal_mask = pd.Series([True, True, True, True, True])
    cal_score, val_score = bme_kge(q_obs, q_sim, q_bm, cal_mask)
    assert cal_score == pytest.approx(1.0)
    assert np.isnan(val_score)

    # Test case 2: Simulation equals benchmark — BME should be 0.0
    # KGE_model = KGE_benchmark, so numerator = 0
    q_obs = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    q_sim = pd.Series([3.0, 3.0, 3.0, 3.0, 3.0])
    q_bm = pd.Series([3.0, 3.0, 3.0, 3.0, 3.0])
    cal_mask = pd.Series([True, True, True, True, True])
    cal_score, val_score = bme_kge(q_obs, q_sim, q_bm, cal_mask)
    assert cal_score == pytest.approx(0.0)
    assert np.isnan(val_score)

    # Test case 3: Perfect benchmark (KGE_benchmark = 1) — BME should be nan
    q_obs = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    q_sim = pd.Series([1.1, 2.1, 3.1, 4.1, 5.1])
    q_bm = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])  # perfect benchmark -> KGE = 1
    cal_mask = pd.Series([True, True, True, True, True])
    cal_score, val_score = bme_kge(q_obs, q_sim, q_bm, cal_mask)
    assert np.isnan(cal_score)
    assert np.isnan(val_score)

    # Test case 4: No val_mask provided — val_score should be nan
    q_obs = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    q_sim = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    q_bm = pd.Series([3.0, 3.0, 3.0, 3.0, 3.0])
    cal_mask = pd.Series([True, True, True, True, True])
    _, val_score = bme_kge(q_obs, q_sim, q_bm, cal_mask)
    assert np.isnan(val_score)

    # Test case 5: Cal and val masks — scores computed independently
    # Cal: perfect simulation -> BME = 1.0, Val: sim equals benchmark -> BME = 0.0
    q_obs = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    q_sim = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    q_bm = pd.Series([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    cal_mask = pd.Series([True, True, True, True, True, False, False, False, False, False])
    val_mask = pd.Series([False, False, False, False, False, True, True, True, True, True])
    cal_score, val_score = bme_kge(q_obs, q_sim, q_bm, cal_mask, val_mask=val_mask)
    assert cal_score == pytest.approx(1.0)
    assert val_score == pytest.approx(0.0)

    # Test case 6: All NaN in obs — should return nan
    q_obs = pd.Series([np.nan, np.nan, np.nan])
    q_sim = pd.Series([1.0, 2.0, 3.0])
    q_bm = pd.Series([1.0, 2.0, 3.0])
    cal_mask = pd.Series([True, True, True])
    cal_score, _ = bme_kge(q_obs, q_sim, q_bm, cal_mask)
    assert np.isnan(cal_score)

    # Test case 7: Returns a tuple of two values
    q_obs = pd.Series([1.0, 2.0, 3.0])
    q_sim = pd.Series([1.0, 2.0, 3.0])
    q_bm = pd.Series([3.0, 3.0, 3.0])
    cal_mask = pd.Series([True, True, True])
    result = bme_kge(q_obs, q_sim, q_bm, cal_mask)
    assert isinstance(result, tuple)
    assert len(result) == 2

    # Test case 8: Cal mask selects correct timesteps
    # Cal covers first 3 timesteps where sim == obs -> BME = 1.0
    # Last 2 timesteps have sim=100 — if incorrectly included, BME would be << 1.0
    q_obs = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    q_sim = pd.Series([1.0, 2.0, 3.0, 100, 100])
    q_bm = pd.Series([3.0, 3.0, 3.0, 3.0, 3.0])
    cal_mask = pd.Series([True, True, True, False, False])
    cal_score, _ = bme_kge(q_obs, q_sim, q_bm, cal_mask)
    assert cal_score == pytest.approx(1.0)

    # Test case 9: NaN in series — affected timestep should be dropped
    # q_obs is NaN at index 2, q_sim has 999 there — if NaN handling breaks, score would be corrupted
    # After dropping index 2: obs=[1,2,4,5], sim=[1,2,4,5], bm=[3,3,3,3] -> BME = 1.0
    q_obs = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    q_sim = pd.Series([1.0, 2.0, 999.0, 4.0, 5.0])
    q_bm = pd.Series([3.0, 3.0, 3.0, 3.0, 3.0])
    cal_mask = pd.Series([True, True, True, True, True])
    cal_score, _ = bme_kge(q_obs, q_sim, q_bm, cal_mask)
    assert cal_score == pytest.approx(1.0)

    # Test case 10: Constant benchmark (no variance) — should still return a valid float
    # std(q_bm) = 0 triggers the zero-variance guard in kge(), but kge_benchmark != 1
    # so the skill score formula should still produce a valid result
    q_obs = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    q_sim = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5])
    q_bm = pd.Series([3.0, 3.0, 3.0, 3.0, 3.0])  # constant, zero variance
    cal_mask = pd.Series([True, True, True, True, True])
    cal_score, _ = bme_kge(q_obs, q_sim, q_bm, cal_mask)
    assert not np.isnan(cal_score)
    assert np.isfinite(cal_score)


if __name__ == "__main__":
    pytest.main([__file__])
