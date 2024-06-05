import pandas as pd
import pytest

from ..utils import rain_to_melt


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


if __name__ == "__main__":
    pytest.main([__file__])
