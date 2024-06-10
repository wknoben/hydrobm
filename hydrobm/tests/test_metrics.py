import numpy as np
import pytest

from ..metrics import calculate_metric, kge, mse, nse, rmse


def test_nse():
    # Test with sim equal to obs: perfect simulation
    obs = np.random.randint(0, 100, 1000)  # random time series
    sim = obs
    assert np.allclose(nse(obs, sim), 1)

    # Test with sim equal to mean(obs): no skill
    sim = np.ones(np.shape(obs)) * np.mean(obs)
    assert np.allclose(nse(obs, sim), 0)


def test_kge():
    # Test with sim equal to obs: perfect simulation
    obs = np.random.randint(0, 100, 1000)  # random time series
    sim = obs
    assert np.allclose(kge(obs, sim), 1)

    # Test with sim equal to mean(obs): no skill
    sim = np.ones(np.shape(obs)) * np.mean(obs)
    assert np.allclose(kge(obs, sim), 1 - np.sqrt(2))


def test_rmse():
    # Test with sim equal to obs: perfect simulation
    obs = np.random.randint(0, 100, 1000)  # random time series
    sim = obs
    assert np.allclose(rmse(obs, sim), 0)

    # Test with sim equal to mean(obs): no skill
    sim = np.ones(np.shape(obs)) * np.mean(obs)
    assert np.allclose(rmse(obs, sim), np.std(obs))


def test_mse():
    # Test with sim equal to obs: perfect simulation
    obs = np.random.randint(0, 100, 1000)  # random time series
    sim = obs
    assert np.allclose(mse(obs, sim), 0)

    # Test with sim equal to mean(obs): no skill
    sim = np.ones(np.shape(obs)) * np.mean(obs)
    assert np.allclose(mse(obs, sim), np.std(obs) ** 2)


def test_calculate_metric():
    # Repeat tests above in a loop

    # MSE
    obs = np.random.randint(0, 100, 1000)  # random time series
    sim = obs
    assert np.allclose(calculate_metric(obs, sim, "mse"), 0)

    sim = np.ones(np.shape(obs)) * np.mean(obs)
    assert np.allclose(calculate_metric(obs, sim, "mse"), np.std(obs) ** 2)

    # RMSE
    obs = np.random.randint(0, 100, 1000)  # random time series
    sim = obs
    assert np.allclose(calculate_metric(obs, sim, "rmse"), 0)

    sim = np.ones(np.shape(obs)) * np.mean(obs)
    assert np.allclose(calculate_metric(obs, sim, "rmse"), np.std(obs))

    # NSE
    obs = np.random.randint(0, 100, 1000)  # random time series
    sim = obs
    assert np.allclose(calculate_metric(obs, sim, "nse"), 1)

    sim = np.ones(np.shape(obs)) * np.mean(obs)
    assert np.allclose(calculate_metric(obs, sim, "nse"), 0)

    # KGE
    obs = np.random.randint(0, 100, 1000)  # random time series
    sim = obs
    assert np.allclose(calculate_metric(obs, sim, "kge"), 1)

    sim = np.ones(np.shape(obs)) * np.mean(obs)
    assert np.allclose(calculate_metric(obs, sim, "kge"), 1 - np.sqrt(2))


if __name__ == "__main__":
    pytest.main([__file__])
