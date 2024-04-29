
import numpy as np
from ..metrics import nse, rmse, kge

def test_nse():
    # Test with sim equal to obs: perfect simulation
    obs = np.random.randint(0,100,1000) # random time series
    sim = obs
    assert np.allclose(nse(obs, sim), 1)

    # Test with sim equal to mean(obs): no skill
    sim = np.ones(np.shape(obs)) * np.mean(obs)
    assert np.allclose(nse(obs, sim), 0)

def test_kge():
    # Test with sim equal to obs: perfect simulation
    obs = np.random.randint(0,100,1000) # random time series
    sim = obs
    assert np.allclose(kge(obs, sim), 1)

    # Test with seam equal to mean(obs): no skill
    sim = np.ones(np.shape(obs)) * np.mean(obs) 
    assert np.allclose(kge(obs, sim), 1-np.sqrt(2))

def test_rmse():
    # Test with sim equal to obs: perfect simulation
    obs = np.random.randint(0,100,1000) # random time series
    sim = obs
    assert np.allclose(rmse(obs, sim), 0)

    # Test with sim equal to mean(obs): no skill
    sim = np.ones(np.shape(obs)) * np.mean(obs)
    assert np.allclose(rmse(obs, sim), np.std(obs))