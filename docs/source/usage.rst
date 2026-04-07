=====
Usage
=====

Start by importing Benchmarks for Hydrologic Timeseries.

.. code-block:: python

    import hydrobm


Main calculation function
-------------------------

HydroBM provides a main function to calculate the benchmark timeseries.
This is a catch-all function that lets you set up a complete benchmarking
exercise for a given time series of observed streamflow (and optionally
other variables, depending on the selected benchmarks). Functions are
accessible outside of this main function too for more granular setups.

.. autofunction:: hydrobm.calculate.calc_bm


Benchmark Efficiency (BME) function
-------------------------------------------

HydroBM also provides a function to calculate skill scores termed benchmark
efficiencies (BME) (Schaefli & Gupta, 2007) between hydrological model
simulations and benchmark timeseries. This function supports the
Schaefli and Gupta (2007) and Siebert (2001) formulation of the BME skill score,
as well as a skill score formulation of the KGE (Knoben et al. 2019). This function
is functionally identical to calc_bm, but also requires simulated streamflow and
the desired formulation of the BME.

.. autofunction:: hydrobm.calculate.calc_bme


Benchmarks
----------

Within their respective category, benchmarks are all set up to require
the same inputs. Click on each benchmark in the table for more information.

Benchmarks that rely on streamflow data only
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   hydrobm.benchmarks.bm_mean_flow
   hydrobm.benchmarks.bm_median_flow
   hydrobm.benchmarks.bm_annual_mean_flow
   hydrobm.benchmarks.bm_annual_median_flow
   hydrobm.benchmarks.bm_monthly_mean_flow
   hydrobm.benchmarks.bm_monthly_median_flow
   hydrobm.benchmarks.bm_daily_mean_flow
   hydrobm.benchmarks.bm_daily_median_flow

Benchmarks that rely on precipitation and streamflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   hydrobm.benchmarks.bm_rainfall_runoff_ratio_to_all
   hydrobm.benchmarks.bm_rainfall_runoff_ratio_to_annual
   hydrobm.benchmarks.bm_rainfall_runoff_ratio_to_monthly
   hydrobm.benchmarks.bm_rainfall_runoff_ratio_to_daily
   hydrobm.benchmarks.bm_rainfall_runoff_ratio_to_timestep
   hydrobm.benchmarks.bm_monthly_rainfall_runoff_ratio_to_monthly
   hydrobm.benchmarks.bm_monthly_rainfall_runoff_ratio_to_daily
   hydrobm.benchmarks.bm_monthly_rainfall_runoff_ratio_to_timestep
   hydrobm.benchmarks.bm_annual_scaled_daily_mean_flow
   hydrobm.benchmarks.bm_monthly_scaled_daily_mean_flow
   hydrobm.benchmarks.bm_scaled_precipitation_benchmark

Parsimonious model benchmarks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   hydrobm.benchmarks.bm_eckhardt_baseflow
   hydrobm.benchmarks.bm_adjusted_precipitation_benchmark
   hydrobm.benchmarks.bm_adjusted_smoothed_precipitation_benchmark



Benchmark support functions
---------------------------

.. autofunction:: hydrobm.benchmarks.create_bm

.. autofunction:: hydrobm.benchmarks.evaluate_bm


Benchmark optimization functions
--------------------------------
Only used by the Eckhardt Baseflow, Adjusted Precipitation Benchmark (APB), and
Adjusted Smoothed Precipitation Benchmark (ASPB) to optimize or estimate
their respective parameters.

.. autofunction:: hydrobm.utils.optimize_apb

.. autofunction:: hydrobm.utils.brute_force_apb

.. autofunction:: hydrobm.utils.minimize_scalar_apb

.. autofunction:: hydrobm.utils.optimize_aspb

.. autofunction:: hydrobm.utils.brute_force_aspb

.. autofunction:: hydrobm.utils.minimize_aspb

.. autofunction:: hydrobm.utils.estimate_eckhardt_parameters

.. autofunction:: hydrobm.utils.eckhardt_filter

Metrics
-------

.. autofunction:: hydrobm.metrics.mse

.. autofunction:: hydrobm.metrics.rmse

.. autofunction:: hydrobm.metrics.nse

.. autofunction:: hydrobm.metrics.kge

Metric support functions
------------------------

.. autofunction:: hydrobm.metrics.calculate_metric

.. autofunction:: hydrobm.metrics.filter_nan

Utilities
---------

.. autofunction:: hydrobm.utils.rain_to_melt

.. autofunction:: hydrobm.utils.bme_nse

.. autofunction:: hydrobm.utils.bme_kge


References
------------

Knoben, W. J. M., Freer, J. E., & Woods, R. A. (2019). Technical note: Inherent benchmark or not? Comparing
Nash–Sutcliffe and Kling–Gupta efficiency scores. Hydrology and Earth System Sciences, 23(10), 4323–4331.
https://doi.org/10.5194/hess-23-4323-2019

Schaefli, B., & Gupta, H. V. (2007). Do Nash values have value? Hydrological Processes, 21(15), 2075–2080.
https://doi.org/10.1002/hyp.6825

Seibert, J. (2001). On the need for benchmarks in hydrological modelling. Hydrological Processes, 15(6),
1063–1064. https://doi.org/10.1002/hyp.446
