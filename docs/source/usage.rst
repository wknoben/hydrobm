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

Benchmarks that rely on on precipitation and streamflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   hydrobm.benchmarks.bm_rainfall_runoff_ratio_to_all
   hydrobm.benchmarks.bm_rainfall_runoff_ratio_to_annual
   hydrobm.benchmarks.bm_rainfall_runoff_ratio_to_monthly
   hydrobm.benchmarks.bm_rainfall_runoff_ratio_to_daily
   hydrobm.benchmarks.bm_rainfall_runoff_ratio_to_timestep
   hydrobm.benchmarks.monthly_rainfall_runoff_ratio_to_monthly
   hydrobm.benchmarks.monthly_rainfall_runoff_ratio_to_daily
   hydrobm.benchmarks.monthly_rainfall_runoff_ratio_to_timestep
   hydrobm.benchmarks.scaled_precipitation_benchmark
   hydrobm.benchmarks.adjusted_precipitation_benchmark
   hydrobm.benchmarks.adjusted_smoothed_precipitation_benchmark


Benchmark support functions
---------------------------

.. autofunction:: hydrobm.benchmarks.create_bm

.. autofunction:: hydrobm.benchmarks.evaluate_bm


Benchmark optimization functions
--------------------------------
Only used by the Adjusted Precipitation Benchmark (APB) and
Adjusted Smoothed Precipitation Benchmark (ASPB) to optimize
their respective parameters.

.. autofunction:: hydrobm.utils.optimize_apb

.. autofunction:: hydrobm.utils.brute_force_apb

.. autofunction:: hydrobm.utils.minimize_scalar_apb

.. autofunction:: hydrobm.utils.optimize_aspb

.. autofunction:: hydrobm.utils.brute_force_aspb

.. autofunction:: hydrobm.utils.minimize_aspb

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
