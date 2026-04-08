===============
Release History
===============

v1.1.0 (2026-04-08)
-------------------
Additions:

- Added three new benchmarks: `Annual Scaled Daily Mean Flow`, `Monthly Scaled Daily Mean Flow`, `Eckhardt Baseflow` (`GitHub PR #24 <pr-24_>`_).
- Added ability to calculate Benchmark Efficiencies as part of the metric toolkit (`GitHub PR #27 <pr-27_>`_).
- Added example plots of each individual benchmark to the documentation.

Bugfixes:

- Fixed occassional divide-by-zero error in monthly rainfall-runoff ratio benchmarks (`GitHub PR #21 <pr-21_>`_).
- Fixed cases where `annual_mean_flow` and `annual_median_flow` would incorrectly return simulations for the evaluation period if calibration period did not cover full calendar years (`GitHub PR #28 <pr-28_>`_).

Changes:

- Harmonized benchmark functions names used internally (now all start with `bm_`).
- Updated various parts to account for changes in Pandas v3 (chained assignment, deprecated syntax).
- Updated test logic in `test_annual_mean_flow`.
- Fixed various typos.


v1.0.2 (2024-10-04)
-------------------
Changes:

- Corrected license listed on pypi.org.


v1.0.1 (2024-10-03)
-------------------
Bugfixes:

- Fixed a bug that stopped benchmark evaluation with user-specified streamflow variable names.


v1.0.0 (2024-09-12)
----------------------------
Initial release with core functionality:

- Capability to work with pandas DataFrames (e.g., CSV data) and xarray DataSets (e.g., netCDF data).
- Capability to create 19 benchmark timeseries for hydrologic models.
- Documention on readthedocs.io.

Additional Features:

- Capability to calculate 4 performance metrics for the benchmark simulations.
- Capability to run a simple degree-day model for snow accumulation and melt simulation.

Notes:

- Release coincides with submission of a related commentary to Hydrologic Processes.


.. GitHub PR references (used above in release notes)
.. Keep these at the bottom to avoid duplication errors

.. _pr-21: https://github.com/wknoben/hydrobm/pull/21
.. _pr-24: https://github.com/wknoben/hydrobm/pull/24
.. _pr-27: https://github.com/wknoben/hydrobm/pull/27
.. _pr-28: https://github.com/wknoben/hydrobm/pull/28
