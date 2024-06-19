# Example use of HydroBM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from hydrobm.calculate import calc_bm

# Get the example data
data_file = "./hydrobm/example/camels_01022500_minimal.nc"
data = xr.open_dataset(data_file)

# Create an exploratory plot
fig, ax = plt.subplots(1, 1)
data["total_precipitation_sum"].plot(ax=ax, label="precipitation")
data["streamflow"].plot(ax=ax, label="streamflow")
data["temperature_2m_mean"].plot(ax=ax, label="temperature")
plt.legend()
plt.show()

# Specify the calculation and evaluation periods, as boolean masks
cal_mask = data["date"].values < np.datetime64("1999-01-01")
val_mask = ~cal_mask

# Specify the benchmarks and metrics to calculate
benchmarks_to_calculate = [
    # Streamflow benchmarks
    "mean_flow",
    "median_flow",
    "annual_mean_flow",
    "annual_median_flow",
    "monthly_mean_flow",
    "monthly_median_flow",
    "daily_mean_flow",
    "daily_median_flow",
    # Long-term rainfall-runoff ratio benchmarks
    "rainfall_runoff_ratio_to_all",
    "rainfall_runoff_ratio_to_annual",
    "rainfall_runoff_ratio_to_monthly",
    "rainfall_runoff_ratio_to_daily",
    "rainfall_runoff_ratio_to_timestep",
    # Short-term rainfall-runoff ratio benchmarks
    "monthly_rainfall_runoff_ratio_to_monthly",
    "monthly_rainfall_runoff_ratio_to_daily",
    "monthly_rainfall_runoff_ratio_to_timestep",
    # Schaefli & Gupta (2007) benchmarks
    "scaled_precipitation_benchmark",  # equivalent to "rainfall_runoff_ratio_to_daily"
    "adjusted_precipitation_benchmark",
    "adjusted_smoothed_precipitation_benchmark",
]
metrics = ["nse", "kge", "mse", "rmse"]

# Calculate the benchmarks and scores
benchmarks, scores = calc_bm(
    data,
    # Time period selection
    cal_mask,
    val_mask=val_mask,
    # Variable names in 'data'
    precipitation="total_precipitation_sum",
    streamflow="streamflow",
    optimization_method="brute_force",
    # Benchmark choices
    benchmarks=benchmarks_to_calculate,
    metrics=metrics,
    # Snow model inputs
    calc_snowmelt=True,
    temperature="temperature_2m_mean",
    snowmelt_threshold=0.0,
    snowmelt_rate=3.0,
)

# Print the scores with some basic formatting applied
for key, val in scores.items():
    if key == "benchmarks":
        print(f"{key}: {val}")
    else:
        pm = [f"{num:.2f}" for num in val]
        print(f"{key}: {pm}")


# Select the four best benchmarks for plotting
def top_n_indices_and_values(values_list, n=3):
    arr = np.array(values_list)  # numpy array
    nan_idx = np.where(np.isnan(arr))  # find nan values
    arr_sort = arr.argsort()  # sort the full array, nans go at the end
    arr_sort = arr_sort[~np.in1d(arr_sort, nan_idx)]  # remove nans
    indices = arr_sort[-n:]  # get the top n indices
    values = arr[indices]  # get the values
    return indices.tolist(), values.tolist()


idx, vals = top_n_indices_and_values(scores["kge_val"], 4)
top_benchmarks = [scores["benchmarks"][id] for id in idx]
top_kge_vals = [scores["kge_val"][id] for id in idx]

# Print streamflow along with the four best benchmarks
fig, ax = plt.subplots(4, 1, figsize=(14, 14))
for i, (bm, kge) in enumerate(zip(top_benchmarks, top_kge_vals)):
    data["streamflow"].plot(ax=ax[i], linewidth=2, label="streamflow")
    benchmarks[f"bm_{bm}"].plot(ax=ax[i], label=bm)
    ax[i].legend(loc="upper left")
    ax[i].set_title(f"{bm} (KGE: {kge:.2f})")
    ax[i].set_xlabel("")  # drops 'Date'

plt.tight_layout()
plt.show()


# Save the benchmark models and scores
col_names = scores.pop("benchmarks", None)
df = pd.DataFrame(scores, index=col_names)
df = df.T
df.to_csv("scores.csv")
benchmarks.to_csv("benchmarks_flows.csv")
