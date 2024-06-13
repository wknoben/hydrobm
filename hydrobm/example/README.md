## HydroBM example

This folder contains example data obtained from the Caravan data set (Kratzert et al., 2023), and a script that shows how to run the HydroBM package for this test case. To reduce repository size only the "total_precipitation_sum", "temperature_2m_mean", and "streamflow" variables are retained in `camels_01022500_minimal.nc`.

### Subsetting code
```
import numpy as np
import xarray as xr

# Load the data
data_file = 'camels_01022500.nc'
mini_file = 'camels_01022500_minimal.nc'

data = xr.open_dataset(data_file)
keep = ['date', 'streamflow', 'total_precipitation_sum', 'temperature_2m_mean']
mini = data.drop_vars([var for var in data.variables if not var in keep])
mask = np.isnan(mini['streamflow'])
mini = mini.isel(date=~mask)
mini.to_netcdf(mini_file)

```

#### References
Kratzert, F., Nearing, G., Addor, N., Erickson, T., Gauch, M., Gilon, O., Gudmundsson, L., Hassidim, A., Klotz, D., Nevo, S., Shalev, G., and Matias, Y.: Caravan - A global community dataset for large-sample hydrology, Scientific Data, 10, 61, https://doi.org/10.1038/s41597- 023-01975-w, 2023.
