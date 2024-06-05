import numpy as np


# Basic snow accumulation and melt model
def rain_to_melt(
    data, precipitation="precipitation", temperature="temperature", snow_and_melt_temp=0.0, snow_and_melt_rate=3.0
):
    """Calculate snow accumulation and melt based on temperature thresholds.

    Parameters
    ----------
    data : pandas DataFrame
        Input data containing precipitation and temperature columns.
    precipitation : str, optional
        Name of the precipitation column in the input data. Default is 'precipitation'.
    temperature : str, optional'
        Name of the temperature column in the input data. Default is 'temperature'.
    snow_and_melt_temp : float, optional
        Temperature threshold for snow accumulation and melt. Default is 0.0 [C].
    snow_and_melt_rate : float, optional
        Snow melt rate if temperature above threshold. Default is 3.0 [mm/hour/degree C].

    Returns
    -------
    data : pandas DataFrame
        Input data with additional columns for snow depth and rain plus melt.
    """

    # Docs: 3 degrees C is a conservative estimate (see e.g.: https://tc.copernicus.org/articles/17/211/2023/)

    # Check that melt rate is not negative
    if snow_and_melt_rate < 0:
        raise ValueError(f"Snow melt rate must be non-negative. Currently set to: {snow_and_melt_rate}.")

    # Run a really simple time-stepping scheme to account for snow accumulation and melt.
    # We'll deal with the time step implicitly, simply assuming that delta t = 1.
    # We can get away with Explicit Euler (Snew = Sold + snowfall - snowmelt) because fall
    # and melt are mutually exclusive: no problems with ad-hoc operator splitting here.
    snow_depth = []
    rain_plus_melt = []
    Sold = 0
    for _, row in data.iterrows():
        # Determine snowfall or melt
        if row[temperature] > snow_and_melt_temp:
            melt = np.min([Sold, snow_and_melt_rate * (row[temperature] - snow_and_melt_temp)])
            rain = row[precipitation]
            snow = 0
        else:
            melt = 0
            rain = 0
            snow = row[precipitation]

        # Update the snow pack
        Snew = Sold + snow - melt

        # Retain the values
        snow_depth.append(Snew)
        rain_plus_melt.append(rain + melt)

        # Prepare for the next time step
        Sold = Snew

    # Outputs
    data["snow_depth"] = snow_depth
    data["rain_plus_melt"] = rain_plus_melt

    return data
