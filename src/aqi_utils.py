# (conc_low, conc_high, aqi_low, aqi_high) — EPA PM2.5 breakpoints, µg/m³
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]


def compute_aqi_from_pm25(pm25: float) -> float | None:
    """
    Convert a PM2.5 concentration (µg/m³) to a US AQI value (0-500) using
    the EPA piecewise-linear breakpoint formula. Returns None if pm25 is
    missing/invalid or exceeds the table's top breakpoint.
    """
    if pm25 is None:
        return None
    try:
        pm25 = float(pm25)
    except (ValueError, TypeError):
        return None
    if pm25 < 0:
        return None

    for conc_low, conc_high, aqi_low, aqi_high in PM25_BREAKPOINTS:
        if conc_low <= pm25 <= conc_high:
            aqi = ((aqi_high - aqi_low) / (conc_high - conc_low)) * (pm25 - conc_low) + aqi_low
            return round(aqi, 1)

    # Above the top breakpoint (500.4) — cap at 500 rather than extrapolate.
    if pm25 > PM25_BREAKPOINTS[-1][1]:
        return 500.0

    return None