import logging

# Time range for data processing
START_DATE = '2019-03-31'
END_DATE = '2023-03-31'

# Base features
BASE_FEATURES = [
    'PM2.5 (µg/m³)', 'PM10 (µg/m³)', 'Nitric Oxide (µg/m³)', 'Nitrogen Dioxide (µg/m³)', 'Nitrogen Oxides (ppb)',
    'Ammonia (µg/m³)', 'Sulphur Dioxide (µg/m³)', 'Carbon Monoxide (mg/m³)', 'Ozone (µg/m³)', 'Benzene (µg/m³)',
    'Temperature (°C)', 'Relative Humidity (%)', 'Wind Speed (m/s)', 'Wind Direction (°)', 'Rainfall (mm)'
]

# Cyclic temporal features
CYCLIC_FEATURES = [
    'hour_sin', 'hour_cos',
    'day_of_week_sin', 'day_of_week_cos',
    'month_sin', 'month_cos'
]

# All core features
CORE_FEATURES = BASE_FEATURES + CYCLIC_FEATURES

# Physical limits
LIMITS = {
    'PM2.5 (µg/m³)': (0, 1000), 'PM10 (µg/m³)': (0, 2000), 'Nitric Oxide (µg/m³)': (0, 2000),
    'Nitrogen Dioxide (µg/m³)': (0, 1000), 'Nitrogen Oxides (ppb)': (0, 2000), 'Ammonia (µg/m³)': (0, 1000),
    'Sulphur Dioxide (µg/m³)': (0, 1000), 'Carbon Monoxide (mg/m³)': (0, 50), 'Ozone (µg/m³)': (0, 500),
    'Benzene (µg/m³)': (0, 500), 'Temperature (°C)': (-20, 55), 'Relative Humidity (%)': (0, 100),
    'Wind Speed (m/s)': (0, 40), 'Wind Direction (°)': (0, 360), 'Rainfall (mm)': (0, 500)
}

# Feature mapping for merging columns
FEATURE_MAPPING = {
    "PM2.5 (µg/m³)": ["PM2.5 (ug/m3)"],
    "PM10 (µg/m³)": ["PM10 (ug/m3)"],
    "Nitric Oxide (µg/m³)": ["NO (ug/m3)"],
    "Nitrogen Dioxide (µg/m³)": ["NO2 (ug/m3)"],
    "Nitrogen Oxides (ppb)": ["NOx (ppb)"],
    "Ammonia (µg/m³)": ["NH3 (ug/m3)"],
    "Sulphur Dioxide (µg/m³)": ["SO2 (ug/m3)"],
    "Carbon Monoxide (mg/m³)": ["CO (mg/m3)"],
    "Ozone (µg/m³)": ["Ozone (ug/m3)"],
    "Benzene (µg/m³)": ["Benzene (ug/m3)"],
    "Temperature (°C)": ["AT (degree C)", "AT (degree)", "AT ()", "Temp (degree C)", "Temp ()"],
    "Relative Humidity (%)": ["RH (%)", "RH ()"],
    "Wind Speed (m/s)": ["WS (m/s)", "WS ()"],
    "Wind Direction (°)": ["WD (degree)", "WD (deg)", "WD ()", "WD (degree C)"],
    "Rainfall (mm)": ["RF (mm)", "RF ()", "RF (mm).1", "RF (mm).2", "RF (mm).3", "RF (mm).4", "RF (mm).5", "RF (mm).6", "RF (mm).7"]
}

# Columns to drop
DROP_COLS = [
    'BP (W/mt2)', 'BP (mg/m3)', 'RH (W/mt2)', 'WD (degree C)', 'WS (ug/m3)', 'AT (ug/m3)', 'Temp (ug/m3)', 'SR (ug/m3)',
    'NO (ppb)', 'NO (ppm)', 'NO (mg/m3)', 'NH3 (ppb)', 'Ozone (ppb)', 'CO (ug/m3)', 'CO (ng/m3)', 'NOx (ppm)', 'NOx (ug/m3)', 'Benzene (mg/m3)',
    'Hg (ug/m3)', 'HCHO (ug/m3)', 'SPM (ug/m3)', 'CO2 (mg/m3)', 'CH4 ()', 'NMHC ()', 'THC ()', 'Gust (m/s)', 'VWS (m/s)',
]

# Configure logging
LOG_FILE = 'log.txt'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filename=LOG_FILE,
    filemode='w',
    force=True
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
