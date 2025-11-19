import logging

# Time range for data processing
START_DATE = '2021-03-31'
END_DATE = '2023-03-31'

# Base weather/pollution features to process (12 features from raw data)
# Removed: Ammonia (µg/m³), Benzene (µg/m³), Rainfall (mm) due to high missing data
BASE_FEATURES = [
    'PM2.5 (µg/m³)', 'PM10 (µg/m³)', 'Nitric Oxide (µg/m³)', 'Nitrogen Dioxide (µg/m³)',
    'Sulphur Dioxide (µg/m³)', 'Carbon Monoxide (mg/m³)', 'Ozone (µg/m³)',
    'Temperature (°C)', 'Relative Humidity (%)', 'Wind Speed (m/s)', 'Wind Direction (°)'
]

# Cyclic temporal features added during preprocessing (6 features)
CYCLIC_FEATURES = ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos']

# All core features: BASE + CYCLIC (21 total)
CORE_FEATURES = BASE_FEATURES + CYCLIC_FEATURES

# Physical limits - More restrictive for India context
# Based on typical Indian air quality patterns and monsoon characteristics
LIMITS = {
    'PM2.5 (µg/m³)': (0, 500),           # India often exceeds 400 during winter, rare >500
    'PM10 (µg/m³)': (0, 1000),           # Peak values ~800-1000 during winter fog
    'Nitric Oxide (µg/m³)': (0, 500),    # Urban peak ~300-400 ppb equivalent
    'Nitrogen Dioxide (µg/m³)': (0, 200), # Urban NO2 rarely exceeds 150-200
    'Sulphur Dioxide (µg/m³)': (0, 200), # Industrial/coal areas ~80-150, rare >200
    'Carbon Monoxide (mg/m³)': (0, 10),  # India CO typically <5, peak industrial ~8
    'Ozone (µg/m³)': (0, 300),           # Peak O3 ~150-200 in summer, rare >300
    'Temperature (°C)': (5, 50),         # India range: 5°C (winter hills) to 48°C (peak summer)
    'Relative Humidity (%)': (10, 95),   # Monsoon can reach 95%, post-monsoon drops to ~15-20
    'Wind Speed (m/s)': (0, 20),         # Typical max ~12-15 m/s, rare >20
    'Wind Direction (°)': (0, 360)       # Unchanged, always 0-360
}

# Feature mapping for merging columns
FEATURE_MAPPING = {
    "PM2.5 (µg/m³)": ["PM2.5 (ug/m3)"],
    "PM10 (µg/m³)": ["PM10 (ug/m3)"],
    "Nitric Oxide (µg/m³)": ["NO (ug/m3)"],
    "Nitrogen Dioxide (µg/m³)": ["NO2 (ug/m3)"],
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
    level=logging.DEBUG,
    format='%(asctime)s - %(message)s',
    filename=LOG_FILE,
    filemode='a',  # Append mode to keep all logs
    force=True
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add stream handler to see logs in terminal too
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
