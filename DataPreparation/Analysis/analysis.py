import pandas as pd
import numpy as np
from pathlib import Path
import logging
from pandas.tseries.offsets import QuarterBegin

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filename='log.txt',
    filemode='w'
)

data_dir = Path('../Data')
station_files = sorted(data_dir.glob("*.csv"))

all_data = [pd.read_csv(file, parse_dates=['From Date']).assign(station_code=file.stem) for file in station_files]

combined_df = pd.concat(all_data, ignore_index=True)

# Perform collective analysis
start_date = combined_df['From Date'].min()
end_date = combined_df['From Date'].max()
total_rows = len(combined_df)

features = combined_df.columns.tolist()

# Calculate feature completeness
feature_completeness = {feature: 100 * (1 - combined_df[feature].isna().sum() / total_rows) for feature in features}
null_percentage = 100 * combined_df.isna().sum().sum() / (total_rows * len(combined_df.columns))

# Sort features by data availability
sorted_features = sorted(feature_completeness.items(), key=lambda x: x[1], reverse=True)

# QUARTERLY MISSING TIMESTEP ANALYSIS (Vectorized, per quarter)

# Efficient per-quarter missing timestep analysis
time_grid = pd.date_range(start=start_date, end=end_date, freq='1h')  # 1-hourly
data_quarters = pd.DatetimeIndex(time_grid).to_period('Q')
station_codes = combined_df['station_code'].unique().tolist()
raw_features = [f for f in combined_df.columns if f not in ['From Date', 'To Date', 'station_code', 'datetime']]

combined_df['datetime'] = pd.to_datetime(combined_df['From Date'], errors='coerce')
df_actual = combined_df.set_index(['station_code', 'datetime'])

quarter_labels = data_quarters.astype(str).unique().tolist()
logging.info("Quarterly missing timestep summary (vectorized, per quarter):")
for q in quarter_labels:
    quarter_mask = (data_quarters.astype(str) == q)
    quarter_times = time_grid[quarter_mask]
    quarter_index = pd.MultiIndex.from_product([station_codes, quarter_times], names=['station_code', 'datetime'])
    # Reindex only for this quarter
    df_quarter = df_actual.reindex(quarter_index)
    # Missing if all raw features are NaN for a timestep
    missing_mask = df_quarter[raw_features].isna().all(axis=1)
    missing_count = missing_mask.sum()
    total_expected = len(quarter_index)
    percent_missing = (missing_count / total_expected) * 100 if total_expected > 0 else 0.0
    logging.info(f"Quarter {q}: {missing_count}/{total_expected} timesteps missing ({percent_missing:.2f}%)")

# Log results
logging.info(f"  Start Date: {start_date}")
logging.info(f"  End Date: {end_date}")
logging.info(f"  Total Rows: {total_rows}")
logging.info(f"  Number of Features: {len(features)}")
logging.info(f"  Null Percentage: {null_percentage:.2f}%")
logging.info(f"  Feature Completeness :")
for feature, completeness in sorted_features:
    logging.info(f"    {feature}: {completeness:.2f}%")
