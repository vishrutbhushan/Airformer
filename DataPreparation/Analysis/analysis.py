import pandas as pd
import numpy as np
from pathlib import Path
import logging

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

# Log results
logging.info(f"  Start Date: {start_date}")
logging.info(f"  End Date: {end_date}")
logging.info(f"  Total Rows: {total_rows}")
logging.info(f"  Number of Features: {len(features)}")
logging.info(f"  Null Percentage: {null_percentage:.2f}%")
logging.info(f"  Feature Completeness :")
for feature, completeness in sorted_features:
    logging.info(f"    {feature}: {completeness:.2f}%")
