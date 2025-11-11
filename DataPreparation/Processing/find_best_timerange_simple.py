#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import random
import logging
from config import CORE_FEATURES, FEATURE_MAPPING

# Setup logging with both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('timerange_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DATA_DIR = Path('../Data')
START_DATE = pd.Timestamp('2010-01-01')
END_DATE = pd.Timestamp('2023-03-31')

def get_feature_cols(df_columns):
    possible_cols = []
    for core_feature, alt_names in FEATURE_MAPPING.items():
        for alt_name in alt_names:
            if alt_name in df_columns:
                possible_cols.append(alt_name)
                break
    return possible_cols

def has_all_features(csv_path):
    try:
        df = pd.read_csv(csv_path, nrows=1)
        cols = get_feature_cols(df.columns)
        return len(cols) == len(CORE_FEATURES)
    except:
        return False

def get_date_col(df):
    for col in df.columns:
        if col.lower() in ['from date', 'datetime', 'date']:
            return col
    return df.columns[0]

# Find valid stations
logger.info("Step 1: Finding stations with all core features...")
csv_files = sorted([f for f in DATA_DIR.glob('*.csv') if f.name != 'stations_info.csv'])
valid_stations = [f for f in csv_files if has_all_features(f)]
logger.info(f"Found {len(valid_stations)} valid stations\n")

# Generate year-month range
logger.info("Step 2: Generating time periods...")
year_months = pd.date_range(START_DATE, END_DATE, freq='MS')
logger.info(f"Analyzing {len(year_months)} months from {START_DATE.date()} to {END_DATE.date()}\n")

# Check coverage for each month
logger.info("Step 3: Checking coverage by month...\n")
results = []

for idx, month_start in enumerate(year_months):
    month_end = month_start + pd.DateOffset(months=1) - pd.Timedelta(hours=1)
    
    if (idx + 1) % 12 == 0:
        logger.info(f"Processing {month_start.strftime('%Y-%m')} ({idx + 1}/{len(year_months)})")
    
    coverage_count = 0
    
    for station_path in valid_stations:
        try:
            df = pd.read_csv(station_path)
            date_col = get_date_col(df)
            df['dt'] = pd.to_datetime(df[date_col], errors='coerce')
            df_month = df[(df['dt'] >= month_start) & (df['dt'] <= month_end)]
            
            if len(df_month) > 0:
                # Random sample from month
                sample_df = df_month.sample(n=min(20, len(df_month)))
                cols = get_feature_cols(df.columns)
                has_data = (sample_df[cols].notna().sum().sum() > 0)
                if has_data:
                    coverage_count += 1
        except:
            pass
    
    pct = (coverage_count / len(valid_stations)) * 100
    year = month_start.year
    results.append({
        'date': month_start,
        'year': year,
        'month': month_start.month,
        'stations': coverage_count,
        'pct': pct
    })

# Print results by year
logger.info("\n" + "="*80)
logger.info("YEAR-WISE COVERAGE ANALYSIS")
logger.info("="*80 + "\n")

for year in sorted(set(r['year'] for r in results)):
    year_data = [r for r in results if r['year'] == year]
    avg_pct = np.mean([r['pct'] for r in year_data])
    min_pct = np.min([r['pct'] for r in year_data])
    max_pct = np.max([r['pct'] for r in year_data])
    
    logger.info(f"{year}: Avg={avg_pct:.1f}% | Min={min_pct:.1f}% | Max={max_pct:.1f}%")

logger.info("\n" + "="*80 + "\n")
