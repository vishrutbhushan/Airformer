"""
Visualize PM2.5 measurements over timesteps for each preprocessed station.
Generates graphs showing temporal patterns and data quality.
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path

# Configuration
INPUT_FOLDER = "../DataPreprocessed"
OUTPUT_FOLDER = "../Visualizations"
PM25_COLUMN = "PM2.5 (µg/m³)"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Find all preprocessed station files
files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.csv")))

if not files:
    print(f"No CSV files found in {INPUT_FOLDER}")
    exit(1)

print(f"Found {len(files)} preprocessed station files")
print(f"Generating visualizations in {OUTPUT_FOLDER}...\n")

successful = 0
failed = 0

for idx, file in enumerate(files):
    station_code = Path(file).stem
    
    try:
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Check if PM2.5 column exists
        if PM25_COLUMN not in df.columns:
            print(f"[{idx+1}/{len(files)}] ⚠️  {station_code}: No {PM25_COLUMN} column")
            failed += 1
            continue
        
        # Get PM2.5 data
        pm25_data = df[PM25_COLUMN]
        
        # Check for nulls
        null_count = pm25_data.isna().sum()
        if null_count > 0:
            print(f"[{idx+1}/{len(files)}] ⚠️  {station_code}: {null_count} null values in PM2.5!")
            failed += 1
            continue
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 5))
        
        # Plot PM2.5 data
        ax.plot(pm25_data.values, linewidth=0.8, color='#FF6B6B', alpha=0.8)
        
        # Styling
        ax.set_title(f"PM2.5 Measurements over Time - Station {station_code}", 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("Timesteps (3-hourly intervals)", fontsize=11)
        ax.set_ylabel(f"{PM25_COLUMN}", fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add statistics to plot
        mean_val = pm25_data.mean()
        min_val = pm25_data.min()
        max_val = pm25_data.max()
        std_val = pm25_data.std()
        
        stats_text = f"Mean: {mean_val:.1f} | Min: {min_val:.1f} | Max: {max_val:.1f} | Std: {std_val:.1f}"
        ax.text(0.5, -0.12, stats_text, transform=ax.transAxes, 
               fontsize=10, ha='center', 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Save figure
        output_file = os.path.join(OUTPUT_FOLDER, f"{station_code}_pm25.png")
        plt.tight_layout()
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"[{idx+1}/{len(files)}] ✓ {station_code}: {len(df)} timesteps | "
              f"Mean PM2.5: {mean_val:.1f} µg/m³")
        successful += 1
        
    except Exception as e:
        print(f"[{idx+1}/{len(files)}] ✗ {station_code}: Error - {str(e)}")
        failed += 1
        continue

print(f"\n{'='*60}")
print(f"Visualization Complete!")
print(f"{'='*60}")
print(f"Successfully generated: {successful} graphs")
print(f"Failed: {failed}")
print(f"Output folder: {OUTPUT_FOLDER}")
print(f"{'='*60}")
