"""
Create a summary dashboard showing PM2.5 for all stations in one view.
Also generates statistics CSV for analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path
import numpy as np

# Configuration
INPUT_FOLDER = "../DataPreprocessed"
OUTPUT_FOLDER = "../Visualizations"
PM25_COLUMN = "PM2.5 (µg/m³)"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Find all files
files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.csv")))

if not files:
    print(f"No CSV files found in {INPUT_FOLDER}")
    exit(1)

print(f"Found {len(files)} preprocessed station files")
print(f"Generating summary dashboard...\n")

# Collect statistics for all stations
stats_list = []
all_pm25_data = {}

for idx, file in enumerate(files):
    station_code = Path(file).stem
    
    try:
        df = pd.read_csv(file)
        
        if PM25_COLUMN not in df.columns:
            continue
        
        pm25_data = df[PM25_COLUMN]
        null_count = pm25_data.isna().sum()
        
        if null_count > 0:
            print(f"⚠️  {station_code}: {null_count} nulls - SKIPPING")
            continue
        
        all_pm25_data[station_code] = pm25_data.values
        
        # Collect statistics
        stats_list.append({
            'Station': station_code,
            'Timesteps': len(df),
            'Mean_PM25': pm25_data.mean(),
            'Median_PM25': pm25_data.median(),
            'Min_PM25': pm25_data.min(),
            'Max_PM25': pm25_data.max(),
            'Std_PM25': pm25_data.std(),
            'Nulls': null_count
        })
        
        print(f"✓ {station_code}: Mean={pm25_data.mean():.1f}, "
              f"Min={pm25_data.min():.1f}, Max={pm25_data.max():.1f}")
    
    except Exception as e:
        print(f"✗ {station_code}: {str(e)}")
        continue

if not all_pm25_data:
    print("No valid data found!")
    exit(1)

print(f"\n{'='*60}")
print(f"Creating visualizations for {len(all_pm25_data)} stations...")
print(f"{'='*60}\n")

# 1. Create overlay plot of all PM2.5 trends (normalized)
print("1. Creating normalized overlay plot...")
fig, ax = plt.subplots(figsize=(16, 7))

colors = plt.cm.tab20c(np.linspace(0, 1, len(all_pm25_data)))

for (station, data), color in zip(all_pm25_data.items(), colors):
    # Normalize to 0-1 for comparison
    normalized = (data - data.min()) / (data.max() - data.min() + 1e-6)
    ax.plot(normalized, label=station, linewidth=0.7, alpha=0.7, color=color)

ax.set_title("Normalized PM2.5 Trends Overlay (All Stations)", 
            fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel("Timesteps (3-hourly intervals)", fontsize=11)
ax.set_ylabel("Normalized PM2.5 (0-1)", fontsize=11)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "all_stations_pm25_normalized.png"), 
           dpi=100, bbox_inches='tight')
plt.close()
print("   ✓ Saved: all_stations_pm25_normalized.png")

# 2. Create box plot comparison
print("2. Creating box plot comparison...")
fig, ax = plt.subplots(figsize=(14, 6))

data_for_box = [all_pm25_data[station] for station in sorted(all_pm25_data.keys())]
stations_sorted = sorted(all_pm25_data.keys())

bp = ax.boxplot(data_for_box, labels=stations_sorted, patch_artist=True)

# Color the boxes
for patch in bp['boxes']:
    patch.set_facecolor('#FF6B6B')
    patch.set_alpha(0.7)

ax.set_title("PM2.5 Distribution Comparison Across All Stations", 
            fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel("Station", fontsize=11)
ax.set_ylabel(f"{PM25_COLUMN}", fontsize=11)
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "all_stations_pm25_boxplot.png"), 
           dpi=100, bbox_inches='tight')
plt.close()
print("   ✓ Saved: all_stations_pm25_boxplot.png")

# 3. Create statistics CSV
print("3. Creating statistics CSV...")
stats_df = pd.DataFrame(stats_list).sort_values('Mean_PM25', ascending=False)
stats_csv = os.path.join(OUTPUT_FOLDER, "pm25_statistics.csv")
stats_df.to_csv(stats_csv, index=False)
print(f"   ✓ Saved: pm25_statistics.csv ({len(stats_df)} stations)")

# 4. Create mean/std distribution plot
print("4. Creating mean/std distribution plot...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Mean PM2.5
ax1.barh(stats_df['Station'], stats_df['Mean_PM25'], color='#FF6B6B', alpha=0.7)
ax1.set_title("Mean PM2.5 by Station", fontsize=12, fontweight='bold')
ax1.set_xlabel("Mean PM2.5 (µg/m³)", fontsize=10)
ax1.grid(True, alpha=0.3, axis='x')

# Std PM2.5
ax2.barh(stats_df['Station'], stats_df['Std_PM25'], color='#4ECDC4', alpha=0.7)
ax2.set_title("PM2.5 Variability (Std Dev) by Station", fontsize=12, fontweight='bold')
ax2.set_xlabel("Std Dev (µg/m³)", fontsize=10)
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "pm25_mean_std_comparison.png"), 
           dpi=100, bbox_inches='tight')
plt.close()
print("   ✓ Saved: pm25_mean_std_comparison.png")

print(f"\n{'='*60}")
print(f"Summary Statistics:")
print(f"{'='*60}")
print(f"Total stations processed: {len(all_pm25_data)}")
print(f"Global Mean PM2.5: {stats_df['Mean_PM25'].mean():.1f} µg/m³")
print(f"Global Min PM2.5: {stats_df['Min_PM25'].min():.1f} µg/m³")
print(f"Global Max PM2.5: {stats_df['Max_PM25'].max():.1f} µg/m³")
print(f"\nTop 5 Highest Mean PM2.5:")
for idx, row in stats_df.head(5).iterrows():
    print(f"  {row['Station']}: {row['Mean_PM25']:.1f} µg/m³")
print(f"\nBottom 5 Lowest Mean PM2.5:")
for idx, row in stats_df.tail(5).iterrows():
    print(f"  {row['Station']}: {row['Mean_PM25']:.1f} µg/m³")
print(f"{'='*60}")
print(f"All visualizations saved to: {OUTPUT_FOLDER}")
print(f"{'='*60}")
