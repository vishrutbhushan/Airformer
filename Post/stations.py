import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(__file__)
COORDS_CSV = os.path.join(BASE_DIR, "../DataPreparation/SpatialDataGeneration/stations_with_coords.csv")
DATAPREPROCESSED_DIR = os.path.join(BASE_DIR, "../DataPreparation/DataPreprocessed")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG_ALL = os.path.join(OUT_DIR, "stations_map_all.png")
OUT_PNG_PROCESSED = os.path.join(OUT_DIR, "stations_map_processed.png")
OUT_PNG_COMPARISON = os.path.join(OUT_DIR, "stations_comparison.png")

INDIA_GEOJSON = os.path.join(BASE_DIR, "india_coarse.geojson")


def extract_station_code(filename):
    """Extract station code from filename (e.g., 'AP001.csv' -> 'AP001')"""
    return os.path.splitext(filename)[0]


def load_all_stations():
    """Load all stations from the coordinates CSV file"""
    if not os.path.exists(COORDS_CSV):
        print(f"Coordinates CSV not found: {COORDS_CSV}")
        return None
    
    try:
        df = pd.read_csv(COORDS_CSV)
        # Ensure we have the required columns
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            print("CSV does not contain latitude/longitude columns")
            return None
        
        df = df.dropna(subset=["latitude", "longitude"])
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df = df.dropna(subset=["latitude", "longitude"])
        
        return df if len(df) > 0 else None
    except Exception as e:
        print(f"Error loading coordinates CSV: {e}")
        return None


def load_processed_stations():
    """Load stations that were actually used (from DataPreprocessed folder)"""
    if not os.path.exists(DATAPREPROCESSED_DIR):
        print(f"DataPreprocessed directory not found: {DATAPREPROCESSED_DIR}")
        return None
    
    # Get all CSV files in DataPreprocessed folder
    csv_files = [f for f in os.listdir(DATAPREPROCESSED_DIR) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in: {DATAPREPROCESSED_DIR}")
        return None
    
    # Extract station codes from filenames
    processed_codes = set(extract_station_code(f) for f in csv_files)
    print(f"   Found {len(processed_codes)} unique station codes in DataPreprocessed")
    
    # Load all stations and filter to only processed ones
    df_all = load_all_stations()
    if df_all is None:
        return None
    
    # Match stations using file_name column
    df_processed = df_all[df_all['file_name'].isin(processed_codes)].copy()
    
    return df_processed if len(df_processed) > 0 else None


def plot_india_boundary(ax):
    """Helper function to plot India boundary"""
    try:
        import json
        if os.path.exists(INDIA_GEOJSON):
            with open(INDIA_GEOJSON, "r", encoding="utf-8") as f:
                gj = json.load(f)
            for feature in gj.get("features", []):
                geom = feature.get("geometry")
                if not geom:
                    continue
                coords = []
                if geom["type"] == "Polygon":
                    coords = geom["coordinates"]
                elif geom["type"] == "MultiPolygon":
                    coords = [poly for poly in geom["coordinates"]]
                # plot outer rings
                def plot_ring(ring):
                    xs = [p[0] for p in ring]
                    ys = [p[1] for p in ring]
                    ax.plot(xs, ys, color="#444444", linewidth=0.8)

                if geom["type"] == "Polygon":
                    for ring in coords:
                        plot_ring(ring)
                else:
                    for poly in coords:
                        for ring in poly:
                            plot_ring(ring)
    except Exception:
        pass


def plot_static(df, output_file, title="Monitoring Stations (India)", color="red"):
    """Plot stations on a map"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    
    plot_india_boundary(ax)

    # Scatter stations
    ax.scatter(df["longitude"], df["latitude"], s=20, c=color, alpha=0.8, edgecolors="k", linewidth=0.3)

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(60, 100)
    ax.set_ylim(6, 38)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def load_processed_stations(datapreprocessed_dir):
    """Load stations that were actually used (from DataPreprocessed folder)"""
    if not os.path.exists(datapreprocessed_dir):
        print(f"DataPreprocessed directory not found: {datapreprocessed_dir}")
        return None
    
    # Get all CSV files in DataPreprocessed folder
    csv_files = [f for f in os.listdir(datapreprocessed_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in: {datapreprocessed_dir}")
        return None
    
    # Extract station codes from filenames
    processed_codes = set(extract_station_code(f) for f in csv_files)
    print(f"   Found {len(processed_codes)} unique station codes in DataPreprocessed")
    
    # Load all stations and filter to only processed ones
    df_all = load_all_stations()
    if df_all is None:
        return None
    
    # Match stations using file_name column
    df_processed = df_all[df_all['file_name'].isin(processed_codes)].copy()
    
    return df_processed if len(df_processed) > 0 else None


def plot_comparison(df_all, df_processed):
    """Plot comparison between all stations and processed stations"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
    
    # Plot all stations
    plot_india_boundary(ax1)
    ax1.scatter(df_all["longitude"], df_all["latitude"], s=20, c="lightcoral", alpha=0.6, edgecolors="k", linewidth=0.3, label="All stations")
    ax1.set_title(f"All Stations (n={len(df_all)})")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_xlim(60, 100)
    ax1.set_ylim(6, 38)
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend()
    
    # Plot processed stations
    plot_india_boundary(ax2)
    ax2.scatter(df_processed["longitude"], df_processed["latitude"], s=20, c="green", alpha=0.8, edgecolors="k", linewidth=0.3, label="Kept stations")
    ax2.set_title(f"Stations After Preprocessing (n={len(df_processed)})")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_xlim(60, 100)
    ax2.set_ylim(6, 38)
    ax2.set_aspect("equal", adjustable="box")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    fig.savefig(OUT_PNG_COMPARISON, dpi=200)
    plt.close(fig)
    
    # Calculate statistics
    n_removed = len(df_all) - len(df_processed)
    pct_kept = (len(df_processed) / len(df_all)) * 100 if len(df_all) > 0 else 0
    
    print(f"\nStation Statistics:")
    print(f"  Original stations: {len(df_all)}")
    print(f"  Stations kept: {len(df_processed)}")
    print(f"  Stations removed: {n_removed}")
    print(f"  Percentage kept: {pct_kept:.1f}%")


if __name__ == "__main__":
    print("="*70)
    print("STATION VISUALIZATION")
    print("="*70)
    
    # Load all stations from Data folder
    print("\n1. Loading all stations from Data folder...")
    df_all = load_all_stations()
    
    if df_all is None:
        print("   ✗ Could not load any stations from Data folder")
        exit(1)
    
    print(f"   ✓ Loaded {len(df_all)} stations from Data folder")
    
    # Plot all stations
    print("\n2. Plotting all stations...")
    plot_static(df_all, OUT_PNG_ALL, title="All Monitoring Stations (India)", color="lightcoral")
    print(f"   ✓ Saved to: {OUT_PNG_ALL}")
    
    # Load processed stations from DataPreprocessed folder
    print("\n3. Loading stations used in preprocessing (from DataPreprocessed folder)...")
    df_processed = load_processed_stations(DATAPREPROCESSED_DIR)
    
    if df_processed is not None and len(df_processed) > 0:
        print(f"   ✓ Loaded {len(df_processed)} processed stations")
        
        # Plot processed stations
        print("\n4. Plotting processed stations...")
        plot_static(df_processed, OUT_PNG_PROCESSED, 
                   title=f"Stations Used in Preprocessing (n={len(df_processed)})", 
                   color="green")
        print(f"   ✓ Saved to: {OUT_PNG_PROCESSED}")
        
        # Plot comparison
        print("\n5. Creating comparison plot...")
        plot_comparison(df_all, df_processed)
        print(f"   ✓ Saved to: {OUT_PNG_COMPARISON}")

    else:
        print("   ✗ Could not load processed stations from DataPreprocessed folder")
        print(f"   ✗ Please ensure preprocessing has been run and DataPreprocessed folder exists at:")
        print(f"     {DATAPREPROCESSED_DIR}")
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
