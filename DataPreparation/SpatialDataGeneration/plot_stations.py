import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)
INPUT_CSV = os.path.join(BASE_DIR, "stations_with_coords.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "stations_map.png")

INDIA_GEOJSON = os.path.join(BASE_DIR, "india_coarse.geojson")


def load_stations(path):
    df = pd.read_csv(path, dtype={"file_name": str})
    df = df.dropna(subset=["latitude", "longitude"])  # drop missing
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])  # ensure numeric
    return df


def plot_static(df):
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
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

    # Scatter stations
    ax.scatter(df["longitude"], df["latitude"], s=20, c="red", alpha=0.8, edgecolors="k", linewidth=0.3)

    ax.set_title("Monitoring Stations (India)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(60, 100)
    ax.set_ylim(6, 38)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    df = load_stations(INPUT_CSV)
    plot_static(df)
