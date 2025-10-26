import pandas as pd
import googlemaps
import requests
import time
import os
import numpy as np

API_KEY = "AIzaSyDUoXOZnd9WS2PNvKlJ_58C38NNQarteGA"
gmaps = googlemaps.Client(key=API_KEY)

input_csv = "stations_info.csv"
output_csv = "stations_with_coords.csv"

if os.path.exists(output_csv):
    df_existing = pd.read_csv(output_csv)
    failed_mask = (df_existing['latitude'].isna()) | (df_existing['longitude'].isna()) | \
                  (df_existing['latitude'] == 'None') | (df_existing['longitude'] == 'None') | \
                  (df_existing['latitude'] == '') | (df_existing['longitude'] == '')
    failed_indices = df_existing[failed_mask].index.tolist()
    if len(failed_indices) > 0:
        df = df_existing.copy()
        retry_only = True
    else:
        exit()
else:
    df = pd.read_csv(input_csv)
    df['latitude'] = None
    df['longitude'] = None
    failed_indices = list(range(len(df)))
    retry_only = False

processed_count = 0
for idx in failed_indices:
    row = df.iloc[idx]
    query = f"{row['station_location']}, {row['city']}, {row['state']}, India"

    lat, lon = None, None
    try:
        result = gmaps.geocode(query)
        if result:
            loc = result[0]["geometry"]["location"]
            lat, lon = loc["lat"], loc["lng"]
            location_type = result[0]["geometry"].get("location_type", "")
            if location_type in ["APPROXIMATE", "GEOMETRIC_CENTER"]:
                pass
        else:
            city_query = f"{row['city']}, {row['state']}, India"
            city_result = gmaps.geocode(city_query)
            if city_result:
                loc = city_result[0]["geometry"]["location"]
                lat, lon = loc["lat"], loc["lng"]
    except Exception as e:
        pass

    df.at[idx, 'latitude'] = lat
    df.at[idx, 'longitude'] = lon
    
    processed_count += 1
    time.sleep(0.1)

df.to_csv(output_csv, index=False)

final_missing = df[(df['latitude'].isna()) | (df['longitude'].isna()) |
                   (df['latitude'] == 'None') | (df['longitude'] == 'None') |
                   (df['latitude'] == '') | (df['longitude'] == '')]

if len(final_missing) > 0:
    for _, failed_row in final_missing.iterrows():
        pass
