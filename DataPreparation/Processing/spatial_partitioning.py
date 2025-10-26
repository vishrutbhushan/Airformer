import os
import numpy as np
import pickle
from math import radians, sin, cos, sqrt, atan2, degrees
from config import logger

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def get_bearing(lat1, lon1, lat2, lon2):
    import math
    # Check for NaN or invalid coordinates
    if any(math.isnan(x) or x is None for x in [lat1, lon1, lat2, lon2]):
        logger.warning(f"Invalid coordinates in get_bearing: ({lat1}, {lon1}) -> ({lat2}, {lon2})")
        return float('nan')
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    initial_bearing = atan2(x, y)
    bearing = (degrees(initial_bearing) + 360) % 360
    return bearing

def get_sector(bearing, n_sectors=8):
    import math
    sector_size = 360.0 / n_sectors
    if bearing is None or math.isnan(bearing):
        logger.warning(f"NaN bearing encountered in get_sector; assigning to sector 0.")
        return 0
    sector = int((bearing + sector_size / 2) % 360 / sector_size)
    return sector

def create_dartboard(stations, radii, n_sectors=8):
    n = len(stations)
    n_rings = len(radii)
    n_regions = 1 + n_rings * n_sectors
    logger.info(f"Computing distance matrix for {n} stations...")
    dist_matrix = np.zeros((n, n))
    bearing_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = haversine_distance(
                stations[i]['latitude'], stations[i]['longitude'],
                stations[j]['latitude'], stations[j]['longitude']
            )
            b = get_bearing(
                stations[i]['latitude'], stations[i]['longitude'],
                stations[j]['latitude'], stations[j]['longitude']
            )
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
            bearing_matrix[i, j] = b
            bearing_matrix[j, i] = (b + 180) % 360
    logger.info(f"Creating {n_rings} rings with radii: {radii} km and {n_sectors} sectors")
    # Initialize assignment and mask matrices with the updated n_regions
    assignment = np.zeros((n, n, n_regions))
    mask = np.ones((n, n_regions), dtype=bool)
    # Assign the query station itself to region 0
    for i in range(n):
        assignment[i, i, 0] = 1.0
        mask[i, 0] = False
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = dist_matrix[i, j]
            bearing = bearing_matrix[i, j]
            # identical coordinates -> assign to region 0
            if np.isclose(d, 0.0):
                region_idx = 0
                assignment[i, j, region_idx] = 1.0
                mask[i, region_idx] = False
                continue

            # find ring: ring numbers start at 1 for first radius (0-50 km)
            ring_number = None
            for r_idx, radius in enumerate(radii):
                if d < radius:
                    ring_number = r_idx + 1
                    break
            # if beyond largest radius, skip assignment (least importance)
            if ring_number is None:
                continue
            sector_idx = get_sector(bearing, n_sectors)
            # region indexing: 0 reserved, then 1.. map to rings/sectors
            region_idx = 1 + (ring_number - 1) * n_sectors + sector_idx
            assignment[i, j, region_idx] = 1.0
            mask[i, region_idx] = False
    logger.info("Normalizing region assignments...")
    for i in range(n):
        for region in range(n_regions):
            total = assignment[i, :, region].sum()
            if total > 0:
                assignment[i, :, region] /= total
    for region in range(n_regions):
        active = (~mask[:, region]).sum()
        if region == 0:
            logger.info(f"Region {region} (Station itself / identical coords): {active} stations have connections")
            continue
        # compute human-friendly ring/sector
        rel = region - 1
        ring_idx = rel // n_sectors  # 0-based index into radii
        sector_idx = rel % n_sectors
        if ring_idx < len(radii):
            if ring_idx == 0:
                ring_desc = f"0-{radii[ring_idx]}km"
            else:
                ring_desc = f"{radii[ring_idx-1]}-{radii[ring_idx]}km"
            logger.info(f"Region {region} (Ring {ring_idx+1}: {ring_desc}, Sector {sector_idx}): {active} stations have connections")
        else:
            logger.info(f"Region {region} (Ring {ring_idx+1}: >{radii[-1] if radii else 0}km, Sector {sector_idx}): {active} stations have connections")
    return assignment, mask

def create_dartboard_partitions(metadata_file, output_base_dir="./Dataset/INDIAN_AIR/local_partition"):
    logger.info("Creating basic dartboard partitions")
    logger.info(f"Loading station metadata from: {metadata_file}")
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    stations = metadata['stations']
    # Validate station coordinates and filter out invalid entries
    import math
    valid_stations = []
    invalid_entries = []
    for s in stations:
        lat = s.get('latitude')
        lon = s.get('longitude')
        try:
            latf = float(lat)
            lonf = float(lon)
        except Exception:
            invalid_entries.append(s.get('station_code') or s.get('station_code', '<unknown>'))
            continue
        if math.isnan(latf) or math.isnan(lonf):
            invalid_entries.append(s.get('station_code') or s.get('station_code', '<unknown>'))
            continue
        valid_stations.append({'station_code': s.get('station_code'), 'latitude': latf, 'longitude': lonf})
    n_stations = len(valid_stations)
    logger.info(f"Loaded {len(stations)} stations from metadata, {n_stations} have valid coordinates")
    if invalid_entries:
        logger.warning(f"Filtered out {len(invalid_entries)} stations with invalid coords: {invalid_entries[:10]}{'...' if len(invalid_entries)>10 else ''}")
    logger.info(f"Time range: {metadata.get('time_range', 'N/A')}")
    logger.info(f"Features: {len(metadata['features'])}")
    configs = [
        ("50", [50]),
        ("50-200", [50, 200]),
        ("50-200-500", [50, 200, 500])
    ]
    os.makedirs(output_base_dir, exist_ok=True)
    for name, radii in configs:
        logger.info(f"\nCreating partition: {name} (radii: {radii} km)")
        if n_stations == 0:
            logger.error("No stations with valid coordinates available to create partitions. Skipping.")
            continue
        assignment, mask = create_dartboard(valid_stations, radii)
        output_dir = os.path.join(output_base_dir, name)
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, 'assignment.npy'), assignment)
        np.save(os.path.join(output_dir, 'mask.npy'), mask)
        logger.info(f"Saved to {output_dir}: assignment{assignment.shape}, mask{mask.shape}")
    logger.info("Dartboard partitioning complete.")
