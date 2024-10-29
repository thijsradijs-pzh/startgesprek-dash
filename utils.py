# utils.py

import geopandas as gpd
import numpy as np

# Constants
CSV_FOLDER_PATH = './app_data/'
COLORMAP = 'viridis'

def calculate_view_state(gdf):
    """Calculate view state (zoom and center) based on GeoJSON bounds."""
    bounds = gdf.total_bounds  # Returns (minx, miny, maxx, maxy)
    
    # Calculate center
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2
    
    # Calculate zoom level
    # Using the Mercator projection formula to estimate appropriate zoom
    lat_diff = bounds[3] - bounds[1]
    lon_diff = bounds[2] - bounds[0]
    
    # Max difference in degrees
    max_diff = max(lat_diff, lon_diff)
    
    # Estimate zoom level (smaller diff = higher zoom)
    # These values are tuned for typical web maps
    if max_diff < 0.1:
        zoom = 13
    elif max_diff < 0.5:
        zoom = 11
    elif max_diff < 1.0:
        zoom = 10
    elif max_diff < 2.0:
        zoom = 9
    elif max_diff < 5.0:
        zoom = 8
    else:
        zoom = 7
        
    return {
        'longitude': float(center_lon),
        'latitude': float(center_lat),
        'zoom': zoom
    }

def clean_dataset_name(name):
    """Replaces underscores with spaces and capitalizes for cleaner display."""
    return ' '.join(word.capitalize() for word in name.replace('_', ' ').split())