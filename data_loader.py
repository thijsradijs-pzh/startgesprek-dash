# data_loader.py

import os
import pandas as pd
import geopandas as gpd
import networkx as nx
import json

from pysal.lib import weights

from utils import CSV_FOLDER_PATH, clean_dataset_name

# Load GeoDataFrame from shapefile or GeoJSON
def load_gdf(gdf_path):
    """Loads a GeoDataFrame from a shapefile or GeoJSON."""
    gdf = gpd.read_file(gdf_path)
    if 'hex9' not in gdf.columns:
        print("The GeoJSON must contain a 'hex9' column.")
        return None
    gdf = gdf.set_index('hex9')
    # Ensure the GeoDataFrame is in WGS84 CRS
    gdf = gdf.to_crs(epsg=4326)
    return gdf

# Load selected CSVs
def load_selected_csvs(folder_path, selected_csvs, all_hexagons):
    """Loads specified CSV files from the folder."""
    dataframes = {}
    for file in selected_csvs:
        file_name = f"{file}.csv"
        file_path = os.path.join(folder_path, file_name)
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                if 'hex9' not in df.columns:
                    print(f"'hex9' column not found in {file_name}")
                    continue
                if 'value' not in df.columns:
                    print(f"'value' column not found in {file_name}")
                    continue
                df['hex9'] = df['hex9'].astype(str)
                df = df.set_index('hex9')
                # Reindex to include all hexagons, fill missing values with zero
                df = df.reindex(all_hexagons, fill_value=0)
                dataframes[file] = df
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
    return dataframes

# Load the data when the module is imported
def load_data():
    # Load the GeoDataFrame
    idx = load_gdf('./app_data/h3_9_zh_delta.geojson')
    if idx is None:
        raise Exception("Failed to load GeoDataFrame")
    all_hexagons = idx.index.tolist()
    # Convert 'hex9' to string
    idx.index = idx.index.astype(str)
    # List of CSV files to load
    selected_csvs = [
        'Natuur',
        'Water',
        'Akkerboeren',
        'Stad'
    ]
    # Load CSV dataframes
    dataframes = load_selected_csvs(CSV_FOLDER_PATH, selected_csvs, all_hexagons)
    # Initialize weights and graph
    w = weights.Queen.from_dataframe(idx, ids=idx.index.tolist())
    try:
        g = nx.read_graphml('./app_data/G.graphml')
    except Exception as e:
        print(f"Error loading graph: {e}")
        g = None
    # Create clean dataset names and mapping
    dataset_names = list(dataframes.keys())
    clean_dataset_names = [clean_dataset_name(ds) for ds in dataset_names]
    clean_names_map = dict(zip(clean_dataset_names, dataset_names))
    # Ensure the GeoDataFrame is in WGS84 CRS
    idx = idx.to_crs(epsg=4326)
    # Prepare GeoJSON for mapping
    idx_reset = idx.reset_index()
    # Ensure 'hex9' is included in properties as string
    idx_reset['hex9'] = idx_reset['hex9'].astype(str)
    idx_json = json.loads(idx_reset.to_json())
    return idx, dataframes, w, g, clean_dataset_names, clean_names_map, idx_json

# Load the data
idx, dataframes, w, g, clean_dataset_names, clean_names_map, idx_json = load_data()
