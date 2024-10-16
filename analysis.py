# analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pysal.explore import esda
from pysal.lib import weights

from utils import apply_color_mapping

# Fuzzify each layer
def fuzzify_each_layer(df_list, fuzz_type='close', colormap_name='magma'):
    """Fuzzifies each selected criterion separately and returns a list of fuzzified DataFrames."""
    fuzzified_dataframes = []
    colormap = plt.get_cmap(colormap_name)
    
    for df in df_list:
        df_array = np.array(df['value'])
        # Avoid division by zero
        range_diff = df_array.max() - df_array.min()
        if range_diff == 0:
            # Assign constant fuzzified value
            print(f"Zero range in data for layer. Assigning constant fuzzified value.")
            fuzzified_array = np.ones_like(df_array)
        else:
            # Apply fuzzification depending on the fuzz_type
            if fuzz_type == "close":
                fuzzified_array = np.maximum(0, (df_array - df_array.min()) / range_diff)
            else:
                fuzzified_array = np.maximum(0, 1 - (df_array - df_array.min()) / range_diff)
        
        # Create a new DataFrame for the fuzzified result
        fuzzified_df = df.copy()
        fuzzified_df['fuzzy'] = np.round(fuzzified_array, 3)  # Add the fuzzified values
        
        # Apply the colormap
        apply_color_mapping(fuzzified_df, 'fuzzy', colormap_name)
        
        # Append fuzzified dataframe to the list
        fuzzified_dataframes.append(fuzzified_df.reset_index())
    
    return fuzzified_dataframes

# Stack fuzzified layers
def stack_fuzzified_layers(fuzzified_dataframes, suffix=''):
    """Stacks multiple fuzzified DataFrames by joining them on 'hex9' index."""
    # Start with the first DataFrame in the list
    stacked_df = fuzzified_dataframes[0][['hex9', 'fuzzy']].copy()
    stacked_df.rename(columns={'fuzzy': f'fuzzy_1{suffix}'}, inplace=True)  # Rename the fuzzy column to include suffix

    # Add remaining fuzzified DataFrames
    for i, df in enumerate(fuzzified_dataframes[1:], start=2):
        df = df[['hex9', 'fuzzy']].copy()
        df.rename(columns={'fuzzy': f'fuzzy_{i}{suffix}'}, inplace=True)
        stacked_df = pd.merge(stacked_df, df, on='hex9', how='outer')

    # Sum the fuzzified columns to get an overall score for each hexagon
    fuzzy_cols = [col for col in stacked_df.columns if col.startswith('fuzzy_')]
    stacked_df[f'fuzzy_sum{suffix}'] = stacked_df[fuzzy_cols].sum(axis=1)

    return stacked_df

# Perform spatial analysis on the stacked DataFrame
def perform_spatial_analysis_on_stack(stacked_df, idx, w, g, seed=42):
    """Performs spatial suitability analysis on the stacked DataFrame with multiple fuzzified layers."""
    # Drop rows with NaN values and ensure alignment with the spatial index
    stacked_df = stacked_df.dropna(subset=stacked_df.filter(like='fuzzy').columns).set_index('hex9')
    stacked_df = stacked_df.loc[stacked_df.index.intersection(idx.index)]
    
    # Create a new weights object based on the subset of data
    try:
        idx_subset = idx.loc[stacked_df.index]
        w_subset_result = weights.Queen.from_dataframe(idx_subset, ids=idx_subset.index.tolist())
    except Exception as e:
        print(f"Error creating spatial weights subset: {e}")
        return pd.DataFrame()
    
    # Ensure 'fuzzy_sum' is calculated
    if 'fuzzy_sum' not in stacked_df.columns:
        fuzzy_cols = [col for col in stacked_df.columns if col.startswith('fuzzy_') and not col.startswith('fuzzy_sum')]
        stacked_df['fuzzy_sum'] = stacked_df[fuzzy_cols].sum(axis=1)
    
    # Check for variance in 'fuzzy_sum'
    if stacked_df['fuzzy_sum'].var() == 0:
        print("No variability in 'fuzzy_sum'. Moran's I cannot be computed.")
        return pd.DataFrame()
    
    # Perform Moran's I spatial autocorrelation analysis on the summed fuzzy values
    try:
        lisa = esda.Moran_Local(stacked_df['fuzzy_sum'], w_subset_result, seed=seed)
    except Exception as e:
        print(f"Error performing Moran's I analysis: {e}")
        return pd.DataFrame()
    
    significant_locations = stacked_df[(lisa.q == 1) & (lisa.p_sim < 0.01)].index
    significant_df = stacked_df.loc[significant_locations]
    
    if not significant_df.empty:
        most_relevant_locations = significant_df
    else:
        print("No significant locations found after applying filters.")
        return pd.DataFrame()
    
    return most_relevant_locations
