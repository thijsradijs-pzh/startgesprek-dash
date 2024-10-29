# analysis.py

import numpy as np
import pandas as pd

def fuzzify_each_layer(df_list, fuzz_type='close'):
    """Fuzzifies each selected criterion separately."""
    fuzzified_dataframes = []
    
    for df in df_list:
        df_array = np.array(df['value'])
        range_diff = df_array.max() - df_array.min()
        
        if range_diff == 0:
            fuzzified_array = np.ones_like(df_array)
        else:
            if fuzz_type == "close":
                fuzzified_array = np.maximum(0, (df_array - df_array.min()) / range_diff)
            else:
                fuzzified_array = np.maximum(0, 1 - (df_array - df_array.min()) / range_diff)
        
        fuzzified_df = df.copy()
        fuzzified_df['fuzzy'] = np.round(fuzzified_array, 3)
        fuzzified_dataframes.append(fuzzified_df.reset_index())
    
    return fuzzified_dataframes

def stack_fuzzified_layers(fuzzified_dataframes, suffix=''):
    """Stacks multiple fuzzified DataFrames."""
    if not fuzzified_dataframes:
        return None
        
    stacked_df = fuzzified_dataframes[0][['hex9', 'fuzzy']].copy()
    stacked_df.rename(columns={'fuzzy': f'fuzzy_1{suffix}'}, inplace=True)

    for i, df in enumerate(fuzzified_dataframes[1:], start=2):
        df = df[['hex9', 'fuzzy']].copy()
        df.rename(columns={'fuzzy': f'fuzzy_{i}{suffix}'}, inplace=True)
        stacked_df = pd.merge(stacked_df, df, on='hex9', how='outer')

    fuzzy_cols = [col for col in stacked_df.columns if col.startswith('fuzzy_')]
    stacked_df[f'fuzzy_sum{suffix}'] = stacked_df[fuzzy_cols].sum(axis=1)

    return stacked_df

def analyze_locations(stacked_df):
    """Analyze and find suitable locations based on fuzzy sum values."""
    if 'fuzzy_sum' not in stacked_df.columns:
        fuzzy_cols = [col for col in stacked_df.columns if col.startswith('fuzzy_') and not col.startswith('fuzzy_sum')]
        stacked_df['fuzzy_sum'] = stacked_df[fuzzy_cols].sum(axis=1)
    
    # Find locations with high suitability (top 10%)
    threshold = stacked_df['fuzzy_sum'].quantile(0.9)
    suitable_locations = stacked_df[stacked_df['fuzzy_sum'] >= threshold].copy()
    
    if suitable_locations.empty:
        return pd.DataFrame()
        
    return suitable_locations