import os
import json
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
import networkx as nx
from pysal.explore import esda
from pysal.lib import weights
import matplotlib
matplotlib.use('Agg')  # To prevent GUI windows from popping up

import dash
from dash import html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

# Constants
CSV_FOLDER_PATH = './app_data/'
COLORMAP = 'magma'
VIEW_STATE = {
    'longitude': 4.390,
    'latitude': 51.891,
    'zoom': 8
}

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # For deployment purposes

# Helper function to list and load specified CSV files
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
                df = df.set_index('hex9')
                # Reindex to include all hexagons, fill missing values with zero
                df = df.reindex(all_hexagons, fill_value=0)
                dataframes[file] = df
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
    return dataframes

# Load GeoDataFrame from shapefile
def load_gdf(gdf_path):
    """Loads a GeoDataFrame from a shapefile."""
    gdf = gpd.read_file(gdf_path)
    if 'hex9' not in gdf.columns:
        print("The shapefile must contain a 'hex9' column.")
        return None
    gdf = gdf.set_index('hex9')
    # Ensure the GeoDataFrame is in WGS84 CRS
    gdf = gdf.to_crs(epsg=4326)
    return gdf

def apply_color_mapping(df, value_column, colormap):
    """Applies a color map to a specified column of a DataFrame."""
    norm = plt.Normalize(vmin=df[value_column].min(), vmax=df[value_column].max())
    colormap_func = plt.get_cmap(colormap)
    df['color'] = df[value_column].apply(
        lambda x: [int(c * 255) for c in colormap_func(norm(x))[:3]]
    )  # Get RGB values

# Fuzzify input variables with "close" and "far", returning each fuzzified layer individually
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
            print(f"Zero range in data for layer {df.name}. Assigning constant fuzzified value.")
            fuzzified_array = np.ones_like(df_array)
        else:
            # Apply fuzzification depending on the fuzz_type
            if fuzz_type == "close":
                fuzzified_array = np.maximum(0, (df_array - df_array.min()) / range_diff)
            else:  # fuzz_type == "far"
                fuzzified_array = np.maximum(0, 1 - (df_array - df_array.min()) / range_diff)
        
        # Create a new DataFrame for the fuzzified result
        fuzzified_df = df.copy()
        fuzzified_df['fuzzy'] = np.round(fuzzified_array, 3)  # Add the fuzzified values
        
        # Apply the colormap
        apply_color_mapping(fuzzified_df, 'fuzzy', colormap_name)
        
        # Append fuzzified dataframe to the list
        fuzzified_dataframes.append(fuzzified_df.reset_index())
    
    return fuzzified_dataframes

# Stack the individual fuzzified layers into a single DataFrame
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

# Spatial suitability analysis on the stacked DataFrame
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

# Helper function to clean dataset names
def clean_dataset_name(name):
    """Replaces underscores with spaces and capitalizes for cleaner display."""
    return ' '.join(word.capitalize() for word in name.replace('_', ' ').split())

# Load data
idx = load_gdf('./app_data/h3_9_zh_delta.geojson')
if idx is None:
    raise Exception("Failed to load GeoDataFrame")
all_hexagons = idx.index.tolist()

selected_csvs = [
    'natuur_zh_delta',
    'water_zh_delta',
    'akkerboeren_zh_delta', 
    'Stad'
]
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
idx_json = json.loads(idx_reset.to_json())

# Layout
app.layout = html.Div([
    dbc.Container([
        html.H1("Geschiktheidsanalyse voor Gezamenlijke wasplaatsen in de Zuid-Hollandse Delta"),
        dcc.Markdown("""
Welkom bij de **Geschiktheidsanalyse voor Gezamenlijke wasplaatsen in de Zuid-Hollandse Delta**! Met deze tool kun je locaties vinden die het meest geschikt zijn voor nieuwe bouwprojecten, gebaseerd op verschillende criteria.

### Hoe werkt het?

1. **Kies je criteria**:  
   Gebruik de zijbalk om te selecteren op welke kenmerken je **dichtbij** of **ver weg** wilt zijn.

2. **Voer een analyse uit**:  
   Klik op **"Bouw Geschiktheidskaart"** om een kaart te maken met gebieden die aan je gekozen criteria voldoen.

3. **Bekijk en sla resultaten op**:  
   De resultaten worden op de kaart getoond. Ben je tevreden? Sla je selectie op en ga verder naar **Fase 2** voor meer analyses.

Veel succes met je analyse!
"""),
        html.Hr(),
        html.H2("Dataset Visualisaties"),
        html.Label("Selecteer een dataset:"),
        dcc.Dropdown(
            id='dataset-dropdown',
            options=[{'label': name, 'value': name} for name in clean_dataset_names],
            value=clean_dataset_names[0],  # Default to the first dataset
            clearable=False
        ),
        dcc.Graph(id='dataset-visualization-map'),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.H3("Selecteer Criteria voor Geschiktheidsanalyse"),
                html.Label("Dichtbij"),
                dcc.Checklist(
                    id='selected-close',
                    options=[{'label': name, 'value': name} for name in clean_dataset_names],
                    value=[],
                    labelStyle={'display': 'block'}
                ),
                html.Label("Ver weg van"),
                dcc.Checklist(
                    id='selected-far',
                    options=[{'label': name, 'value': name} for name in clean_dataset_names],
                    value=[],
                    labelStyle={'display': 'block'}
                ),
                html.Br(),
                html.Button("Bouw Geschiktheidskaart", id='submit-button', n_clicks=0, className='btn btn-primary'),
                html.Br(), html.Br(),
                html.Div(id='save-results-div'),
            ], md=3),
            dbc.Col([
                dcc.Loading(
                    id="loading-1",
                    type="default",
                    children=[
                        html.Div(id='analysis-message'),  # Added to display messages
                        dcc.Graph(id='suitability-map'),
                        html.Br(),
                        html.Label("Selecteer percentiel voor geschiktheid (fuzzy sum drempel):"),
                        dcc.Slider(
                            id='percentile-slider',
                            min=0,
                            max=100,
                            step=0.5,
                            value=90,
                            marks={i: f'{i}%' for i in range(0, 101, 10)}
                        ),
                        html.Br(),
                        html.Div(id='top-10-table'),
                    ]
                ),
            ], md=9),
        ])
    ], fluid=True),
    dcc.Store(id='analysis-results')
])

# Callbacks
@app.callback(
    Output('dataset-visualization-map', 'figure'),
    Input('dataset-dropdown', 'value')
)
def update_dataset_visualization(selected_dataset):
    if selected_dataset is None:
        return dash.no_update
    # Map the clean name back to the dataset name
    dataset_name = clean_names_map[selected_dataset]
    df = dataframes[dataset_name]
    merged_df = df.reset_index()
    # Create the figure
    fig = px.choropleth_mapbox(
        merged_df,
        geojson=idx_json,
        locations='hex9',
        color='value',
        color_continuous_scale=COLORMAP,
        mapbox_style="carto-positron",
        zoom=VIEW_STATE['zoom'],
        center={"lat": VIEW_STATE['latitude'], "lon": VIEW_STATE['longitude']},
        opacity=0.5,
        labels={'value': selected_dataset},
        featureidkey='properties.hex9'  # Adjusted to match 'properties.hex9'
    )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig

@app.callback(
    Output('analysis-results', 'data'),
    Output('analysis-message', 'children'),
    Input('submit-button', 'n_clicks'),
    State('selected-close', 'value'),
    State('selected-far', 'value')
)
def perform_analysis(n_clicks, selected_close, selected_far):
    if n_clicks is None or n_clicks == 0:
        return None, None
    if not selected_close and not selected_far:
        return None, html.Div("Selecteer alstublieft ten minste één criterium voor de analyse.", style={'color': 'red'})
    # Map clean names back to original names
    selected_variables_close = [clean_names_map[name] for name in selected_close]
    selected_variables_far = [clean_names_map[name] for name in selected_far]
    # Proceed with the analysis
    try:
        # Fuzzify the selected datasets
        fuzzified_dataframes_close = fuzzify_each_layer(
            [dataframes[var] for var in selected_variables_close], 'close', COLORMAP
        ) if selected_variables_close else []
        fuzzified_dataframes_far = fuzzify_each_layer(
            [dataframes[var] for var in selected_variables_far], 'far', COLORMAP
        ) if selected_variables_far else []
        # Stack the fuzzified layers
        stacked_df_close = stack_fuzzified_layers(fuzzified_dataframes_close, suffix='_close') if fuzzified_dataframes_close else None
        stacked_df_far = stack_fuzzified_layers(fuzzified_dataframes_far, suffix='_far') if fuzzified_dataframes_far else None
        # Combine the stacked DataFrames
        if stacked_df_close is not None and stacked_df_far is not None:
            stacked_df = pd.merge(
                stacked_df_close,
                stacked_df_far,
                on='hex9',
                how='outer'
            )
        elif stacked_df_close is not None:
            stacked_df = stacked_df_close
        else:
            stacked_df = stacked_df_far
        # Sum all 'fuzzy_' columns to get the final 'fuzzy_sum'
        fuzzy_cols = [col for col in stacked_df.columns if col.startswith('fuzzy_') and not col.startswith('fuzzy_sum')]
        stacked_df['fuzzy_sum'] = stacked_df[fuzzy_cols].sum(axis=1)
        # Perform spatial analysis
        all_loi = perform_spatial_analysis_on_stack(stacked_df, idx, w, g)
        if not all_loi.empty:
            # Store results with index preserved
            return {
                'all_loi': all_loi.to_json(orient='split'),
                'stacked_df': stacked_df.to_json(orient='split')
            }, None
        else:
            print("Geen significante locaties gevonden. Pas je criteria aan.")
            return None, html.Div("Geen significante locaties gevonden. Pas je criteria aan.", style={'color': 'red'})
    except Exception as e:
        print(f"Er is een fout opgetreden tijdens de analyse: {e}")
        return None, html.Div(f"Er is een fout opgetreden tijdens de analyse: {e}", style={'color': 'red'})

@app.callback(
    Output('suitability-map', 'figure'),
    Output('top-10-table', 'children'),
    Input('analysis-results', 'data'),
    Input('percentile-slider', 'value'),
)
def update_suitability_map(analysis_results, percentile):
    if analysis_results is None:
        # Return an empty figure and a message
        fig = px.choropleth_mapbox(
            mapbox_style="carto-positron",
            zoom=VIEW_STATE['zoom'],
            center={"lat": VIEW_STATE['latitude'], "lon": VIEW_STATE['longitude']},
        )
        return fig, html.Div("Geen resultaten om weer te geven.", style={'color': 'red'})
    # Load the data with index preserved
    all_loi = pd.read_json(analysis_results['all_loi'], orient='split')
    stacked_df = pd.read_json(analysis_results['stacked_df'], orient='split')
    if all_loi.empty:
        return dash.no_update, dash.no_update
    # Apply percentile threshold
    fuzzy_sum_threshold = all_loi['fuzzy_sum'].quantile(percentile / 100.0)
    most_relevant_locations = all_loi[all_loi['fuzzy_sum'] >= fuzzy_sum_threshold]
    # Merge with geometries
    idx_reset = idx.reset_index()
    merged_df = stacked_df.merge(idx_reset[['hex9', 'geometry']], on='hex9', how='left')
    merged_df = merged_df.dropna(subset=['geometry'])
    # Create the map
    fig = px.choropleth_mapbox(
        merged_df,
        geojson=idx_json,
        locations='hex9',
        color='fuzzy_sum',
        color_continuous_scale='Viridis',
        mapbox_style="carto-positron",
        zoom=VIEW_STATE['zoom'],
        center={"lat": VIEW_STATE['latitude'], "lon": VIEW_STATE['longitude']},
        opacity=0.5,
        labels={'fuzzy_sum': 'Geschiktheid'},
        featureidkey='properties.hex9'  # Adjusted to match 'properties.hex9'
    )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    # Overlay the most relevant locations
    if not most_relevant_locations.empty:
        relevant_idx = idx.loc[most_relevant_locations.index]
        centroids = relevant_idx.centroid
        centroids_df = pd.DataFrame({
            'hex9': centroids.index,
            'lat': centroids.y,
            'lon': centroids.x,
            'fuzzy_sum': most_relevant_locations['fuzzy_sum']
        })
        # Use Plotly Express to create a scatter mapbox and update markers
        scatter = px.scatter_mapbox(
            centroids_df,
            lat='lat',
            lon='lon',
            hover_name='hex9',
            hover_data=['fuzzy_sum'],
            color_discrete_sequence=['red'],
            zoom=VIEW_STATE['zoom'],
            center={"lat": VIEW_STATE['latitude'], "lon": VIEW_STATE['longitude']},
        )
        scatter.update_traces(marker=dict(size=8))
        fig.add_trace(scatter.data[0])
    else:
        print("No significant locations found after applying filters.")
        # Optionally, provide feedback to the user
        fig.add_annotation(
            text="Geen significante locaties gevonden na toepassing van filters.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            font=dict(size=16, color="red")
        )
    # Create the table
    if not most_relevant_locations.empty:
        top_10 = most_relevant_locations.reset_index().head(10)
        # Rename 'index' column to 'hex9'
        top_10.rename(columns={'index': 'hex9'}, inplace=True)
        fuzzy_columns = [col for col in top_10.columns if col.startswith('fuzzy_') and col != 'fuzzy_sum']
        columns_to_display = ['hex9'] + fuzzy_columns + ['fuzzy_sum']
        table = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in columns_to_display],
            data=top_10[columns_to_display].to_dict('records'),
            style_table={'overflowX': 'auto'},
            page_size=10,
        )
        return fig, table
    else:
        return fig, html.Div("Geen locaties voldoen aan de huidige filters.", style={'color': 'red'})

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
