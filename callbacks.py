# callbacks.py

from dash import html, dcc, Input, Output, State, dash_table
import dash

from app import app

import pandas as pd
import plotly.express as px

from utils import COLORMAP, VIEW_STATE
from data_loader import dataframes, clean_dataset_names, clean_names_map, idx, idx_json
from analysis import fuzzify_each_layer, stack_fuzzified_layers, perform_spatial_analysis_on_stack

from data_loader import w, g

# Callback to show/hide loading overlay
@app.callback(
    Output('map-loading-overlay', 'style'),
    Input('submit-button', 'n_clicks'),
    State('selected-close', 'value'),
    State('selected-far', 'value')
)
def toggle_loading_overlay(n_clicks, selected_close, selected_far):
    if n_clicks is None or n_clicks == 0:
        return {'display': 'none'}
    if not selected_close and not selected_far:
        return {'display': 'none'}
    
    return {
        'display': 'flex',
        'position': 'absolute',
        'top': 0,
        'left': 0,
        'width': '100%',
        'height': '100%',
        'backgroundColor': 'rgba(0, 0, 0, 0.7)',
        'zIndex': 1000,
        'justifyContent': 'center',
        'alignItems': 'center',
        'flexDirection': 'column'
    }

# Combined callback function for analysis
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
        return None, html.Div(
            "Selecteer alstublieft ten minste één criterium voor de analyse.",
            style={'color': 'red'}
        )

    # Map clean names back to original names
    selected_variables_close = [clean_names_map[name] for name in selected_close]
    selected_variables_far = [clean_names_map[name] for name in selected_far]

    try:
        # Fuzzify the selected datasets
        fuzzified_dataframes_close = fuzzify_each_layer(
            [dataframes[var] for var in selected_variables_close], 'close', COLORMAP
        ) if selected_variables_close else []
        fuzzified_dataframes_far = fuzzify_each_layer(
            [dataframes[var] for var in selected_variables_far], 'far', COLORMAP
        ) if selected_variables_far else []

        # Stack the fuzzified layers
        stacked_df_close = stack_fuzzified_layers(
            fuzzified_dataframes_close, suffix='_close'
        ) if fuzzified_dataframes_close else None
        stacked_df_far = stack_fuzzified_layers(
            fuzzified_dataframes_far, suffix='_far'
        ) if fuzzified_dataframes_far else None

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
        fuzzy_cols = [
            col for col in stacked_df.columns
            if col.startswith('fuzzy_') and not col.startswith('fuzzy_sum')
        ]
        stacked_df['fuzzy_sum'] = stacked_df[fuzzy_cols].sum(axis=1)

        # Perform spatial analysis
        all_loi = perform_spatial_analysis_on_stack(stacked_df, idx, w, g)
        if not all_loi.empty:
            return {
                'all_loi': all_loi.to_json(orient='split'),
                'stacked_df': stacked_df.to_json(orient='split')
            }, None
        else:
            return None, html.Div(
                "Geen significante locaties gevonden. Pas je criteria aan.",
                style={'color': 'red'}
            )

    except Exception as e:
        return None, html.Div(
            f"Er is een fout opgetreden tijdens de analyse: {e}",
            style={'color': 'red'}
        )

# Callback to update map and table
@app.callback(
    Output('main-map', 'figure'),
    Output('top-10-table', 'children'),
    Output('map-loading-overlay', 'style', allow_duplicate=True),
    Input('dataset-dropdown', 'value'),
    Input('analysis-results', 'data'),
    Input('percentile-slider', 'value'),
    prevent_initial_call=True
)
def update_main_map(selected_dataset, analysis_results, percentile):
    # Hide loading overlay once we start processing results
    loading_style = {
        'display': 'none',
        'position': 'absolute',
        'top': 0,
        'left': 0,
        'width': '100%',
        'height': '100%',
        'backgroundColor': 'rgba(0, 0, 0, 0.7)',
        'zIndex': 1000,
        'justifyContent': 'center',
        'alignItems': 'center',
        'flexDirection': 'column'
    }

    if analysis_results is None:
        # Display the selected dataset
        if selected_dataset is None:
            return dash.no_update, None, loading_style
        # Map the clean name back to the dataset name
        dataset_name = clean_names_map[selected_dataset]
        df = dataframes[dataset_name].reset_index()
        df['hex9'] = df['hex9'].astype(str)
        merged_df = df.copy()
        # Set 0 values as transparent by adding an 'opacity' column
        merged_df['opacity'] = merged_df['value'].apply(lambda x: 0 if x == 0 else 1)
        # Create the figure
        fig = px.choropleth_mapbox(
            merged_df,
            geojson=idx_json,
            locations='hex9',
            color='value',
            color_continuous_scale=COLORMAP,
            opacity=merged_df['opacity'],
            labels={'value': selected_dataset},
            featureidkey='properties.hex9',
            mapbox_style='carto-darkmatter',
            zoom=VIEW_STATE['zoom'],
            center={"lat": VIEW_STATE['latitude'], "lon": VIEW_STATE['longitude']}
        )
        # Update the colorbar styling
        fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                bgcolor='black',
                tickfont=dict(color='white'),
                titlefont=dict(color='white'),
            ),
            paper_bgcolor='black',
            plot_bgcolor='black',
        )
        return fig, None, loading_style

    # Load the data with index preserved
    all_loi = pd.read_json(analysis_results['all_loi'], orient='split')
    stacked_df = pd.read_json(analysis_results['stacked_df'], orient='split')
    
    if all_loi.empty:
        return dash.no_update, dash.no_update, loading_style
    
    # Apply percentile threshold
    fuzzy_sum_threshold = all_loi['fuzzy_sum'].quantile(percentile / 100.0)
    most_relevant_locations = all_loi[all_loi['fuzzy_sum'] >= fuzzy_sum_threshold]
    
    # Merge with geometries
    idx_reset = idx.reset_index()
    merged_df = stacked_df.merge(idx_reset[['hex9', 'geometry']], on='hex9', how='left')
    merged_df = merged_df.dropna(subset=['geometry'])
    
    # Set opacity for visualization
    merged_df['opacity'] = merged_df['fuzzy_sum'].apply(lambda x: 1 if x > 3 else 0)
    
    # Create the map
    fig = px.choropleth_mapbox(
        merged_df,
        geojson=idx_json,
        locations='hex9',
        color='fuzzy_sum',
        color_continuous_scale='Viridis',
        opacity=merged_df['opacity'],
        labels={'fuzzy_sum': 'Geschiktheid'},
        featureidkey='properties.hex9',
        mapbox_style='carto-darkmatter',
        zoom=VIEW_STATE['zoom'],
        center={"lat": VIEW_STATE['latitude'], "lon": VIEW_STATE['longitude']},
    )
    
    # Update the colorbar styling
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            bgcolor='black',
            tickfont=dict(color='white'),
            titlefont=dict(color='white'),
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
    )

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
        
        # Add scatter trace
        scatter = px.scatter_mapbox(
            centroids_df,
            lat='lat',
            lon='lon',
            hover_name='hex9',
            hover_data=['fuzzy_sum'],
            color_discrete_sequence=['red'],
        )
        scatter.update_traces(marker=dict(size=8))
        fig.add_trace(scatter.data[0])

        # Create the table
        top_10 = most_relevant_locations.reset_index().head(10)
        if 'index' in top_10.columns:
            top_10.rename(columns={'index': 'hex9'}, inplace=True)
        
        fuzzy_columns = [col for col in top_10.columns if col.startswith('fuzzy_') and col != 'fuzzy_sum']
        columns_to_display = ['hex9'] + fuzzy_columns + ['fuzzy_sum']
        
        table = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in columns_to_display],
            data=top_10[columns_to_display].to_dict('records'),
            style_table={'overflowX': 'auto'},
            page_size=10,
            style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
            style_cell={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'},
        )
        return fig, table, loading_style
    else:
        return fig, html.Div("Geen locaties voldoen aan de huidige filters.", style={'color': 'red'}), loading_style