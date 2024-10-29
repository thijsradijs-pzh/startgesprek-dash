# callbacks.py

from dash import html, dash_table, Input, Output, State, ctx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import LineString

from app import app
from utils import COLORMAP
from data_loader import dataframes, clean_names_map, idx, idx_json, view_state
from analysis import fuzzify_each_layer, stack_fuzzified_layers, analyze_locations

def get_farms_in_vicinity(hex_id, farm_data, distance_km):
    """Find farm hexagons near a selected suitable location within specified km."""
    connections = []
    farm_hexagons = []
    
    # Get farm locations where value > 0
    farm_locations = farm_data[farm_data['value'] > 0].copy()
    
    # Get geometries and centroids
    farm_geometries = idx.loc[farm_locations.index]
    farm_centroids = farm_geometries.geometry.centroid
    
    # Get geometry and centroid for selected location
    suitable_geometry = idx.loc[hex_id]
    suitable_centroid = suitable_geometry.geometry.centroid

    for farm_hex in farm_locations.index:
        farm_centroid = farm_centroids.loc[farm_hex]
        
        # Calculate distance in kilometers (assuming coordinates are in degrees)
        # Using an approximation where 1 degree ≈ 111 km
        dx = (farm_centroid.x - suitable_centroid.x) * 111
        dy = (farm_centroid.y - suitable_centroid.y) * 111
        distance = (dx**2 + dy**2)**0.5
        
        # If within range, create a connection
        if distance <= distance_km:
            connection = {
                'suitable_hex': hex_id,
                'farm_hex': farm_hex,
                'distance_km': round(distance, 2),
                'line': LineString([
                    (suitable_centroid.x, suitable_centroid.y),
                    (farm_centroid.x, farm_centroid.y)
                ])
            }
            connections.append(connection)
            farm_hexagons.append(farm_hex)
    
    return pd.DataFrame(connections) if connections else pd.DataFrame(), list(set(farm_hexagons))

@app.callback(
    Output('analysis-results', 'data'),
    Output('analysis-message', 'children'),
    Output('results-container', 'style'),
    Input('submit-button', 'n_clicks'),
    State('selected-close', 'value'),
    State('selected-far', 'value')
)
def perform_analysis(n_clicks, selected_close, selected_far):
    if n_clicks is None or n_clicks == 0:
        return None, None, {'display': 'none'}
    
    if not selected_close and not selected_far:
        return None, html.Div(
            "Selecteer alstublieft ten minste één criterium voor de analyse.",
            className="alert alert-warning"
        ), {'display': 'none'}

    try:
        # Map clean names back to original names
        selected_variables_close = [clean_names_map[name] for name in selected_close] if selected_close else []
        selected_variables_far = [clean_names_map[name] for name in selected_far] if selected_far else []

        # Fuzzify the selected datasets
        fuzzified_dataframes_close = fuzzify_each_layer(
            [dataframes[var] for var in selected_variables_close], 
            'close'
        ) if selected_variables_close else []
        
        fuzzified_dataframes_far = fuzzify_each_layer(
            [dataframes[var] for var in selected_variables_far], 
            'far'
        ) if selected_variables_far else []

        # Stack the fuzzified layers
        stacked_df_close = stack_fuzzified_layers(
            fuzzified_dataframes_close, 
            suffix='_close'
        ) if fuzzified_dataframes_close else None
        
        stacked_df_far = stack_fuzzified_layers(
            fuzzified_dataframes_far, 
            suffix='_far'
        ) if fuzzified_dataframes_far else None

        # Combine the stacked DataFrames
        if stacked_df_close is not None and stacked_df_far is not None:
            stacked_df = pd.merge(stacked_df_close, stacked_df_far, on='hex9', how='outer')
        elif stacked_df_close is not None:
            stacked_df = stacked_df_close
        else:
            stacked_df = stacked_df_far

        # Analyze locations
        suitable_locations = analyze_locations(stacked_df)
        
        if not suitable_locations.empty:
            suitable_locations = suitable_locations.reset_index()
            return {
                'suitable_locations': suitable_locations.to_json(orient='split'),
                'stacked_df': stacked_df.to_json(orient='split')
            }, None, {'display': 'block'}
        else:
            return None, html.Div(
                "Geen geschikte locaties gevonden. Pas je criteria aan.",
                className="alert alert-warning"
            ), {'display': 'none'}

    except Exception as e:
        return None, html.Div(
            f"Er is een fout opgetreden tijdens de analyse: {e}",
            className="alert alert-danger"
        ), {'display': 'none'}

@app.callback(
    Output('main-map', 'figure'),
    Output('top-10-table', 'children'),
    Output('farm-info', 'children'),
    Input('analysis-results', 'data'),
    Input('distance-slider', 'value'),
    Input('selected-row', 'data')
)
def update_visualization(analysis_results, distance_km, selected_location_number):
    if analysis_results is None:
        # Return empty map with light theme and dynamic view state
        fig = px.choropleth_mapbox(
            pd.DataFrame(),
            mapbox_style='carto-positron',
            zoom=view_state['zoom'],
            center={"lat": view_state['latitude'], "lon": view_state['longitude']}
        )
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        return fig, None, None

    # Load the analysis results
    stacked_df = pd.read_json(analysis_results['stacked_df'], orient='split')
    
    # Get only top 10 locations and add numbered index
    top_10 = stacked_df.nlargest(10, 'fuzzy_sum')[['hex9', 'fuzzy_sum']].copy()
    top_10['location_number'] = range(1, len(top_10) + 1)
    
    # Get centroids for label positioning
    geometries = idx.loc[top_10['hex9']].geometry
    centroids = geometries.apply(lambda x: x.centroid)
    top_10['lon'] = centroids.apply(lambda x: x.x)
    top_10['lat'] = centroids.apply(lambda x: x.y)
    
    # Create base map with top 10 suitable locations and dynamic view state
    fig = px.choropleth_mapbox(
        top_10,
        geojson=idx_json,
        locations='hex9',
        color='fuzzy_sum',
        color_continuous_scale='viridis',
        opacity=0.7,
        featureidkey='properties.hex9',
        mapbox_style='carto-positron',
        zoom=view_state['zoom'],
        center={"lat": view_state['latitude'], "lon": view_state['longitude']}
    )

    # Update hover template
    fig.update_traces(
        hovertemplate="<b>Locatie %{customdata[0]}</b><br>" +
                     "Geschiktheid: %{customdata[1]:.3f}<br>" +
                     "<extra></extra>",
        customdata=top_10[['location_number', 'fuzzy_sum']].values
    )

    # Initialize farm info
    farm_info = None

    # If a location is selected, show farms in vicinity
    if selected_location_number is not None and distance_km:
        # Get hex_id for selected location
        selected_hex = top_10.loc[top_10['location_number'] == selected_location_number, 'hex9'].iloc[0]
        
        connections_df, farm_hexagons = get_farms_in_vicinity(
            selected_hex,
            dataframes['Akkerboeren'],
            distance_km
        )

        # Add farm hexagons
        if farm_hexagons:
            farm_data = pd.DataFrame(index=farm_hexagons)
            farm_data['hex9'] = farm_data.index
            
            farm_layer = px.choropleth_mapbox(
                farm_data,
                geojson=idx_json,
                locations='hex9',
                featureidkey='properties.hex9',
                opacity=0.5,
                color_discrete_sequence=['blue']
            )
            
            # Update hover template for farm hexagons
            farm_layer.update_traces(
                hovertemplate="<b>Boerderij</b><br>" +
                             "Hex ID: %{location}<br>" +
                             "<extra></extra>"
            )
            
            fig.add_traces(farm_layer.data)

        # Add connection lines
        if not connections_df.empty:
            for _, connection in connections_df.iterrows():
                line = connection['line']
                fig.add_trace(go.Scattermapbox(
                    lon=[coord[0] for coord in line.coords],
                    lat=[coord[1] for coord in line.coords],
                    mode='lines',
                    line=dict(width=1, color='gray'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Create farm info summary
            farm_info = html.Div([
                html.H4(f"Boerderijen binnen {distance_km} km van Locatie {selected_location_number}", className="mt-4 mb-3"),
                html.P(f"Aantal boerderijen gevonden: {len(farm_hexagons)}"),
                html.P(f"Gemiddelde afstand: {connections_df['distance_km'].mean():.2f} km"),
                html.P(f"Kleinste afstand: {connections_df['distance_km'].min():.2f} km"),
                html.P(f"Grootste afstand: {connections_df['distance_km'].max():.2f} km")
            ])

    # Update layout
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    # Create interactive table with clickable rows
    table = html.Div([
        html.H4("Top 10 Meest Geschikte Locaties", className="mt-4 mb-3"),
        dash_table.DataTable(
            id='location-table',
            columns=[
                {"name": "Locatie", "id": "location_number"},
                {"name": "Geschiktheid", "id": "fuzzy_sum"},
                {"name": "Hexagon ID", "id": "hex9"}
            ],
            data=top_10[['location_number', 'fuzzy_sum', 'hex9']].round(3).to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={
                'backgroundColor': 'white',
                'color': 'black',
                'textAlign': 'left',
                'padding': '10px'
            },
            style_header={
                'backgroundColor': '#f8f9fa',
                'fontWeight': 'bold'
            },
            style_data_conditional=[{
                'if': {'row_index': i},
                'backgroundColor': 'rgba(255, 0, 0, 0.2)'
            } for i in range(len(top_10)) if (i + 1) == selected_location_number],
            row_selectable='single',
            selected_rows=[selected_location_number - 1] if selected_location_number else [],
            style_cell_conditional=[
                {'if': {'column_id': 'location_number'},
                 'width': '100px'},
                {'if': {'column_id': 'fuzzy_sum'},
                 'width': '150px'},
                {'if': {'column_id': 'hex9'},
                 'width': '200px'}
            ]
        )
    ])

    return fig, table, farm_info

@app.callback(
    Output('selected-row', 'data'),
    Input('location-table', 'selected_rows'),
    prevent_initial_call=True
)
def update_selection(selected_rows):
    if selected_rows is None or len(selected_rows) == 0:
        return None
    return selected_rows[0] + 1