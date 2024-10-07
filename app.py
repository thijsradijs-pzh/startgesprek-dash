import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import pydeck as pdk
import os
import dash_deck

# Constants for Geospatial Visualization
CSV_FOLDER_PATH = './app_data/'
COLORMAP = 'magma'
VIEW_STATE = pdk.ViewState(longitude=4.390, latitude=51.891, zoom=8)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Multi Criteria Analyse Tool"  # Set page title

# Helper function to load CSV files
def list_and_load_csvs(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dataframes = {}
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df_name = os.path.splitext(file)[0]
        df = pd.read_csv(file_path)
        if not df.empty:
            dataframes[df_name] = df
    return dataframes

# Load the CSVs for the page
dataframes = list_and_load_csvs(CSV_FOLDER_PATH)

# Home page layout
home_layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("🌱 Multi Criteria Analyse, Een Interactieve Tool"), width=12),
    ]),
    dbc.Row([
        dbc.Col(html.Div([
            dcc.Markdown("""
            💡 **Over de tool...**
            Deze tool is bedoeld als startgesprek voor verschillende stakeholders, 
            niet als leidend in besluitvorming. Het is een **tweestaps** leerproces om gebruikers te betrekken bij het 
            leren over de voordelen en afwegingen van een bepaald scenario.
            """),
            dcc.Markdown("""
            🧭 **Hoe de tool te gebruiken...**
            - **Fase 1: Geschiktheidsanalyse**: Voer een analyse uit op meerdere criteria.
            - **Fase 2: Beleidsverkenner**: Verken combinaties van kandidaat-locaties voor een scenario.
            """),
            html.A("Ga naar Geschiktheidsanalyse", href="/fase-1", className="btn btn-primary")
        ]), width=12)
    ]),
])

# Phase 1 layout
phase_1_layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Geschiktheidsanalyse"), width=12),
    ]),
    dbc.Row([
        dbc.Col([
            html.P("Selecteer een dataset:"),
            dcc.Dropdown(
                id='dataset-dropdown',
                options=[{'label': name, 'value': name} for name in dataframes.keys()],
                value=list(dataframes.keys())[0]
            ),
            dcc.Graph(id='data-graph'),
            html.Div(id='deck-map')
        ], width=8)
    ])
])

# Layout switcher based on URL
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),  # This will track the URL
    html.Div(id='page-content')  # This div will render the current page layout
])

# Callback to dynamically update the page layout based on the URL
@app.callback(
    dash.dependencies.Output('page-content', 'children'),
    [dash.dependencies.Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/fase-1':
        return phase_1_layout
    else:
        return home_layout  # Default to home page

# Callback to update the graph based on the selected dataset
@app.callback(
    dash.dependencies.Output('data-graph', 'figure'),
    [dash.dependencies.Input('dataset-dropdown', 'value')]
)
def update_graph(selected_dataset):
    df = dataframes[selected_dataset]
    fig = {
        'data': [{'x': df['some_column'], 'y': df['another_column'], 'type': 'scatter', 'mode': 'markers'}],
        'layout': {'title': 'Dataset Visualization'}
    }
    return fig

# Callback to update the map visualization using dash-deck
@app.callback(
    dash.dependencies.Output('deck-map', 'children'),
    [dash.dependencies.Input('dataset-dropdown', 'value')]
)
def update_map(selected_dataset):
    df = dataframes[selected_dataset]
    
    # Create a layer using pydeck
    layer = pdk.Layer(
        "HexagonLayer",
        data=df,
        get_position=["longitude", "latitude"],
        radius=200,
        elevation_scale=4,
        elevation_range=[0, 1000],
        pickable=True,
        extruded=True,
    )

    view_state = VIEW_STATE
    
    deck = dash_deck.DeckGL(
        map_style='mapbox://styles/mapbox/light-v9',
        layers=[layer],
        initial_view_state=view_state
    )
    
    return deck

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
