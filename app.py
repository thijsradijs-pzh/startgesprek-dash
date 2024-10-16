# app.py

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

from data_loader import dataframes, clean_dataset_names

# Initialize the app with DARKLY theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Geschiktheidsanalyse Gezamenlijke Wasplaatsen"
server = app.server  # For deployment purposes

# Layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                # Column for user controls
                dbc.Col(
                    [
                        html.H2("Geschiktheidsanalyse Gezamenlijke Wasplaatsen"),
                        html.P(
                            "Selecteer je criteria en bekijk de resultaten op de kaart."
                        ),
                        html.Hr(),
                        html.H3("Dataset Visualisaties"),
                        html.Label("Selecteer een dataset:"),
                        dcc.Dropdown(
                            id='dataset-dropdown',
                            options=[{'label': name, 'value': name} for name in clean_dataset_names],
                            value=clean_dataset_names[0],  # Default to the first dataset
                            clearable=False,
                            style={'color': '#000000'}  # Ensure text inside dropdown is visible
                        ),
                        html.Hr(),
                        html.H3("Selecteer Criteria voor Geschiktheidsanalyse"),
                        html.Label("Dichtbij"),
                        dcc.Checklist(
                            id='selected-close',
                            options=[{'label': name, 'value': name} for name in clean_dataset_names],
                            value=[],
                            labelStyle={'display': 'block'},
                            inputStyle={"margin-right": "5px"},
                            style={'color': '#ffffff'}
                        ),
                        html.Label("Ver weg van"),
                        dcc.Checklist(
                            id='selected-far',
                            options=[{'label': name, 'value': name} for name in clean_dataset_names],
                            value=[],
                            labelStyle={'display': 'block'},
                            inputStyle={"margin-right": "5px"},
                            style={'color': '#ffffff'}
                        ),
                        html.Br(),
                        html.Button("Bouw Geschiktheidskaart", id='submit-button', n_clicks=0, className='btn btn-primary'),
                        html.Br(), html.Br(),
                        html.Div(id='analysis-message'),  # Added to display messages
                        html.P("Selecteer percentiel voor geschiktheid (fuzzy sum drempel):"),
                        dcc.Slider(
                            id='percentile-slider',
                            min=0,
                            max=100,
                            step=0.5,
                            value=90,
                            marks={i: f'{i}%' for i in range(0, 101, 10)},
                            tooltip={'always_visible': False, 'placement': 'bottom'},
                        ),
                        html.Br(),
                        html.Div(id='top-10-table'),
                    ],
                    md=3,  # Adjusted column width
                    className="bg-dark text-white p-4",
                ),
                # Column for app graphs and plots
                dbc.Col(
                    [
                        dcc.Loading(
                            id="loading-1",
                            type="default",
                            children=[
                                dcc.Graph(id='main-map', style={'height': '100vh'}, config={'scrollZoom': True})
                            ]
                        ),
                    ],
                    md=9,  # Adjusted column width
                    className="p-0",  # Remove padding to maximize map size
                ),
            ],
            className="g-0",  # Remove gutters between columns
        ),
        # Hidden div inside the app that stores the intermediate value
        dcc.Store(id='analysis-results'),
    ],
    fluid=True,
    className="dbc"
)
