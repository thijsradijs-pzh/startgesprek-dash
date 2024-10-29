# app.py

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

from data_loader import clean_dataset_names

# Initialize the app with BOOTSTRAP theme and suppress callback exceptions
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
app.title = "Geschiktheidsanalyse Gezamenlijke Wasplaatsen"
server = app.server

# Layout
app.layout = dbc.Container([
    # Step 1: Introduction
    dbc.Row(
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H3("Stap 1: Over deze tool")),
                dbc.CardBody([
                    html.P(
                        "Deze tool helpt bij het identificeren van geschikte locaties voor gezamenlijke wasplaatsen. "
                        "U kunt verschillende criteria selecteren om de geschiktheid van locaties te bepalen:"
                    ),
                    html.Ul([
                        html.Li("Selecteer criteria die dichtbij moeten zijn"),
                        html.Li("Selecteer criteria die juist ver weg moeten zijn"),
                        html.Li("De tool analyseert deze voorkeuren en toont de meest geschikte locaties op de kaart")
                    ])
                ])
            ], className="mb-4")
        )
    ),

    # Step 2: Criteria Selection
    dbc.Row(
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H3("Stap 2: Selecteer uw criteria")),
                dbc.CardBody([
                    dbc.Row([
                        # Close criteria
                        dbc.Col([
                            html.H4("Dichtbij", className="mb-3"),
                            dcc.Checklist(
                                id='selected-close',
                                options=[{'label': name, 'value': name} for name in clean_dataset_names],
                                value=[],
                                labelStyle={'display': 'block'},
                                inputStyle={"margin-right": "5px"},
                            ),
                        ], md=6),
                        # Far criteria
                        dbc.Col([
                            html.H4("Ver weg van", className="mb-3"),
                            dcc.Checklist(
                                id='selected-far',
                                options=[{'label': name, 'value': name} for name in clean_dataset_names],
                                value=[],
                                labelStyle={'display': 'block'},
                                inputStyle={"margin-right": "5px"},
                            ),
                        ], md=6),
                    ]),
                    html.Div(
                        dbc.Button(
                            "Bouw Geschiktheidskaart",
                            id='submit-button',
                            color="primary",
                            className="mt-4",
                            n_clicks=0
                        ),
                        className="text-center"
                    ),
                    html.Div(id='analysis-message', className="mt-3"),
                ])
            ], className="mb-4")
        )
    ),

    # Map and Results (initially hidden)
    dbc.Row(
        dbc.Col(
            html.Div([
                dbc.Card([
                    dbc.CardHeader(html.H3("Resultaten")),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-1",
                            type="default",
                            children=[
                                dcc.Graph(
                                    id='main-map',
                                    style={'height': '70vh'},
                                    config={'scrollZoom': True}
                                )
                            ]
                        ),
                        html.Div(id='top-10-table', className="mt-4"),
                        # Add distance slider here
                        html.Div([
                            html.Label("Selecteer maximale afstand (km):", className="mt-3"),
                            dcc.Slider(
                                id='distance-slider',
                                min=1,
                                max=25,
                                step=1,
                                value=5,
                                marks={i: f'{i} km' for i in range(0, 26, 5)},
                                className="mt-2 mb-4"
                            )
                        ], id='distance-control', className="mt-4"),
                        html.Div(id='farm-info', className="mt-4"),
                        dcc.Store(id='selected-row')
                    ])
                ])
            ], id='results-container', style={'display': 'none'})
        )
    ),

    # Store components
    dcc.Store(id='analysis-results'),
], fluid=True, className="py-4")