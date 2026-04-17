"""
Dashboard for LLM-Enhanced Financial ABM
Interactive interface for calibration, simulation, and analysis
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
import json
from datetime import datetime
import threading
import time

from core.model import MarketModel

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "LLM-Enhanced ABM Dashboard"

# Global state
model_state = {
    'model': None,
    'running': False,
    'progress': 0,
    'status': 'Ready',
    'results': None
}

# Styles
CARD_STYLE = {
    'backgroundColor': '#f8f9fa',
    'padding': '20px',
    'marginBottom': '20px',
    'borderRadius': '5px',
    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
}

BUTTON_STYLE = {
    'marginRight': '10px',
    'marginTop': '10px'
}

# Layout
app.layout = html.Div([
    html.Div([
        html.H1("LLM-Enhanced Financial ABM", style={'textAlign': 'center', 'color': '#2c3e50'}),
        html.P("Agent-Based Market Simulation with LLM-Programmed Behaviors",
               style={'textAlign': 'center', 'color': '#7f8c8d'})
    ], style={'padding': '20px', 'backgroundColor': '#ecf0f1'}),

    dcc.Tabs(id='tabs', value='setup', children=[
        dcc.Tab(label='Setup', value='setup'),
        dcc.Tab(label='Calibration', value='calibration'),
        dcc.Tab(label='Simulation', value='simulation'),
        dcc.Tab(label='Results', value='results'),
        dcc.Tab(label='Validation', value='validation'),
    ]),

    html.Div(id='tab-content', style={'padding': '20px'}),

    # Hidden divs for state management
    dcc.Store(id='model-store'),
    dcc.Interval(id='progress-interval', interval=1000, disabled=True),
], style={'fontFamily': 'Arial, sans-serif'})


# Tab content renderer
@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value')
)
def render_tab_content(tab):
    if tab == 'setup':
        return render_setup_tab()
    elif tab == 'calibration':
        return render_calibration_tab()
    elif tab == 'simulation':
        return render_simulation_tab()
    elif tab == 'results':
        return render_results_tab()
    elif tab == 'validation':
        return render_validation_tab()


def render_setup_tab():
    """Setup and configuration tab"""
    return html.Div([
        html.H2("Simulation Setup"),

        html.Div([
            html.H3("LLM Provider"),
            dcc.RadioItems(
                id='llm-provider',
                options=[
                    {'label': ' Mock (Fast, No API)', 'value': 'mock'},
                    {'label': ' Anthropic Claude', 'value': 'anthropic'},
                    {'label': ' OpenAI GPT', 'value': 'openai'},
                ],
                value='mock',
                labelStyle={'display': 'block', 'marginBottom': '10px'}
            ),
        ], style=CARD_STYLE),

        html.Div([
            html.H3("Simulation Parameters"),
            html.Label("Total Simulation Days:"),
            dcc.Input(id='total-days', type='number', value=504, min=1, max=2000,
                     style={'width': '100%', 'marginBottom': '10px'}),

            html.Label("Number of Agents:"),
            dcc.Input(id='num-agents', type='number', value=150, min=10, max=500,
                     style={'width': '100%', 'marginBottom': '10px'}),

            html.Label("Total Capital ($B):"),
            dcc.Input(id='total-capital', type='number', value=10, min=1, max=100,
                     style={'width': '100%', 'marginBottom': '10px'}),

            html.Label("Random Seed:"),
            dcc.Input(id='random-seed', type='number', value=42,
                     style={'width': '100%', 'marginBottom': '10px'}),
        ], style=CARD_STYLE),

        html.Div([
            html.H3("Recalibration Settings"),
            dcc.Checklist(
                id='enable-recalibration',
                options=[{'label': ' Enable periodic recalibration', 'value': 'enabled'}],
                value=['enabled']
            ),
            html.Br(),
            html.Label("Quarterly Reflection (days):"),
            dcc.Input(id='quarterly-days', type='number', value=90, min=30, max=180,
                     style={'width': '100%', 'marginBottom': '10px'}),
        ], style=CARD_STYLE),

        html.Button('Initialize Model', id='init-button', n_clicks=0,
                   style={**BUTTON_STYLE, 'backgroundColor': '#3498db', 'color': 'white',
                          'padding': '10px 20px', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),

        html.Div(id='init-status', style={'marginTop': '20px', 'fontWeight': 'bold'})
    ])


def render_calibration_tab():
    """Agent calibration tab"""
    return html.Div([
        html.H2("Agent Calibration"),

        html.Div([
            html.H3("Calibration Status"),
            html.Div(id='calibration-status', children='Model not initialized'),
            html.Br(),
            html.Button('Run Calibration', id='calibrate-button', n_clicks=0,
                       style={**BUTTON_STYLE, 'backgroundColor': '#9b59b6', 'color': 'white',
                              'padding': '10px 20px', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
        ], style=CARD_STYLE),

        html.Div([
            html.H3("Agent Distribution"),
            dcc.Graph(id='agent-distribution-chart'),
        ], style=CARD_STYLE),

        html.Div([
            html.H3("Sample Persona"),
            html.Pre(id='sample-persona', style={'backgroundColor': '#2c3e50', 'color': '#ecf0f1',
                                                  'padding': '15px', 'borderRadius': '5px',
                                                  'overflow': 'auto', 'maxHeight': '400px'})
        ], style=CARD_STYLE),
    ])


def render_simulation_tab():
    """Simulation control and monitoring tab"""
    return html.Div([
        html.H2("Run Simulation"),

        html.Div([
            html.H3("Simulation Control"),
            html.Label("Days to Run:"),
            dcc.Input(id='sim-days', type='number', value=100, min=1, max=2000,
                     style={'width': '200px', 'marginRight': '20px'}),

            html.Button('Run', id='run-button', n_clicks=0,
                       style={**BUTTON_STYLE, 'backgroundColor': '#27ae60', 'color': 'white',
                              'padding': '10px 20px', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),

            html.Button('Pause', id='pause-button', n_clicks=0, disabled=True,
                       style={**BUTTON_STYLE, 'backgroundColor': '#e67e22', 'color': 'white',
                              'padding': '10px 20px', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),

            html.Button('Stop', id='stop-button', n_clicks=0, disabled=True,
                       style={**BUTTON_STYLE, 'backgroundColor': '#e74c3c', 'color': 'white',
                              'padding': '10px 20px', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
        ], style=CARD_STYLE),

        html.Div([
            html.H3("Progress"),
            dcc.Graph(id='progress-chart'),
            html.Div(id='sim-status', style={'marginTop': '20px', 'fontSize': '18px', 'fontWeight': 'bold'}),
        ], style=CARD_STYLE),

        html.Div([
            html.H3("Live Market Prices"),
            dcc.Graph(id='live-prices-chart'),
        ], style=CARD_STYLE),
    ])


def render_results_tab():
    """Results visualization tab"""
    return html.Div([
        html.H2("Simulation Results"),

        html.Button('Refresh Results', id='refresh-results-button', n_clicks=0,
                   style={**BUTTON_STYLE, 'backgroundColor': '#3498db', 'color': 'white',
                          'padding': '10px 20px', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),

        html.Div([
            html.H3("Summary Metrics"),
            html.Div(id='summary-metrics'),
        ], style=CARD_STYLE),

        html.Div([
            html.H3("Price Evolution"),
            dcc.Graph(id='price-evolution-chart'),
        ], style=CARD_STYLE),

        html.Div([
            html.H3("Agent Performance Distribution"),
            dcc.Graph(id='agent-performance-chart'),
        ], style=CARD_STYLE),

        html.Div([
            html.H3("Market Events Timeline"),
            html.Div(id='events-timeline'),
        ], style=CARD_STYLE),
    ])


def render_validation_tab():
    """Validation and stylized facts tab"""
    return html.Div([
        html.H2("Model Validation"),

        html.Button('Calculate Stylized Facts', id='calculate-facts-button', n_clicks=0,
                   style={**BUTTON_STYLE, 'backgroundColor': '#16a085', 'color': 'white',
                          'padding': '10px 20px', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),

        html.Div([
            html.H3("Stylized Facts"),
            html.Div(id='stylized-facts-table'),
        ], style=CARD_STYLE),

        html.Div([
            html.H3("Return Distribution"),
            dcc.Graph(id='return-distribution-chart'),
        ], style=CARD_STYLE),

        html.Div([
            html.H3("Volatility Clustering"),
            dcc.Graph(id='volatility-clustering-chart'),
        ], style=CARD_STYLE),
    ])


# Callbacks

@app.callback(
    Output('init-status', 'children'),
    Input('init-button', 'n_clicks'),
    State('llm-provider', 'value'),
    State('total-days', 'value'),
    State('num-agents', 'value'),
    prevent_initial_call=True
)
def initialize_model(n_clicks, provider, total_days, num_agents):
    """Initialize the model"""
    if n_clicks == 0:
        return ""

    try:
        model_state['model'] = MarketModel()
        model_state['status'] = 'Initialized'
        return html.Div([
            html.P("[OK] Model initialized successfully!", style={'color': 'green'}),
            html.P(f"Provider: {provider.upper()}", style={'color': '#7f8c8d'}),
            html.P(f"Configured for {total_days} days with {num_agents} agents", style={'color': '#7f8c8d'})
        ])
    except Exception as e:
        return html.P(f"[ERROR] {str(e)}", style={'color': 'red'})


@app.callback(
    [Output('calibration-status', 'children'),
     Output('agent-distribution-chart', 'figure'),
     Output('sample-persona', 'children')],
    Input('calibrate-button', 'n_clicks'),
    State('llm-provider', 'value'),
    prevent_initial_call=True
)
def run_calibration(n_clicks, provider):
    """Run agent calibration"""
    if n_clicks == 0 or model_state['model'] is None:
        return "Model not initialized", {}, ""

    try:
        use_llm = (provider != 'mock')
        model = model_state['model']

        status_msg = f"Running calibration... (Mode: {provider.upper()})"

        # Run setup (calibration happens here)
        model.setup(calibrate_agents=use_llm)

        # Create distribution chart
        agent_types = {}
        for agent in model.agents:
            agent_types[agent.agent_type] = agent_types.get(agent.agent_type, 0) + 1

        fig = px.bar(x=list(agent_types.keys()), y=list(agent_types.values()),
                     labels={'x': 'Agent Type', 'y': 'Count'},
                     title='Agent Distribution by Type')
        fig.update_layout(showlegend=False)

        # Get sample persona
        if model.agents:
            sample_agent = model.agents[0]
            persona_str = json.dumps(sample_agent.persona, indent=2)
        else:
            persona_str = "No agents created"

        status_msg = f"[OK] Calibration complete! {len(model.agents)} agents created"

        return status_msg, fig, persona_str

    except Exception as e:
        return f"[ERROR] Error: {str(e)}", {}, ""


@app.callback(
    [Output('sim-status', 'children'),
     Output('progress-chart', 'figure'),
     Output('live-prices-chart', 'figure')],
    Input('run-button', 'n_clicks'),
    State('sim-days', 'value'),
    prevent_initial_call=True
)
def run_simulation(n_clicks, days):
    """Run simulation"""
    if n_clicks == 0 or model_state['model'] is None:
        return "Model not initialized", {}, {}

    try:
        model = model_state['model']

        # Run simulation
        model.run(steps=days)

        # Progress chart
        progress_fig = go.Figure()
        progress_fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=model.current_day,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Days Completed"},
            gauge={'axis': {'range': [None, model.total_days]}}
        ))

        # Live prices
        if model.asset_universe:
            prices_data = []
            for ticker in ['TECH', 'VALUE', 'SAFE']:
                asset = model.asset_universe.get_asset(ticker)
                if asset:
                    prices_data.append({
                        'Asset': ticker,
                        'Price': asset.price,
                        'Initial': asset.price_history[0] if asset.price_history else asset.price
                    })

            df = pd.DataFrame(prices_data)
            df['Change %'] = ((df['Price'] / df['Initial']) - 1) * 100

            prices_fig = px.bar(df, x='Asset', y='Change %',
                               title='Asset Price Changes',
                               color='Change %',
                               color_continuous_scale=['red', 'yellow', 'green'])
        else:
            prices_fig = {}

        status = f"[OK] Simulation complete! Ran {model.current_day} days"

        return status, progress_fig, prices_fig

    except Exception as e:
        return f"[ERROR] Error: {str(e)}", {}, {}


@app.callback(
    [Output('summary-metrics', 'children'),
     Output('price-evolution-chart', 'figure'),
     Output('agent-performance-chart', 'figure'),
     Output('events-timeline', 'children')],
    Input('refresh-results-button', 'n_clicks'),
    prevent_initial_call=True
)
def refresh_results(n_clicks):
    """Refresh results display"""
    if model_state['model'] is None:
        return "No results available", {}, {}, "No events"

    try:
        model = model_state['model']
        summary = model.get_summary()

        # Summary metrics
        metrics_div = html.Div([
            html.Div([
                html.H4(f"Day {summary['current_day']}", style={'color': '#3498db'}),
                html.P("Simulation Progress")
            ], style={'display': 'inline-block', 'marginRight': '40px'}),

            html.Div([
                html.H4(f"{summary['num_agents']}", style={'color': '#9b59b6'}),
                html.P("Active Agents")
            ], style={'display': 'inline-block', 'marginRight': '40px'}),

            html.Div([
                html.H4(f"{summary['num_events']}", style={'color': '#e67e22'}),
                html.P("Market Events")
            ], style={'display': 'inline-block', 'marginRight': '40px'}),

            html.Div([
                html.H4(f"{summary.get('agents', {}).get('avg_return', 0):.2f}%",
                       style={'color': '#27ae60'}),
                html.P("Avg Agent Return")
            ], style={'display': 'inline-block'}),
        ])

        # Price evolution chart
        price_data = []
        for ticker in ['TECH', 'VALUE', 'SAFE']:
            asset = model.asset_universe.get_asset(ticker)
            if asset and asset.price_history:
                for day, price in enumerate(asset.price_history):
                    price_data.append({'Day': day, 'Asset': ticker, 'Price': price})

        if price_data:
            df_prices = pd.DataFrame(price_data)
            price_fig = px.line(df_prices, x='Day', y='Price', color='Asset',
                               title='Asset Price Evolution')
        else:
            price_fig = {}

        # Agent performance
        if model.agents:
            perf_data = []
            for agent in model.agents[:50]:  # Limit to first 50 for performance
                perf = agent.get_performance_history()
                perf_data.append({
                    'Agent': agent.unique_id,
                    'Type': agent.agent_type,
                    'Return %': perf['total_return']
                })

            df_perf = pd.DataFrame(perf_data)
            perf_fig = px.histogram(df_perf, x='Return %', nbins=30,
                                   title='Agent Return Distribution')
        else:
            perf_fig = {}

        # Events timeline
        events_div = html.Div([
            html.P(f"Day {e.day}: {e.description}") for e in model.event_history[-10:]
        ])

        return metrics_div, price_fig, perf_fig, events_div

    except Exception as e:
        return f"Error: {str(e)}", {}, {}, ""


@app.callback(
    [Output('stylized-facts-table', 'children'),
     Output('return-distribution-chart', 'figure'),
     Output('volatility-clustering-chart', 'figure')],
    Input('calculate-facts-button', 'n_clicks'),
    prevent_initial_call=True
)
def calculate_stylized_facts(n_clicks):
    """Calculate and display stylized facts"""
    if model_state['model'] is None:
        return "No data available", {}, {}

    try:
        model = model_state['model']
        facts = model.data_collector.get_stylized_facts()

        # Create table
        facts_table = html.Table([
            html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
            html.Tbody([
                html.Tr([html.Td(k), html.Td(f"{v:.4f}")]) for k, v in list(facts.items())[:20]
            ])
        ], style={'width': '100%', 'borderCollapse': 'collapse'})

        # Return distribution
        return_data = []
        for ticker in ['TECH', 'VALUE', 'SAFE']:
            asset = model.asset_universe.get_asset(ticker)
            if asset and len(asset.price_history) > 1:
                returns = np.diff(asset.price_history) / asset.price_history[:-1]
                for ret in returns:
                    return_data.append({'Asset': ticker, 'Return': ret * 100})

        if return_data:
            df_returns = pd.DataFrame(return_data)
            return_fig = px.histogram(df_returns, x='Return', color='Asset',
                                     title='Return Distribution', nbins=50,
                                     marginal='box')
        else:
            return_fig = {}

        # Volatility clustering
        vol_fig = {}

        return facts_table, return_fig, vol_fig

    except Exception as e:
        return f"Error: {str(e)}", {}, {}


if __name__ == '__main__':
    print("\n" + "="*70)
    print("LLM-Enhanced Financial ABM Dashboard")
    print("="*70)
    print("\nStarting dashboard server...")
    print("Open your browser and navigate to: http://127.0.0.1:8050")
    print("\nPress Ctrl+C to stop the server\n")

    app.run_server(debug=False, host='127.0.0.1', port=8050)
