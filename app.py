import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import warnings

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────
ALL_TICKERS = ["PAYTM.NS", "POLICYBZR.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"]
RBI_EVENTS = {
    "2020-03-27": 4.40, "2020-05-22": 4.00, "2022-05-04": 4.40, 
    "2022-06-08": 4.90, "2022-08-05": 5.40, "2022-09-30": 5.90,
    "2022-12-07": 6.25, "2023-02-08": 6.50, "2025-02-07": 6.25, 
    "2025-04-09": 6.00, "2026-04-09": 5.25,
}

# ── DATA LOADING ──────────────────────────────────────────────
session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0'})

data = yf.download(ALL_TICKERS, start="2020-01-01", auto_adjust=True, session=session)
prices = data["Close"].ffill()
returns = prices.pct_change().dropna(how="all")

# Align RBI shocks to market trading days
rbi_s = pd.Series(RBI_EVENTS)
rbi_s.index = pd.to_datetime(rbi_s.index)
rbi_rate = rbi_s.reindex(returns.index.union(rbi_s.index)).sort_index().ffill()
rbi_rate = rbi_rate.reindex(returns.index) 
rbi_change = rbi_rate.diff().fillna(0)

# ── APP SETUP ─────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server  # CRITICAL: Render needs this for Gunicorn

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H3("🏦 RBI Policy Dashboard", className="text-primary mt-3"), width=8),
        dbc.Col(dbc.Badge("Repo Rate: 5.25%", color="info", className="mt-4 p-2"), width=4),
    ], className="mb-2"),
    
    dbc.Tabs([
        dbc.Tab(label="📊 Lag Analysis", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Stock:", className="mt-3"),
                    dcc.Dropdown(id="lag-ticker", options=[{"label": t, "value": t} for t in ALL_TICKERS], value="PAYTM.NS"),
                ], width=4),
                dbc.Col([
                    html.Label("Max Lag (days):", className="mt-3"),
                    dcc.Slider(id="lag-max", min=10, max=60, step=10, value=30),
                ], width=6),
            ]),
            dcc.Graph(id="ccf-plot"),
            html.Div(id="lag-summary", className="text-center text-muted")
        ]),
        
        dbc.Tab(label="🎯 Success Probability", children=[
            dbc.Row([
                dbc.Col(html.Div(id="success-card"), width=5),
                dbc.Col(dcc.Graph(id="success-horizon-plot"), width=7),
            ], className="mt-3"),
        ]),

        dbc.Tab(label="👨‍💻 Source Code", children=[
            html.H5("Core Correlation Logic", className="mt-4"),
            html.Pre("""def compute_ccf(rate_change, stock_return, max_lag):
    return [rate_change.corr(stock_return.shift(-k)) for k in range(max_lag + 1)]""", 
            style={"backgroundColor": "#f8f9fa", "padding": "15px"})
        ])
    ])
], fluid=True)

# ── CALLBACKS ─────────────────────────────────────────────────
@app.callback(
    Output("ccf-plot", "figure"),
    Output("lag-summary", "children"),
    Input("lag-ticker", "value"),
    Input("lag-max", "value"),
)
def update_ccf(ticker, max_lag):
    y = returns[ticker]
    corrs = [rbi_change.corr(y.shift(-k)) for k in range(max_lag + 1)]
    fig = go.Figure(go.Bar(x=list(range(max_lag+1)), y=corrs))
    fig.update_layout(title=f"Impact of Rate Changes on {ticker}", template="plotly_white")
    return fig, f"Analysis complete for {ticker}"

if __name__ == '__main__':
    app.run_server(debug=False)
