import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
from datetime import date, timedelta
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────
FINTECH_TICKERS = ["PAYTM.NS", "POLICYBZR.NS"]
BANK_TICKERS    = ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"]
ALL_TICKERS     = FINTECH_TICKERS + BANK_TICKERS

# ── FIX 1: RBI events — add new ones here after each MPC meeting ──
# Format: "YYYY-MM-DD": rate_in_percent
RBI_EVENTS = {
    "2020-03-27": 4.40, "2020-05-22": 4.00,
    "2022-05-04": 4.40, "2022-06-08": 4.90,
    "2022-08-05": 5.40, "2022-09-30": 5.90,
    "2022-12-07": 6.25, "2023-02-08": 6.50,
    "2025-02-07": 6.25, "2025-04-09": 6.00,
    "2026-04-09": 5.25,
    # ← ADD NEW RBI EVENTS HERE as they happen
    # Example: "2026-06-06": 5.00,
    # Example: "2026-08-07": 4.75,
}

# ── FIX 2: Always fetch data up to TODAY automatically ─────────
TODAY = date.today().strftime("%Y-%m-%d")

def load_fresh_data():
    """Always downloads up to today's date — never stale."""
    print(f"Downloading fresh data up to {TODAY}...")
    prices = yf.download(
        ALL_TICKERS,
        start="2020-01-01",
        end=TODAY,           # ← always today, not hardcoded
        auto_adjust=True,
        progress=False
    )["Close"]
    prices = prices.ffill()
    returns = prices.pct_change().dropna(how="all")
    print(f"Loaded {len(returns)} trading days up to {TODAY}")
    return returns

returns = load_fresh_data()

# Latest RBI info (auto-picks the most recent event)
latest_rbi_date = max(RBI_EVENTS.keys())
latest_rbi_rate = RBI_EVENTS[latest_rbi_date]

rbi_events_series = pd.Series(RBI_EVENTS, dtype=float)
rbi_events_series.index = pd.to_datetime(rbi_events_series.index)
rbi_rate   = rbi_events_series.reindex(returns.index).ffill()
rbi_change = rbi_events_series.reindex(returns.index).fillna(0).diff().fillna(0)

# ── FIX 3: Auto-refresh data every 24 hours ───────────────────
import threading
import time

def refresh_data_daily():
    """Background thread: reloads stock data every 24 hours."""
    global returns, rbi_rate, rbi_change
    while True:
        time.sleep(86400)  # wait 24 hours
        try:
            print("Auto-refreshing stock data...")
            returns = load_fresh_data()
            rbi_rate   = rbi_events_series.reindex(returns.index).ffill()
            rbi_change = rbi_events_series.reindex(returns.index).fillna(0).diff().fillna(0)
            print("Data refreshed successfully!")
        except Exception as e:
            print(f"Refresh failed: {e}")

# Start background refresh thread
refresh_thread = threading.Thread(target=refresh_data_daily, daemon=True)
refresh_thread.start()

# ── FUNCTIONS (unchanged) ──────────────────────────────────────
def compute_ccf(rate_change, stock_return, max_lag=60):
    aligned = pd.concat([rate_change, stock_return], axis=1).dropna()
    x = aligned.iloc[:, 0].values
    y = aligned.iloc[:, 1].values
    corrs = []
    for k in range(max_lag + 1):
        if k == 0:
            corrs.append(np.corrcoef(
                (x - x.mean())/(x.std()+1e-9),
                (y - y.mean())/(y.std()+1e-9))[0,1])
        else:
            corrs.append(np.corrcoef(
                (x[:-k] - x[:-k].mean())/(x[:-k].std()+1e-9),
                (y[k:]  - y[k:].mean())/(y[k:].std()+1e-9))[0,1])
    return pd.Series(corrs, index=range(max_lag + 1))

def run_simulation(ticker, shock_bps, horizon, n_paths=500):
    if ticker not in returns.columns:
        return None
    r = returns[ticker].dropna()
    rolling_vol = r.rolling(21).std().iloc[-1]
    shock_effect = shock_bps * 0.0003
    np.random.seed(42)
    paths = np.random.normal(shock_effect, rolling_vol,
                              size=(n_paths, horizon))
    cum_paths = np.cumsum(paths, axis=1) * 100
    return {
        "mean":    cum_paths.mean(axis=0),
        "upper":   np.percentile(cum_paths, 90, axis=0),
        "lower":   np.percentile(cum_paths, 10, axis=0),
        "success": round((cum_paths[:, -1] > 0).mean() * 100, 1),
    }

def calc_success_rate(ticker, event_type, horizon):
    if ticker not in returns.columns:
        return 0, 0
    rbi_chg = rbi_change.copy()
    if event_type == "hold":
        ev_dates = rbi_chg[rbi_chg == 0].dropna().index
    elif event_type == "hike":
        ev_dates = rbi_chg[rbi_chg > 0].dropna().index
    else:
        ev_dates = rbi_chg[rbi_chg < 0].dropna().index
    r = returns[ticker]
    successes = []
    for ev in ev_dates:
        idx = r.index.searchsorted(ev)
        if idx + horizon >= len(r):
            continue
        cum = (1 + r.iloc[idx:idx+horizon]).prod() - 1
        successes.append(int(cum > 0))
    if not successes:
        return 0, 0
    return round(np.mean(successes) * 100, 1), len(successes)

# ── APP ───────────────────────────────────────────────────────
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.FLATLY],
                suppress_callback_exceptions=True)
server = app.server

# Badge text auto-updates from RBI_EVENTS dict
badge_text = f"Repo Rate: {latest_rbi_rate}% | Latest MPC: {latest_rbi_date} | Data up to: {TODAY}"

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H3("🏦 RBI Policy Impact — Fintech Dashboard",
                         className="text-primary mt-3"), width=7),
        dbc.Col(dbc.Badge(badge_text,
                           color="info", className="mt-4 p-2",
                           style={"fontSize": "11px"}), width=5),
    ], className="mb-2"),

    dbc.Tabs([
        dbc.Tab(label="📊 Lag Analysis", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Select Stock:", className="mt-3 fw-bold"),
                    dcc.Dropdown(id="lag-ticker",
                        options=[{"label": t, "value": t} for t in ALL_TICKERS],
                        value="PAYTM.NS", clearable=False),
                ], width=4),
                dbc.Col([
                    html.Label("Max Lag (days):", className="mt-3 fw-bold"),
                    dcc.Slider(id="lag-max", min=10, max=60, step=10, value=30,
                               marks={d: str(d) for d in [10,20,30,40,60]}),
                ], width=6),
            ]),
            dcc.Graph(id="ccf-plot", style={"height": "450px"}),
            html.Div(id="lag-summary", className="text-center text-muted mb-3"),
        ]),

        dbc.Tab(label="🎯 Success Probability", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Stock:", className="mt-3 fw-bold"),
                    dcc.Dropdown(id="prob-ticker",
                        options=[{"label": t, "value": t} for t in ALL_TICKERS],
                        value="PAYTM.NS", clearable=False),
                ], width=3),
                dbc.Col([
                    html.Label("Event Type:", className="mt-3 fw-bold"),
                    dcc.Dropdown(id="prob-event",
                        options=[{"label": e.title(), "value": e}
                                 for e in ["hold", "hike", "cut"]],
                        value="hold", clearable=False),
                ], width=3),
                dbc.Col([
                    html.Label("Horizon (days):", className="mt-3 fw-bold"),
                    dcc.Slider(id="prob-horizon", min=5, max=60, step=5, value=30,
                               marks={d: str(d) for d in [5,10,20,30,60]}),
                ], width=4),
            ]),
            dbc.Row([
                dbc.Col(html.Div(id="success-card"), width=5),
                dbc.Col(dcc.Graph(id="success-horizon-plot",
                                  style={"height": "380px"}), width=7),
            ], className="mt-3"),
        ]),

        dbc.Tab(label="🔮 Scenario Simulator", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Stock:", className="mt-3 fw-bold"),
                    dcc.Dropdown(id="sim-ticker",
                        options=[{"label": t, "value": t} for t in ALL_TICKERS],
                        value="PAYTM.NS", clearable=False),
                ], width=3),
                dbc.Col([
                    html.Label("Rate Shock:", className="mt-3 fw-bold"),
                    dcc.Dropdown(id="sim-shock",
                        options=[
                            {"label": "−50 bps (Aggressive Cut)", "value": -50},
                            {"label": "−25 bps (Cut)",            "value": -25},
                            {"label": "0 bps (Hold)",             "value":   0},
                            {"label": "+25 bps (Hike)",           "value":  25},
                            {"label": "+50 bps (Aggressive Hike)","value":  50},
                        ], value=0, clearable=False),
                ], width=3),
                dbc.Col([
                    html.Label("Horizon (days):", className="mt-3 fw-bold"),
                    dcc.Slider(id="sim-horizon", min=10, max=60, step=10, value=30,
                               marks={d: str(d) for d in [10,20,30,60]}),
                ], width=4),
                dbc.Col(
                    html.Button("▶ Run", id="sim-btn", n_clicks=0,
                                className="btn btn-primary w-100 mt-4"),
                    width=2),
            ]),
            dcc.Graph(id="sim-plot", style={"height": "420px"}),
            html.Div(id="sim-result", className="mt-2"),
        ]),
    ]),
], fluid=True, className="px-4")


@app.callback(
    Output("ccf-plot", "figure"),
    Output("lag-summary", "children"),
    Input("lag-ticker", "value"),
    Input("lag-max", "value"),
)
def update_ccf(ticker, max_lag):
    if ticker not in returns.columns:
        return go.Figure(), "No data."
    ccf = compute_ccf(rbi_change, returns[ticker], max_lag)
    ci  = 1.96 / np.sqrt(len(returns[ticker].dropna()))
    peak_lag = int(ccf.abs().idxmax())
    peak_val = round(ccf.abs().max(), 3)
    fig = go.Figure()
    fig.add_bar(x=ccf.index, y=ccf.values,
                marker_color=["crimson" if v < 0 else "steelblue"
                              for v in ccf.values])
    fig.add_hline(y=ci,  line_dash="dash", line_color="gray",
                  annotation_text="95% CI")
    fig.add_hline(y=-ci, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"Cross-Correlation: RBI Rate Change → {ticker} Returns",
        xaxis_title="Lag (Trading Days)",
        yaxis_title="Correlation",
        template="plotly_white")
    return fig, f"Peak lag: {peak_lag} days  |  Peak CCF: {peak_val}"


@app.callback(
    Output("success-card", "children"),
    Output("success-horizon-plot", "figure"),
    Input("prob-ticker", "value"),
    Input("prob-event", "value"),
    Input("prob-horizon", "value"),
)
def update_success(ticker, event_type, horizon):
    rate, n = calc_success_rate(ticker, event_type, horizon)
    color = "success" if rate >= 60 else ("warning" if rate >= 45 else "danger")
    card = dbc.Card([dbc.CardBody([
        html.H2(f"{rate}%", className=f"text-{color} display-4 fw-bold"),
        html.P(f"Success rate of BUYING {ticker}", className="text-muted"),
        html.P(f"{horizon} days after a rate {event_type.upper()}",
               className="fw-semibold"),
        html.Small(f"Based on {n} past RBI events", className="text-muted"),
    ])], className="text-center shadow-sm mt-3")
    horizons = [5, 10, 20, 30, 60]
    rates = [calc_success_rate(ticker, event_type, h)[0] for h in horizons]
    fig = go.Figure()
    fig.add_scatter(x=horizons, y=rates, mode="lines+markers",
                    line=dict(color="steelblue", width=2.5),
                    marker=dict(size=8))
    fig.add_hline(y=50, line_dash="dot", line_color="red",
                  annotation_text="50% baseline")
    fig.update_layout(
        title=f"Success Rate vs Horizon — {ticker} ({event_type})",
        xaxis_title="Horizon (days)", yaxis_title="Success Rate (%)",
        template="plotly_white", yaxis=dict(range=[0, 100]))
    return card, fig


@app.callback(
    Output("sim-plot", "figure"),
    Output("sim-result", "children"),
    Input("sim-btn", "n_clicks"),
    State("sim-ticker", "value"),
    State("sim-shock", "value"),
    State("sim-horizon", "value"),
    prevent_initial_call=True,
)
def update_simulation(n_clicks, ticker, shock, horizon):
    result = run_simulation(ticker, shock, horizon)
    if result is None:
        return go.Figure(), "No data."
    days = list(range(1, horizon + 1))
    fig = go.Figure()
    fig.add_scatter(x=days, y=result["upper"], mode="lines",
                    line=dict(width=0), showlegend=False)
    fig.add_scatter(x=days, y=result["lower"], mode="lines",
                    fill="tonexty", fillcolor="rgba(70,130,180,0.15)",
                    line=dict(width=0), name="80% CI")
    fig.add_scatter(x=days, y=result["mean"], mode="lines",
                    name="Mean Path", line=dict(color="steelblue", width=2.5))
    fig.add_hline(y=0, line_dash="dot", line_color="red")
    fig.update_layout(
        title=f"{ticker} — Simulated Return | {shock:+d} bps | {horizon}d",
        xaxis_title="Days After Announcement",
        yaxis_title="Cumulative Return (%)",
        template="plotly_white")
    prob = result["success"]
    color = "success" if prob >= 55 else ("warning" if prob >= 45 else "danger")
    badge = dbc.Alert(
        f"📊 Probability of positive return: {prob}% "
        f"over {horizon} days after {shock:+d} bps shock",
        color=color, className="text-center fw-bold")
    return fig, badge


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000, debug=False)
