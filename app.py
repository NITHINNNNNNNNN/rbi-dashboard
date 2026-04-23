"""
RBI Policy Impact — Fintech Dashboard
Fully Automated | Zero Manual Updates | Modern Dark UI
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import threading
import time
from datetime import date, datetime, timedelta

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════

FINTECH_TICKERS = ["PAYTM.NS", "POLICYBZR.NS"]
BANK_TICKERS    = ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"]
ALL_TICKERS     = FINTECH_TICKERS + BANK_TICKERS
MACRO_TICKERS   = ["CL=F", "USDINR=X", "^NSEI", "^NSEBANK"]

# ── RBI Events ────────────────────────────────────────────────
# Only add a new line here when RBI meets (every ~2 months).
# Everything else — stock data, volatility, macro, regime — is 100% automatic.
RBI_EVENTS = {
    "2020-03-27": 4.40, "2020-05-22": 4.00,
    "2022-05-04": 4.40, "2022-06-08": 4.90,
    "2022-08-05": 5.40, "2022-09-30": 5.90,
    "2022-12-07": 6.25, "2023-02-08": 6.50,
    "2025-02-07": 6.25, "2025-04-09": 6.00,
    "2026-04-09": 5.25,
    # ADD NEW EVENT BELOW AFTER EACH MPC MEETING:
    # "2026-06-06": 5.00,
}

# ══════════════════════════════════════════════════════════════
# SECTION 2 — DATA STORE (auto-refreshes every 24 hours)
# ══════════════════════════════════════════════════════════════

class DataStore:
    """Central data store. Refreshes automatically every 24 hours."""

    def __init__(self):
        self.returns     = None
        self.rbi_change  = None
        self.rbi_rate    = None
        self.macro       = {}
        self.last_updated = "Loading..."
        self.lock        = threading.Lock()

    def refresh(self):
        today = date.today().strftime("%Y-%m-%d")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Refreshing all data up to {today}...")
        try:
            # ── Stock prices ──────────────────────────────────
            raw = yf.download(
                ALL_TICKERS, start="2020-01-01", end=today,
                auto_adjust=True, progress=False
            )["Close"].ffill()
            new_returns = raw.pct_change().dropna(how="all")

            # ── RBI series aligned to trading calendar ────────
            rbi_s = pd.Series(RBI_EVENTS, dtype=float)
            rbi_s.index = pd.to_datetime(rbi_s.index)
            new_rbi_rate   = rbi_s.reindex(new_returns.index).ffill()
            new_rbi_change = rbi_s.reindex(new_returns.index).fillna(0).diff().fillna(0)

            # ── Macro data ────────────────────────────────────
            macro_raw = yf.download(
                MACRO_TICKERS, start="2020-01-01", end=today,
                auto_adjust=True, progress=False
            )["Close"].ffill()

            new_macro = {}
            for key, col in [("crude","CL=F"),("usdinr","USDINR=X"),
                              ("nifty","^NSEI"),("banknifty","^NSEBANK")]:
                if col in macro_raw.columns:
                    new_macro[key] = macro_raw[col].dropna()

            with self.lock:
                self.returns      = new_returns
                self.rbi_rate     = new_rbi_rate
                self.rbi_change   = new_rbi_change
                self.macro        = new_macro
                self.last_updated = datetime.now().strftime("%d %b %Y  %I:%M %p IST")

            print(f"  Done — {len(new_returns)} rows, {len(new_returns.columns)} stocks")

        except Exception as e:
            print(f"  Refresh error: {e}")


DS = DataStore()

def _background_loop():
    DS.refresh()               # initial load
    while True:
        time.sleep(86400)      # 24 hours
        DS.refresh()

threading.Thread(target=_background_loop, daemon=True).start()

# Give initial load a head start before first request
time.sleep(3)


# ══════════════════════════════════════════════════════════════
# SECTION 3 — ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════

def compute_ccf(rate_change: pd.Series, ret: pd.Series, max_lag: int = 60) -> pd.Series:
    """Cross-Correlation Function between rate changes and stock returns."""
    df = pd.concat([rate_change, ret], axis=1).dropna()
    if len(df) < max_lag + 20:
        return pd.Series(dtype=float)
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values
    xs = (x - x.mean()) / (x.std() + 1e-9)
    ys = (y - y.mean()) / (y.std() + 1e-9)
    corrs = []
    for k in range(max_lag + 1):
        if k == 0:
            corrs.append(float(np.corrcoef(xs, ys)[0, 1]))
        else:
            corrs.append(float(np.corrcoef(xs[:-k], ys[k:])[0, 1]))
    return pd.Series(corrs, index=range(max_lag + 1))


def ewma_vol(ret: pd.Series, span: int = 21) -> float:
    """EWMA daily volatility — more responsive than rolling std."""
    return float(ret.ewm(span=span).std().dropna().iloc[-1])


def calibrate_shock(ticker: str, shock_bps: float) -> float:
    """
    Estimate daily drift from historical post-event returns.
    Uses actual stock behaviour after past rate hikes/cuts/holds.
    Falls back to a conservative default if insufficient history.
    """
    if DS.returns is None or ticker not in DS.returns.columns:
        return shock_bps * 0.00015

    r = DS.returns[ticker].dropna()
    events = DS.rbi_change[DS.rbi_change != 0].dropna()

    same_dir = []
    for ev_date, chg in events.items():
        if (shock_bps > 0 and chg <= 0) or (shock_bps < 0 and chg >= 0):
            continue
        idx = r.index.searchsorted(ev_date)
        if idx + 15 >= len(r):
            continue
        post_avg = float(r.iloc[idx: idx + 15].mean())
        bps_size = abs(chg) * 100
        same_dir.append((bps_size, post_avg))

    if not same_dir:
        # Hold scenario or no matching events
        return 0.0

    # Weighted average: scale effect proportionally to shock size
    avg_bps  = np.mean([b for b, _ in same_dir])
    avg_ret  = np.mean([r for _, r in same_dir])
    if avg_bps == 0:
        return 0.0
    return float(avg_ret * (abs(shock_bps) / avg_bps))


def run_simulation(ticker: str, shock_bps: int, horizon: int, n_paths: int = 2000) -> dict | None:
    """
    Monte Carlo simulation with:
    — EWMA volatility (regime-adjusted)
    — Calibrated drift from historical events
    — Proper compounding (not cumsum)
    — 2,000 paths
    """
    if DS.returns is None or ticker not in DS.returns.columns:
        return None

    r = DS.returns[ticker].dropna()
    if len(r) < 40:
        return None

    daily_vol   = ewma_vol(r, span=21)
    daily_drift = calibrate_shock(ticker, shock_bps)

    # Regime factor: how stressed is the market right now vs. history?
    recent_vol  = float(r.tail(21).std())
    hist_vol    = float(r.std())
    regime      = max(0.5, min(recent_vol / (hist_vol + 1e-9), 3.0))
    adj_vol     = daily_vol * regime

    np.random.seed(42)
    raw = np.random.normal(daily_drift, adj_vol, size=(n_paths, horizon))
    cum = (np.cumprod(1 + raw, axis=1) - 1) * 100   # proper compounding in %

    final = cum[:, -1]
    return {
        "mean":     cum.mean(axis=0),
        "p90":      np.percentile(cum, 90, axis=0),
        "p75":      np.percentile(cum, 75, axis=0),
        "p25":      np.percentile(cum, 25, axis=0),
        "p10":      np.percentile(cum, 10, axis=0),
        "success":  round(float((final > 0).mean() * 100), 1),
        "exp_ret":  round(float(final.mean()), 2),
        "var95":    round(float(np.percentile(final, 5)), 2),
        "regime":   round(regime, 2),
        "ann_vol":  round(daily_vol * np.sqrt(252) * 100, 1),
        "final":    final,
    }


def calc_success_rate(ticker: str, event_type: str, horizon: int):
    """Historical success rate of buying after an RBI event."""
    if DS.returns is None or ticker not in DS.returns.columns:
        return 0, 0, [], []

    chg = DS.rbi_change
    if   event_type == "hold": ev = chg[chg == 0].dropna().index
    elif event_type == "hike": ev = chg[chg  > 0].dropna().index
    else:                       ev = chg[chg  < 0].dropna().index

    r = DS.returns[ticker]
    wins, rets, dates = [], [], []
    for d in ev:
        i = r.index.searchsorted(d)
        if i + horizon >= len(r):
            continue
        cum = float((1 + r.iloc[i: i + horizon]).prod() - 1) * 100
        wins.append(int(cum > 0))
        rets.append(round(cum, 2))
        dates.append(str(d.date()))

    if not wins:
        return 0, 0, [], []
    return round(float(np.mean(wins)) * 100, 1), len(wins), rets, dates


def live_macro() -> dict:
    """Latest values + 1-day change for macro indicators."""
    out = {}
    for key, series in DS.macro.items():
        if series is not None and len(series) >= 2:
            s = series.dropna()
            if len(s) >= 2:
                out[key] = {
                    "val":  round(float(s.iloc[-1]), 2),
                    "chg":  round(float(s.iloc[-1] - s.iloc[-2]), 2),
                    "pct":  round(float((s.iloc[-1] / s.iloc[-2] - 1) * 100), 2),
                }
    return out


def market_regime() -> tuple[str, str]:
    """Detect market stress from Nifty50 recent vs. historical vol."""
    try:
        nifty = DS.macro.get("nifty", None)
        if nifty is not None and len(nifty) > 60:
            nr = nifty.pct_change().dropna()
            rv = float(nr.tail(21).std() * np.sqrt(252) * 100)
            hv = float(nr.std()          * np.sqrt(252) * 100)
            ratio = rv / (hv + 1e-9)
            if   ratio > 1.3: return f"Stressed  {rv:.1f}% vol",  "danger"
            elif ratio > 1.1: return f"Elevated  {rv:.1f}% vol",  "warning"
            else:              return f"Calm  {rv:.1f}% vol",      "success"
    except Exception:
        pass
    return "Normal", "info"


def rbi_stale_days() -> int:
    latest = max(RBI_EVENTS.keys())
    return (datetime.now() - datetime.strptime(latest, "%Y-%m-%d")).days


# ══════════════════════════════════════════════════════════════
# SECTION 4 — APP & LAYOUT
# ══════════════════════════════════════════════════════════════

DARK_BG   = "#0d1117"
CARD_BG   = "#161b22"
BORDER    = "#30363d"
ACCENT    = "#58a6ff"
GREEN     = "#3fb950"
RED       = "#f85149"
YELLOW    = "#d29922"
TEXT      = "#e6edf3"
MUTED     = "#8b949e"

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;600&display=swap"
    ],
    suppress_callback_exceptions=True,
    title="RBI Fintech Dashboard"
)
server = app.server   # Render needs this

latest_rate = RBI_EVENTS[max(RBI_EVENTS.keys())]
latest_date = max(RBI_EVENTS.keys())

app.layout = dbc.Container([

    # ── Auto-refresh UI every 5 minutes ──────────────────────
    dcc.Interval(id="ticker", interval=300_000, n_intervals=0),

    # ── HEADER ───────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.H4("🏦  RBI POLICY IMPACT", className="mb-0 mt-3",
                    style={"fontFamily": "JetBrains Mono", "color": ACCENT,
                           "letterSpacing": "2px", "fontSize": "18px"}),
            html.P("Fintech Trading Decision Dashboard  ·  Fully Automated",
                   className="mb-0",
                   style={"color": MUTED, "fontSize": "12px", "fontFamily": "Inter"}),
        ], width=6),
        dbc.Col([
            html.Div([
                dbc.Badge(f"Repo  {latest_rate}%",  color="primary",
                          className="me-2 py-2 px-3",
                          style={"fontFamily": "JetBrains Mono", "fontSize": "12px"}),
                dbc.Badge(f"MPC  {latest_date}",    color="secondary",
                          className="me-2 py-2 px-3",
                          style={"fontFamily": "JetBrains Mono", "fontSize": "12px"}),
                dbc.Badge(id="regime-badge", color="success",
                          className="py-2 px-3",
                          style={"fontFamily": "JetBrains Mono", "fontSize": "12px"}),
            ], className="mt-3 text-end"),
            html.Div(id="last-updated",
                     className="text-end mt-1",
                     style={"color": MUTED, "fontSize": "11px", "fontFamily": "Inter"}),
        ], width=6),
    ], className="mb-2"),

    # ── STALENESS WARNING ────────────────────────────────────
    dbc.Alert(
        id="stale-alert",
        is_open=rbi_stale_days() > 65,
        dismissable=True,
        color="warning",
        className="py-2 mb-2",
        style={"fontSize": "13px"},
        children=(
            f"⚠️  Last RBI event was {rbi_stale_days()} days ago. "
            "RBI may have met since. Add the new rate to RBI_EVENTS in GitHub → Render auto-deploys in 3 min."
        )
    ),

    # ── LIVE MACRO BAR ───────────────────────────────────────
    dbc.Row(id="macro-bar", className="mb-3 g-2"),

    html.Hr(style={"borderColor": BORDER, "marginBottom": "0"}),

    # ── TABS ─────────────────────────────────────────────────
    dbc.Tabs(id="main-tabs", active_tab="tab-lag",
             style={"fontFamily": "Inter", "fontSize": "13px"},
             children=[

        # ═══════════════════════════════════════════
        # TAB 1 — LAG ANALYSIS
        # ═══════════════════════════════════════════
        dbc.Tab(label="📊  Lag Analysis", tab_id="tab-lag", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("STOCK", className="mt-3",
                               style={"color": MUTED, "fontSize": "11px",
                                      "letterSpacing": "1px", "fontFamily": "JetBrains Mono"}),
                    dcc.Dropdown(id="lag-ticker",
                        options=[{"label": t, "value": t} for t in ALL_TICKERS],
                        value="PAYTM.NS", clearable=False,
                        style={"fontFamily": "Inter", "fontSize": "13px"}),
                ], width=4),
                dbc.Col([
                    html.Label("MAX LAG (TRADING DAYS)", className="mt-3",
                               style={"color": MUTED, "fontSize": "11px",
                                      "letterSpacing": "1px", "fontFamily": "JetBrains Mono"}),
                    dcc.Slider(id="lag-max", min=10, max=90, step=10, value=45,
                               marks={d: {"label": str(d), "style": {"color": MUTED, "fontSize": "11px"}}
                                      for d in [10, 20, 30, 45, 60, 90]}),
                ], width=6),
                dbc.Col([
                    html.Br(),
                    html.Button("⟳  Refresh", id="lag-refresh", n_clicks=0,
                                className="btn w-100 mt-2",
                                style={"background": "transparent", "border": f"1px solid {ACCENT}",
                                       "color": ACCENT, "fontFamily": "JetBrains Mono",
                                       "fontSize": "12px"}),
                ], width=2),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(dcc.Graph(id="ccf-plot",
                                  style={"height": "430px"},
                                  config={"displayModeBar": False}), width=9),
                dbc.Col(html.Div(id="lag-kpi"), width=3),
            ]),
        ]),

        # ═══════════════════════════════════════════
        # TAB 2 — SUCCESS PROBABILITY
        # ═══════════════════════════════════════════
        dbc.Tab(label="🎯  Success Rate", tab_id="tab-prob", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("STOCK", className="mt-3",
                               style={"color": MUTED, "fontSize": "11px", "letterSpacing": "1px",
                                      "fontFamily": "JetBrains Mono"}),
                    dcc.Dropdown(id="prob-ticker",
                        options=[{"label": t, "value": t} for t in ALL_TICKERS],
                        value="PAYTM.NS", clearable=False,
                        style={"fontFamily": "Inter", "fontSize": "13px"}),
                ], width=3),
                dbc.Col([
                    html.Label("RBI EVENT TYPE", className="mt-3",
                               style={"color": MUTED, "fontSize": "11px", "letterSpacing": "1px",
                                      "fontFamily": "JetBrains Mono"}),
                    dcc.Dropdown(id="prob-event",
                        options=[{"label": e.title(), "value": e}
                                 for e in ["hold", "hike", "cut"]],
                        value="hold", clearable=False,
                        style={"fontFamily": "Inter", "fontSize": "13px"}),
                ], width=3),
                dbc.Col([
                    html.Label("HOLDING PERIOD (DAYS)", className="mt-3",
                               style={"color": MUTED, "fontSize": "11px", "letterSpacing": "1px",
                                      "fontFamily": "JetBrains Mono"}),
                    dcc.Slider(id="prob-horizon", min=5, max=60, step=5, value=30,
                               marks={d: {"label": str(d), "style": {"color": MUTED, "fontSize": "11px"}}
                                      for d in [5, 10, 20, 30, 45, 60]}),
                ], width=6),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(html.Div(id="prob-card"), width=4),
                dbc.Col(dcc.Graph(id="prob-curve",
                                  style={"height": "370px"},
                                  config={"displayModeBar": False}), width=8),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="prob-history",
                                  style={"height": "260px"},
                                  config={"displayModeBar": False}), width=12),
            ], className="mt-2"),
        ]),

        # ═══════════════════════════════════════════
        # TAB 3 — SCENARIO SIMULATOR
        # ═══════════════════════════════════════════
        dbc.Tab(label="🔮  Simulator", tab_id="tab-sim", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("STOCK", className="mt-3",
                               style={"color": MUTED, "fontSize": "11px", "letterSpacing": "1px",
                                      "fontFamily": "JetBrains Mono"}),
                    dcc.Dropdown(id="sim-ticker",
                        options=[{"label": t, "value": t} for t in ALL_TICKERS],
                        value="PAYTM.NS", clearable=False,
                        style={"fontFamily": "Inter", "fontSize": "13px"}),
                ], width=3),
                dbc.Col([
                    html.Label("RATE SHOCK", className="mt-3",
                               style={"color": MUTED, "fontSize": "11px", "letterSpacing": "1px",
                                      "fontFamily": "JetBrains Mono"}),
                    dcc.Dropdown(id="sim-shock",
                        options=[
                            {"label": "−50 bps  Aggressive Cut", "value": -50},
                            {"label": "−25 bps  Cut",            "value": -25},
                            {"label": "   0 bps  Hold",          "value":   0},
                            {"label": "+25 bps  Hike",           "value":  25},
                            {"label": "+50 bps  Aggressive Hike","value":  50},
                        ], value=0, clearable=False,
                        style={"fontFamily": "JetBrains Mono", "fontSize": "12px"}),
                ], width=3),
                dbc.Col([
                    html.Label("FORECAST HORIZON (DAYS)", className="mt-3",
                               style={"color": MUTED, "fontSize": "11px", "letterSpacing": "1px",
                                      "fontFamily": "JetBrains Mono"}),
                    dcc.Slider(id="sim-horizon", min=10, max=90, step=10, value=30,
                               marks={d: {"label": str(d), "style": {"color": MUTED, "fontSize": "11px"}}
                                      for d in [10, 20, 30, 60, 90]}),
                ], width=4),
                dbc.Col([
                    html.Br(),
                    html.Button("▶  RUN", id="sim-btn", n_clicks=0,
                                className="btn w-100 mt-2 fw-bold",
                                style={"background": ACCENT, "border": "none",
                                       "color": "#000", "fontFamily": "JetBrains Mono",
                                       "fontSize": "13px", "letterSpacing": "1px"}),
                ], width=2),
            ], className="mb-3"),

            # Risk metric cards (filled after Run)
            dbc.Row(id="sim-kpi-row", className="mb-3 g-2"),

            # Main chart + distribution side by side
            dbc.Row([
                dbc.Col(dcc.Graph(id="sim-plot",
                                  style={"height": "400px"},
                                  config={"displayModeBar": False}), width=8),
                dbc.Col(dcc.Graph(id="sim-dist",
                                  style={"height": "400px"},
                                  config={"displayModeBar": False}), width=4),
            ]),

            html.Div(id="sim-verdict", className="mt-3"),
        ]),

    ]),

], fluid=True, className="px-4 pb-5",
   style={"backgroundColor": DARK_BG, "minHeight": "100vh"})


# ══════════════════════════════════════════════════════════════
# SECTION 5 — CALLBACKS
# ══════════════════════════════════════════════════════════════

# ── Helper: blank dark figure ─────────────────────────────────
def blank_fig(msg="Loading data..."):
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(color=MUTED, size=14))
    fig.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(t=20, b=20))
    return fig


# ── Shared chart theme ────────────────────────────────────────
def dark_layout(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT, size=13, family="Inter"),
                   x=0.01, pad=dict(t=10)),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=MUTED, size=11, family="Inter"),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, color=MUTED),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, color=MUTED),
        margin=dict(t=45, b=40, l=50, r=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=11)),
    )
    return fig


# ── Header + Macro Bar ────────────────────────────────────────
@app.callback(
    Output("last-updated",  "children"),
    Output("regime-badge",  "children"),
    Output("regime-badge",  "color"),
    Output("macro-bar",     "children"),
    Input("ticker",         "n_intervals"),
    Input("lag-refresh",    "n_clicks"),
)
def update_header(_, __):
    rg_text, rg_color = market_regime()
    macro = live_macro()

    labels = {
        "nifty":     ("Nifty 50",   ""),
        "banknifty": ("Bank Nifty", ""),
        "crude":     ("Crude Oil",  "$/bbl"),
        "usdinr":    ("USD / INR",  "₹"),
    }

    cards = []
    for key, (label, unit) in labels.items():
        d = macro.get(key)
        if not d:
            continue
        arrow = "▲" if d["pct"] >= 0 else "▼"
        clr   = GREEN if d["pct"] >= 0 else RED
        cards.append(dbc.Col(
            html.Div([
                html.P(label, style={"color": MUTED, "fontSize": "10px",
                                     "fontFamily": "JetBrains Mono",
                                     "letterSpacing": "1px", "marginBottom": "2px"}),
                html.Span(f"{d['val']:,.2f}", style={"color": TEXT, "fontSize": "15px",
                                                       "fontFamily": "JetBrains Mono",
                                                       "fontWeight": "600"}),
                html.Span(f"  {unit}", style={"color": MUTED, "fontSize": "10px"}),
                html.Br(),
                html.Span(f"{arrow} {abs(d['pct']):.2f}%",
                           style={"color": clr, "fontSize": "11px",
                                  "fontFamily": "JetBrains Mono"}),
            ], style={"background": CARD_BG, "border": f"1px solid {BORDER}",
                      "borderRadius": "8px", "padding": "10px 14px"}),
            width=3
        ))

    return (
        f"Data refreshed: {DS.last_updated}",
        f"Market  {rg_text}",
        rg_color,
        cards,
    )


# ── Tab 1: Lag Analysis ───────────────────────────────────────
@app.callback(
    Output("ccf-plot",  "figure"),
    Output("lag-kpi",   "children"),
    Input("lag-ticker", "value"),
    Input("lag-max",    "value"),
    Input("lag-refresh","n_clicks"),
)
def update_lag(ticker, max_lag, _):
    if DS.returns is None or ticker not in DS.returns.columns:
        return blank_fig(), html.P("Loading...", style={"color": MUTED})

    ccf = compute_ccf(DS.rbi_change, DS.returns[ticker], max_lag)
    if ccf.empty:
        return blank_fig("Insufficient data"), html.P("—", style={"color": MUTED})

    ci        = 1.96 / np.sqrt(max(1, len(DS.returns[ticker].dropna())))
    peak_lag  = int(ccf.abs().idxmax())
    peak_val  = float(ccf.abs().max())
    is_pos    = float(ccf.iloc[peak_lag]) > 0
    is_sig    = ccf.abs() > ci

    # Color: bright if significant, dim if not
    bar_colors = []
    for v, sig in zip(ccf.values, is_sig.values):
        if sig:
            bar_colors.append(ACCENT if v > 0 else RED)
        else:
            bar_colors.append("#2d3748" if v > 0 else "#2d2030")

    fig = go.Figure()
    fig.add_bar(x=ccf.index, y=ccf.values,
                marker_color=bar_colors,
                hovertemplate="Lag %{x}d → CCF %{y:.4f}<extra></extra>")
    fig.add_hline(y= ci, line_dash="dot", line_color=YELLOW, line_width=1,
                  annotation_text="95% CI", annotation_font_color=YELLOW,
                  annotation_font_size=10)
    fig.add_hline(y=-ci, line_dash="dot", line_color=YELLOW, line_width=1)
    fig.add_vline(x=peak_lag, line_dash="dash", line_color=ACCENT,
                  line_width=1, opacity=0.6)

    dark_layout(fig, f"Cross-Correlation: RBI Rate Change → {ticker} Returns")
    fig.update_xaxes(title_text="Lag (Trading Days)")
    fig.update_yaxes(title_text="Correlation Coefficient")

    # Recent annualised vol
    recent_vol = float(DS.returns[ticker].tail(21).ewm(span=21).std().iloc[-1] * np.sqrt(252) * 100)

    def kpi(label, val, color=TEXT):
        return html.Div([
            html.P(label, style={"color": MUTED, "fontSize": "10px",
                                  "fontFamily": "JetBrains Mono",
                                  "letterSpacing": "1px", "marginBottom": "2px"}),
            html.P(val, style={"color": color, "fontSize": "20px",
                               "fontFamily": "JetBrains Mono", "fontWeight": "600",
                               "marginBottom": "12px"}),
        ])

    sidebar = html.Div([
        html.P("KEY STATS", style={"color": MUTED, "fontSize": "10px",
                                    "letterSpacing": "2px",
                                    "fontFamily": "JetBrains Mono",
                                    "marginTop": "16px", "marginBottom": "16px"}),
        kpi("PEAK LAG",  f"{peak_lag} days", ACCENT),
        kpi("PEAK CCF",  f"{peak_val:.3f}",  TEXT),
        kpi("DIRECTION", "Positive ↑" if is_pos else "Negative ↓",
            GREEN if is_pos else RED),
        kpi("CURRENT VOL (ANN.)", f"{recent_vol:.1f}%", YELLOW),
        html.Hr(style={"borderColor": BORDER}),
        html.P(f"Strongest policy impact on {ticker} arrives "
               f"~{peak_lag} trading days after RBI announcement.",
               style={"color": MUTED, "fontSize": "11px",
                      "fontFamily": "Inter", "lineHeight": "1.6"}),
    ], style={"background": CARD_BG, "border": f"1px solid {BORDER}",
              "borderRadius": "8px", "padding": "0 14px 14px"})

    return fig, sidebar


# ── Tab 2: Success Probability ────────────────────────────────
@app.callback(
    Output("prob-card",    "children"),
    Output("prob-curve",   "figure"),
    Output("prob-history", "figure"),
    Input("prob-ticker",   "value"),
    Input("prob-event",    "value"),
    Input("prob-horizon",  "value"),
)
def update_prob(ticker, event_type, horizon):
    rate, n, rets, dates = calc_success_rate(ticker, event_type, horizon)
    clr = GREEN if rate >= 60 else (YELLOW if rate >= 45 else RED)

    # ── Success card ──────────────────────────────────────────
    avg_ret = round(float(np.mean(rets)), 1) if rets else 0
    card = html.Div([
        html.P("SUCCESS RATE", style={"color": MUTED, "fontSize": "10px",
                                       "letterSpacing": "2px",
                                       "fontFamily": "JetBrains Mono",
                                       "marginBottom": "4px", "marginTop": "16px"}),
        html.H1(f"{rate}%", style={"color": clr, "fontSize": "64px",
                                    "fontFamily": "JetBrains Mono",
                                    "fontWeight": "600", "marginBottom": "4px"}),
        html.P(f"Buying {ticker}", style={"color": TEXT, "fontSize": "13px",
                                           "fontFamily": "Inter"}),
        html.P(f"{horizon}d after {event_type.upper()}",
               style={"color": ACCENT, "fontSize": "12px",
                      "fontFamily": "JetBrains Mono"}),
        html.Hr(style={"borderColor": BORDER, "marginTop": "12px"}),
        html.P(f"Based on {n} past RBI events",
               style={"color": MUTED, "fontSize": "11px", "fontFamily": "Inter"}),
        html.P(f"Avg return: {avg_ret:+.1f}%",
               style={"color": YELLOW, "fontSize": "13px",
                      "fontFamily": "JetBrains Mono"}),
    ], style={"background": CARD_BG, "border": f"1px solid {clr}",
              "borderRadius": "10px", "padding": "0 18px 18px",
              "textAlign": "center", "marginTop": "16px"})

    # ── Horizon curve ─────────────────────────────────────────
    hs = [5, 10, 20, 30, 45, 60]
    rs = [calc_success_rate(ticker, event_type, h)[0] for h in hs]

    fig1 = go.Figure()
    fig1.add_scatter(x=hs, y=rs, mode="lines+markers",
                     line=dict(color=ACCENT, width=2.5),
                     marker=dict(size=8, color=ACCENT),
                     hovertemplate="Horizon %{x}d → %{y:.0f}%<extra></extra>")
    fig1.add_hline(y=50, line_dash="dot", line_color=RED, line_width=1,
                   annotation_text="50% baseline", annotation_font_color=RED,
                   annotation_font_size=10)
    dark_layout(fig1, f"Success Rate vs Holding Period  ·  {ticker}  ·  {event_type}")
    fig1.update_xaxes(title_text="Holding Period (days)")
    fig1.update_yaxes(title_text="Success Rate (%)", range=[0, 100])

    # ── Historical returns bar ────────────────────────────────
    fig2 = go.Figure()
    if rets and dates:
        colors = [GREEN if r > 0 else RED for r in rets]
        fig2.add_bar(x=dates, y=rets, marker_color=colors,
                     hovertemplate="%{x}<br>Return: %{y:.2f}%<extra></extra>")
        fig2.add_hline(y=0, line_color=BORDER, line_width=1)
    dark_layout(fig2, f"Actual Returns After Each {event_type.upper()} Event  ·  {horizon}d hold")
    fig2.update_xaxes(title_text="RBI Event Date")
    fig2.update_yaxes(title_text="Return (%)")

    return card, fig1, fig2


# ── Tab 3: Scenario Simulator ─────────────────────────────────
@app.callback(
    Output("sim-kpi-row", "children"),
    Output("sim-plot",    "figure"),
    Output("sim-dist",    "figure"),
    Output("sim-verdict", "children"),
    Input("sim-btn",      "n_clicks"),
    State("sim-ticker",   "value"),
    State("sim-shock",    "value"),
    State("sim-horizon",  "value"),
    prevent_initial_call=True,
)
def update_sim(_, ticker, shock, horizon):
    res = run_simulation(ticker, shock, horizon, n_paths=2000)
    if res is None:
        return [], blank_fig(), blank_fig(), ""

    days = list(range(1, horizon + 1))

    # ── KPI cards ─────────────────────────────────────────────
    def kpi_card(label, value, color):
        return dbc.Col(html.Div([
            html.P(label, style={"color": MUTED, "fontSize": "10px",
                                  "letterSpacing": "1px",
                                  "fontFamily": "JetBrains Mono",
                                  "marginBottom": "4px"}),
            html.P(value, style={"color": color, "fontSize": "18px",
                                  "fontFamily": "JetBrains Mono",
                                  "fontWeight": "600", "marginBottom": "0"}),
        ], style={"background": CARD_BG, "border": f"1px solid {BORDER}",
                  "borderRadius": "8px", "padding": "10px 14px"}),
        width=True)

    regime_clr = (RED if res["regime"] > 1.3 else
                  YELLOW if res["regime"] > 1.1 else GREEN)
    kpis = [
        kpi_card("SUCCESS PROB.",
                 f"{res['success']}%",
                 GREEN if res["success"] >= 55 else YELLOW if res["success"] >= 45 else RED),
        kpi_card("EXPECTED RETURN",
                 f"{res['exp_ret']:+.2f}%",
                 GREEN if res["exp_ret"] > 0 else RED),
        kpi_card("VALUE AT RISK (5%)",
                 f"{res['var95']:.2f}%",
                 RED),
        kpi_card("ANN. VOLATILITY",
                 f"{res['ann_vol']}%",
                 YELLOW),
        kpi_card("MARKET REGIME",
                 f"{'Stressed' if res['regime'] > 1.1 else 'Calm'}  {res['regime']:.2f}×",
                 regime_clr),
    ]

    # ── Monte Carlo chart ─────────────────────────────────────
    fig = go.Figure()

    # 80% CI band (faint)
    fig.add_scatter(x=days, y=list(res["p90"]), mode="lines",
                    line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=days, y=list(res["p10"]), mode="lines", fill="tonexty",
                    fillcolor="rgba(88,166,255,0.07)",
                    line=dict(width=0), name="80% CI")

    # 50% CI band (slightly brighter)
    fig.add_scatter(x=days, y=list(res["p75"]), mode="lines",
                    line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=days, y=list(res["p25"]), mode="lines", fill="tonexty",
                    fillcolor="rgba(88,166,255,0.14)",
                    line=dict(width=0), name="50% CI")

    # Mean path
    fig.add_scatter(x=days, y=list(res["mean"]), mode="lines",
                    name="Expected Path",
                    line=dict(color=ACCENT, width=2.5),
                    hovertemplate="Day %{x}: %{y:.2f}%<extra></extra>")

    fig.add_hline(y=0, line_dash="dot", line_color=RED, line_width=1.5,
                  annotation_text="Break-even  0%",
                  annotation_font_color=RED, annotation_font_size=10)

    dark_layout(fig,
        f"{ticker}  ·  {shock:+d} bps  ·  {horizon}d forecast  ·  2,000 Monte Carlo paths")
    fig.update_xaxes(title_text="Trading Days After Announcement")
    fig.update_yaxes(title_text="Cumulative Return (%)")

    # ── Return distribution histogram ─────────────────────────
    final = res["final"]
    fig_d = go.Figure()
    fig_d.add_histogram(
        x=final, nbinsx=60,
        marker_color=ACCENT, opacity=0.75, name="Outcomes",
        hovertemplate="Return: %{x:.1f}%<br>Count: %{y}<extra></extra>",
    )
    fig_d.add_vline(x=0,             line_dash="dash",  line_color=RED,    line_width=2)
    fig_d.add_vline(x=res["var95"],  line_dash="dot",   line_color=YELLOW, line_width=1.5,
                    annotation_text=f"VaR  {res['var95']:.1f}%",
                    annotation_font_color=YELLOW, annotation_font_size=10)
    fig_d.add_vline(x=res["exp_ret"],line_dash="dot",   line_color=GREEN,  line_width=1.5,
                    annotation_text=f"E[R]  {res['exp_ret']:.1f}%",
                    annotation_font_color=GREEN,  annotation_font_size=10)

    dark_layout(fig_d, "Return Distribution at Horizon End")
    fig_d.update_xaxes(title_text="Cumulative Return (%)")
    fig_d.update_yaxes(title_text="# Simulated Paths")
    fig_d.update_layout(showlegend=False)

    # ── Verdict alert ─────────────────────────────────────────
    clr_name = ("success" if res["success"] >= 55 else
                 "warning" if res["success"] >= 45 else "danger")
    regime_note = (f"  ⚠️ Market is {res['regime']:.1f}× more volatile than usual — "
                   "widen your stop-loss accordingly."
                   if res["regime"] > 1.1 else "")
    verdict = dbc.Alert([
        html.Strong(f"📊  {res['success']}% probability of positive return  "),
        f"for {ticker} over {horizon} days after a {shock:+d} bps shock.  "
        f"Expected: {res['exp_ret']:+.2f}%  |  "
        f"Worst 5%: {res['var95']:.2f}%  |  "
        f"Ann. Vol: {res['ann_vol']}%.{regime_note}"
    ], color=clr_name, className="text-center",
       style={"fontFamily": "Inter", "fontSize": "13px"})

    return kpis, fig, fig_d, verdict


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
