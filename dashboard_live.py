# ============================================================
# dashboard_live.py — Dashboard Live Streamlit
# ============================================================
# USAGE : streamlit run dashboard_live.py
# Se rafraichit automatiquement toutes les 30 secondes.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
import time, sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Momentum Strategy Dashboard",
    page_icon="📈", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 16px; text-align: center;
    }
    .metric-value { font-size: 28px; font-weight: bold; font-family: monospace; }
    .metric-label { font-size: 12px; color: #8b949e; }
    .positive { color: #3fb950; }
    .negative { color: #f85149; }
    h1, h2, h3 { color: #58a6ff !important; font-family: monospace !important; }
</style>
""", unsafe_allow_html=True)

RESULTS_DIR     = Path("./results")
INITIAL_CAPITAL = 100_000
RISK_FREE_RATE  = 0.05
COLORS = {
    "bg": "#0d1117", "paper": "#161b22", "grid": "#21262d",
    "text": "#c9d1d9", "blue": "#58a6ff", "green": "#3fb950",
    "red": "#f85149", "purple": "#d2a8ff", "gray": "#8b949e",
}

@st.cache_data(ttl=30)
def load_returns():
    val_dir = RESULTS_DIR / "validation"
    if val_dir.exists():
        files = sorted(val_dir.glob("returns_*.csv"))
        if files:
            df = pd.read_csv(files[-1], index_col=0, parse_dates=True)
            return df.iloc[:, 0].dropna()
    bt_dirs = sorted(RESULTS_DIR.glob("backtest_*"))
    if bt_dirs:
        f = bt_dirs[-1] / "returns.csv"
        if f.exists():
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            col = "strategy_returns" if "strategy_returns" in df else df.columns[0]
            return df[col].dropna()
    return None

@st.cache_data(ttl=30)
def load_mc():
    val_dir = RESULTS_DIR / "validation"
    if val_dir.exists():
        files = sorted(val_dir.glob("monte_carlo_*.csv"))
        if files:
            return pd.read_csv(files[-1])
    return None

@st.cache_data(ttl=30)
def load_metrics():
    val_dir = RESULTS_DIR / "validation"
    if val_dir.exists():
        files = sorted(val_dir.glob("summary_*.csv"))
        if files:
            df = pd.read_csv(files[-1], index_col=0)
            raw = df.iloc[:, 0].to_dict()
            result = {}
            for k, v in raw.items():
                try:
                    result[k] = float(v)
                except Exception:
                    result[k] = v
            return result
    return {}

def compute_portfolio(returns):
    simple = np.exp(returns) - 1
    pv     = INITIAL_CAPITAL * (1 + simple).cumprod()
    peak   = pv.expanding().max()
    dd     = (pv - peak) / peak
    return pv, dd

def safe_metric(d, key, default=0.0):
    try:
        return float(d.get(key, default))
    except Exception:
        return default

# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.title("Controls")
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=True)
    oos_split    = st.date_input("OOS Split Date", value=datetime(2022, 1, 1))
    show_bm      = st.checkbox("Show Benchmark", value=True)
    st.divider()
    st.caption(f"Last update: {datetime.now().strftime('%H:%M:%S')}")
    if st.button("Force Refresh"):
        st.cache_data.clear()
        st.rerun()

# ── CHARGEMENT ─────────────────────────────────────────────
returns = load_returns()
mc_data = load_mc()
metrics = load_metrics()

if returns is None:
    st.error("Aucune donnee trouvee. Lance run_backtest.py et validator.py d\'abord.")
    st.stop()

pv, dd   = compute_portfolio(returns)
oos_date = pd.Timestamp(oos_split)
rf       = (1 + RISK_FREE_RATE) ** (1/252) - 1

cagr   = float((pv.iloc[-1] / INITIAL_CAPITAL) ** (252/len(returns)) - 1)
sharpe = float((returns.mean() - rf) / returns.std() * np.sqrt(252))
max_dd = float(dd.min())
vol    = float(returns.std() * np.sqrt(252))

# ── HEADER ─────────────────────────────────────────────────
st.title("MOMENTUM STRATEGY DASHBOARD")
st.caption(f"Periode : {returns.index[0].date()} -> {returns.index[-1].date()} | Capital : ${INITIAL_CAPITAL:,.0f}")
st.divider()

# ── METRIQUES ──────────────────────────────────────────────
cols = st.columns(6)
display = [
    ("CAGR",       cagr,   ".1%", cagr > 0),
    ("Sharpe",     sharpe, ".3f", sharpe > 0.5),
    ("Max DD",     max_dd, ".1%", max_dd > -0.20),
    ("Volatilite", vol,    ".1%", True),
    ("OOS CAGR",   safe_metric(metrics, "oos_cagr"),   ".1%", True),
    ("OOS Sharpe", safe_metric(metrics, "oos_sharpe"), ".3f", True),
]
for col, (label, value, fmt, good) in zip(cols, display):
    css = "positive" if good else "negative"
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value {css}">{format(value, fmt)}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── EQUITY CURVE + DRAWDOWN ────────────────────────────────
fig_main = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.7, 0.3], vertical_spacing=0.03,
    subplot_titles=["Equity Curve", "Drawdown"]
)

is_pv  = pv[pv.index < oos_date]
oos_pv = pv[pv.index >= oos_date]

fig_main.add_trace(go.Scatter(
    x=is_pv.index, y=is_pv.values, name="IS",
    line=dict(color=COLORS["blue"], width=2),
    hovertemplate="%{x|%Y-%m-%d} $%{y:,.0f}<extra>IS</extra>"
), row=1, col=1)

fig_main.add_trace(go.Scatter(
    x=oos_pv.index, y=oos_pv.values, name="OOS",
    line=dict(color=COLORS["green"], width=2),
    hovertemplate="%{x|%Y-%m-%d} $%{y:,.0f}<extra>OOS</extra>"
), row=1, col=1)

if show_bm:
    bm = INITIAL_CAPITAL * (1 + pd.Series(0.00025, index=returns.index)).cumprod()
    fig_main.add_trace(go.Scatter(
        x=bm.index, y=bm.values, name="Benchmark",
        line=dict(color=COLORS["gray"], width=1, dash="dot"), opacity=0.5
    ), row=1, col=1)

fig_main.add_trace(go.Scatter(
    x=dd.index, y=dd.values * 100, name="Drawdown",
    fill="tozeroy", line=dict(color=COLORS["red"], width=1),
    fillcolor="rgba(248,81,73,0.3)"
), row=2, col=1)

fig_main.add_vrect(x0=oos_date, x1=pv.index[-1],
                   fillcolor=COLORS["green"], opacity=0.04, layer="below", line_width=0)
fig_main.update_layout(
    height=600, paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["paper"],
    font=dict(color=COLORS["text"], family="monospace"),
    legend=dict(bgcolor="rgba(22,27,34,0.8)"), hovermode="x unified"
)
fig_main.update_xaxes(gridcolor=COLORS["grid"])
fig_main.update_yaxes(gridcolor=COLORS["grid"])
fig_main.update_yaxes(tickprefix="$", tickformat=",.0f", row=1, col=1)
fig_main.update_yaxes(ticksuffix="%", row=2, col=1)
st.plotly_chart(fig_main, use_container_width=True)

# ── MONTE CARLO + ANNUEL ───────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("Monte Carlo Distribution")
    if mc_data is not None and "sharpes" in mc_data.columns:
        sharpes  = mc_data["sharpes"].dropna()
        observed = sharpe
        p5       = float(np.percentile(sharpes, 5))
        fig_mc   = go.Figure()
        fig_mc.add_trace(go.Histogram(x=sharpes, nbinsx=50, marker_color=COLORS["blue"], opacity=0.7))
        fig_mc.add_vline(x=float(observed), line_color=COLORS["green"], line_width=2,
                        annotation_text=f"Obs: {float(observed):.3f}", annotation_font_color=COLORS["green"])
        fig_mc.add_vline(x=p5, line_color=COLORS["red"], line_dash="dash",
                        annotation_text=f"5pct: {p5:.3f}", annotation_font_color=COLORS["red"])
        fig_mc.update_layout(height=300, paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["paper"],
                             font=dict(color=COLORS["text"]), showlegend=False)
        st.plotly_chart(fig_mc, use_container_width=True)
    else:
        st.info("Lance validator.py pour les donnees Monte Carlo")

with col_r:
    st.subheader("Performance Annuelle")
    yearly = {}
    for year in returns.index.year.unique():
        yr = returns[returns.index.year == year].dropna()
        if len(yr) < 50:
            continue
        pvy  = INITIAL_CAPITAL * (1 + (np.exp(yr) - 1)).cumprod()
        c    = float((pvy.iloc[-1] / INITIAL_CAPITAL) ** (252/len(yr)) - 1)
        yearly[year] = c

    years  = sorted(yearly.keys())
    cagrs  = [yearly[y] * 100 for y in years]
    colors = [COLORS["green"] if c > 0 else COLORS["red"] for c in cagrs]
    fig_yr = go.Figure(go.Bar(
        x=years, y=cagrs, marker_color=colors,
        text=[f"{c:.1f}%" for c in cagrs], textposition="outside", opacity=0.85
    ))
    fig_yr.update_layout(height=300, paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["paper"],
                         font=dict(color=COLORS["text"]), showlegend=False)
    fig_yr.update_xaxes(gridcolor=COLORS["grid"])
    fig_yr.update_yaxes(gridcolor=COLORS["grid"], ticksuffix="%")
    st.plotly_chart(fig_yr, use_container_width=True)

# ── ROLLING SHARPE ─────────────────────────────────────────
st.subheader("Rolling Sharpe (252j)")
roll_s = (returns.rolling(252).mean() - rf) / returns.rolling(252).std() * np.sqrt(252)
fig_rs = go.Figure()
fig_rs.add_trace(go.Scatter(
    x=roll_s.index, y=roll_s.values, name="Rolling Sharpe",
    line=dict(color=COLORS["purple"], width=1.5), fill="tozeroy"
))
fig_rs.add_hline(y=0,   line_color=COLORS["gray"],  line_width=0.8)
fig_rs.add_hline(y=0.5, line_color=COLORS["green"], line_dash="dot",
                annotation_text="Seuil 0.5", annotation_font_color=COLORS["green"])
fig_rs.update_layout(height=250, paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["paper"],
                     font=dict(color=COLORS["text"]), showlegend=False)
fig_rs.update_xaxes(gridcolor=COLORS["grid"])
fig_rs.update_yaxes(gridcolor=COLORS["grid"])
st.plotly_chart(fig_rs, use_container_width=True)

st.divider()
st.caption(f"Mise a jour : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Strategie Momentum Phase 2D")

if auto_refresh:
    time.sleep(30)
    st.rerun()
