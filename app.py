import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta

# =============================================================================
# INITIAL CONFIGURATION
# =============================================================================
st.set_page_config(page_title="Star PM StatArb Portfolio Console", page_icon="👑", layout="wide")
st.title("👑 Institutional Kalman Filter Pairs Trading Engine")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ Strategy Allocation & Risk Rules")
ticker_A = st.sidebar.text_input("Asset A", value="SPY")
ticker_B = st.sidebar.text_input("Asset B", value="QQQ")
entry_threshold = st.sidebar.slider("Z-Score Entry (σ)", 1.0, 3.5, 2.0)
exit_threshold = st.sidebar.slider("Z-Score Exit (σ)", 0.0, 1.0, 0.0)
MAX_TRADE_DURATION = 20

# =============================================================================
# DATA PIPELINE
# =============================================================================
@st.cache_data(ttl=3600)
def fetch_data(t1, t2):
    return yf.download([t1, t2, "SPY", "^VIX"], period="4y", auto_adjust=True)['Adj Close'].dropna()

market_df = fetch_data(ticker_A, ticker_B)

if market_df.empty:
    st.error("Market data ingestion failed. Please check tickers.")
    st.stop()

df = market_df[[ticker_A, ticker_B, "SPY"]].copy()
df['VIX'] = market_df["^VIX"]

# =============================================================================
# KALMAN & STRATEGY ENGINE
# =============================================================================
# (Simplified Kalman logic omitted for brevity, ensure your math remains here)
# ... [Insert your Kalman math here] ...

# =============================================================================
# RISK MANAGEMENT & PROTECTION BLOCK
# =============================================================================
days_in_trade = 0
current_state = 0
execution_counter = 0
positions = []

for idx, row in df.iterrows():
    # Z-Score logic
    z = row.get('Z_Score', 0)
    
    # Zombie Liquidation Rule
    if current_state != 0:
        days_in_trade += 1
        if days_in_trade >= MAX_TRADE_DURATION:
            current_state = 0
            days_in_trade = 0
            execution_counter += 1
    else:
        days_in_trade = 0
    
    # Entry/Exit Logic
    if current_state == 0:
        if z <= -entry_threshold: current_state = 1
        elif z >= entry_threshold: current_state = -1
    elif (current_state == 1 and z >= -exit_threshold) or (current_state == -1 and z <= exit_threshold):
        current_state = 0
    
    positions.append(current_state)

df['Position'] = positions

# CALCULATE PERFORMANCE
df['CumMax'] = df['Net_Asset_Value'].cummax()
df['Drawdown'] = (df['Net_Asset_Value'] - df['CumMax']) / df['CumMax']

# =============================================================================
# VISUALIZATION
# =============================================================================
fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
fig.add_trace(go.Scatter(x=df.index, y=df['Net_Asset_Value'], name="Equity Curve"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Drawdown'], name="Drawdown", fill='tozeroy'), row=2, col=1)
st.plotly_chart(fig, use_container_width=True)