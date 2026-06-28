import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta

# =============================================================================
# STYLING & INSTITUTIONAL CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Star PM StatArb Portfolio Console",
    page_icon="👑",
    layout="wide"
)

st.title("👑 Institutional Kalman Filter Pairs Trading Engine")
st.markdown("""
*Proprietary Multi-Asset Alpha Generation Platform* | Author: Lead Portfolio Manager
This terminal runs real-time state-space estimation models to trade cointegrated asset pairs. It includes 
dynamic Kalman updates, risk-targeted leverage adjustment loops, tail-risk trailing liquidations, and deep visual risk analytics.
""")

# =============================================================================
# SIDEBAR SYSTEM CONTROLS (ALPHA PARAMETERS & RISK RULES)
# =============================================================================
st.sidebar.header("🕹️ Strategy Allocation & Risk Rules")

asset_class = st.sidebar.selectbox(
    "Asset Class Universe Target",
    ["Global Equities (SPY/QQQ)", "Precious Metals Futures (GC=F/SI=F)", "Custom Cross-Asset Pair"]
)

if asset_class == "Global Equities (SPY/QQQ)":
    default_A, default_B = "SPY", "QQQ"
elif asset_class == "Precious Metals Futures (GC=F/SI=F)":
    default_A, default_B = "GC=F", "SI=F"
else:
    default_A, default_B = "AAPL", "MSFT"

ticker_A = st.sidebar.text_input("Asset A (Dependent variable Y)", value=default_A)
ticker_B = st.sidebar.text_input("Asset B (Independent variable X)", value=default_B)

entry_threshold = st.sidebar.slider("Z-Score Entry Vector (σ)", min_value=1.0, max_value=3.5, value=2.0, step=0.1)
exit_threshold = st.sidebar.slider("Z-Score Mean-Reversion Target (σ)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

st.sidebar.subheader("🛡️ Portfolio Risk Mandates")
vol_target_pct = st.sidebar.slider("Target Strategy Volatility (%)", min_value=5.0, max_value=25.0, value=12.0, step=1.0)
trailing_stop_multiplier = st.sidebar.slider("Trailing Downside Deviation Stop Window", min_value=2.0, max_value=5.0, value=3.0, step=0.5)

st.sidebar.subheader("🌪️ Macro Volatility Regime Filter")
vix_filter_active = st.sidebar.checkbox("Activate VIX Volatility Filter", value=True)
max_vix_threshold = st.sidebar.slider("Maximum Allowed VIX Level for Entry", min_value=15.0, max_value=40.0, value=30.0, step=1.0)

st.sidebar.subheader("⚙️ Capital Allocations & Friction Drag")
initial_capital = st.sidebar.number_input("Gross Risk Capital Allocation ($)", min_value=100000, value=10000000, step=1000000)
slippage_bps = st.sidebar.slider("Execution Slip + Clearing Costs (Bps/Trade)", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
borrow_cost_ann = st.sidebar.slider("Annual Short Leg Borrow Cost Fee (%)", min_value=0.0, max_value=5.0, value=1.5, step=0.25)

# =============================================================================
# DATA ACQUISITION & INTEGRITY PIPELINE
# =============================================================================
@st.cache_data(ttl=3600)
def fetch_clean_market_data(t1, t2, use_vix=True):
    """Streams data from global market feeds alongside structural benchmarks."""
    end_dt = datetime.today().strftime('%Y-%m-%d')
    start_dt = (datetime.today() - timedelta(days=4*365)).strftime('%Y-%m-%d')
    tickers_to_fetch = [t1, t2, "SPY"]
    if use_vix:
        tickers_to_fetch.append("^VIX")
        
    try:
        data = yf.download(tickers_to_fetch, start=start_dt, end=end_dt)['Adj Close']
        return data.dropna()
    except Exception as e:
        st.error(f"Market Ingestion Pipeline Disrupted: {str(e)}")
        return pd.DataFrame()

market_df = fetch_clean_market_data(ticker_A, ticker_B, use_vix=vix_filter_active)

if market_df.empty or ticker_A not in market_df.columns or ticker_B not in market_df.columns:
    st.warning("Systems offline. Verify that tickers match valid formatting rules.")
else:
    df = market_df[[ticker_A, ticker_B, "SPY"]].copy()
    if vix_filter_active and "^VIX" in market_df.columns:
        df['VIX'] = market_df["^VIX"]
    else:
        df['VIX'] = 0.0

    # =============================================================================
    # MATHEMATICAL VECTOR ENGINE: KALMAN FILTER STATE-SPACE MODEL
    # =============================================================================
    obs_y = df[ticker_A].values
    obs_x = df[ticker_B].values
    n_obs = len(df)
    
    state_means = np.zeros((n_obs, 2))  
    state_covs = np.zeros((n_obs, 2, 2))
    
    delta = 1e-4
    Q = delta / (1 - delta) * np.eye(2) 
    R = 1e-2                             
    
    current_mean = np.zeros(2)
    current_cov = np.eye(2)
    
    filtered_spreads = np.zeros(n_obs)
    filtered_spread_stds = np.zeros(n_obs)
    
    for t in range(n_obs):
        pred_mean = current_mean
        pred_cov = current_cov + Q
        H = np.array([1.0, obs_x[t]])
        y_hat = np.dot(H, pred_mean)
        innovation = obs_y[t] - y_hat
        innovation_covariance = np.dot(H, np.dot(pred_cov, H.T)) + R
        kalman_gain = np.dot(pred_cov, H.T) / innovation_covariance
        current_mean = pred_mean + kalman_gain * innovation
        current_cov = pred_cov - np.outer(kalman_gain, H).dot(pred_cov)
        state_means[t] = current_mean
        state_covs[t] = current_cov
        filtered_spreads[t] = innovation
        filtered_spread_stds[t] = np.sqrt(innovation_covariance)
        
    df['Hedge_Ratio'] = state_means[:, 1]
    df['Intercept'] = state_means[:, 0]
    df['Spread'] = filtered_spreads
    df['Spread_Std'] = filtered_spread_stds
    df['Z_Score'] = df['Spread'] / df['Spread_Std']
    
    df = df.iloc[30:].copy()

    adf_test = adfuller(df['Spread'])
    p_val = adf_test[1]
    is_stationary = p_val < 0.05

    # =============================================================================
    # RISK MANAGEMENT VECTORIZATION: VOLATILITY SCALING & TRAILING STOP LOGIC
    # =============================================================================
    df['Spread_Return_Daily'] = df['Spread'].pct_change().fillna(0)
    df['Spread_Vol_Ann'] = df['Spread_Return_Daily'].rolling(window=20).std() * np.sqrt(252)
    df['Spread_Vol_Ann'] = df['Spread_Vol_Ann'].replace(0, np.nan).bfill()
    
    df['Leverage_Multiplier'] = (vol_target_pct / 100) / df['Spread_Vol_Ann']
    df['Leverage_Multiplier'] = df['Leverage_Multiplier'].clip(lower=0.1, upper=3.0).fillna(1.0)

    positions = []
    current_state = 0  
    execution_counter = 0
    peak_spread_value = 0.0

    for idx, row in df.iterrows():
        z = row['Z_Score']
        current_spread = row['Spread']
        rolling_std = row['Spread_Std']
        vix_level = row['VIX']
        
        if current_state != 0:
            if current_state == 1:
                peak_spread_value = max(peak_spread_value, current_spread)
                if current_spread < (peak_spread_value - (trailing_stop_multiplier * rolling_std)):
                    current_state = 0
                    execution_counter += 1
            elif current_state == -1:
                peak_spread_value = min(peak_spread_value, current_spread)
                if current_spread > (peak_spread_value + (trailing_stop_multiplier * rolling_std)):
                    current_state = 0
                    execution_counter += 1

        if current_state == 0:
            if vix_filter_active and vix_level > max_vix_threshold:
                positions.append(0)
                continue
            if z <= -entry_threshold:
                current_state = 1
                peak_spread_value = current_spread
                execution_counter += 1
            elif z >= entry_threshold:
                current_state = -1
                peak_spread_value = current_spread
                execution_counter += 1
        elif current_state == 1 and z >= -exit_threshold:
            current_state = 0
            execution_counter += 1
        elif current_state == -1 and z <= exit_threshold:
            current_state = 0
            execution_counter += 1
                
        positions.append(current_state)

    df['Position'] = positions

    # =============================================================================
    # TRANSACTION COST & PERFORMANCE TRACKING ENGINE
    # =============================================================================
    df['Ret_A'] = df[ticker_A].pct_change()
    df['Ret_B'] = df[ticker_B].pct_change()
    df['Benchmark_Return'] = df['SPY'].pct_change().fillna(0)
    
    df['Spread_Return_Proxy'] = df['Ret_A'] - (df['Hedge_Ratio'] * df['Ret_B'])
    df['Scaled_Strategy_Return'] = df['Position'].shift(1) * df['Spread_Return_Proxy'] * df['Leverage_Multiplier'].shift(1)
    df['Scaled_Strategy_Return'] = df['Scaled_Strategy_Return'].fillna(0)

    df['State_Transitions'] = df['Position'].diff().abs().fillna(0)
    df['Slippage_Cost'] = df['State_Transitions'] * (slippage_bps / 10000)
    
    daily_borrow_fee = (borrow_cost_ann / 100) / 252
    df['Borrow_Cost'] = np.where(df['Position'] != 0, daily_borrow_fee, 0.0)
    
    df['Net_Strategy_Return'] = df['Scaled_Strategy_Return'] - df['Slippage_Cost'] - df['Borrow_Cost']
    df['Cumulative_Return'] = (1 + df['Net_Strategy_Return']).cumprod()
    df['Net_Asset_Value'] = initial_capital * df['Cumulative_Return']

    # Strategy Realized Rolling Portfolio Volatility
    df['Strategy_Realized_Vol_Ann'] = df['Net_Strategy_Return'].rolling(window=20).std() * np.sqrt(252) * 100

    # =============================================================================
    # PERFORMANCE METRIC MATRIX CALCULATOR
    # --- Risk Management Engine (with Zombie Liquidation) ---
MAX_TRADE_DURATION = 20  # Max days to hold a position
days_in_trade = 0
# Ensure df is defined and not empty before looping
# 1. First, make sure 'df' actually exists
if 'df' in locals() and df is not None and not df.empty:
    
    # 2. Now run your loop safely
    for idx, row in df.iterrows():
        # ZOMBIE LIQUIDATION RULE:
        if days_in_trade >= MAX_TRADE_DURATION:
            current_state = 0
            days_in_trade = 0
            execution_counter += 1
        else:
            days_in_trade = 0
            
    # 3. Finally, do your calculations
    df['CumMax'] = df['Net_Asset_Value'].cummax()
    df['Drawdown'] = (df['Net_Asset_Value'] - df['CumMax']) / df['CumMax']

else:
    # If df is missing, stop the app and show a helpful error
    st.error("Data ingestion failed. 'df' is empty or missing.")
    st.stop()
            
        # ZOMBIE LIQUIDATION RULE:
        if days_in_trade >= MAX_TRADE_DURATION:
            # ... (your existing closure logic)
            
else:
    st.error("Market data is missing. The ingestion pipeline failed.")
    st.stop()

        
 

# VISUALIZE EQUITY VS DRAWDOWN
fig_perf = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
fig_perf.add_trace(go.Scatter(x=df.index, y=df['Net_Asset_Value'], name="Equity Curve"), row=1, col=1)
fig_perf.add_trace(go.Fill(x=df.index, y=df['Drawdown'], name="Drawdown", fill='tozeroy'), row=2, col=1)
st.plotly_chart(fig_perf, use_container_width=True)
# DYNAMIC POSITION SIZING BASED ON VIX REGIME
df['Volatility_Adjustment'] = np.where(df['VIX'] > max_vix_threshold, 0.5, 1.0)
df['Scaled_Strategy_Return'] = df['Position'].shift(1) * df['Spread_Return_Proxy'] * df['Leverage_Multiplier'].shift(1) * df['Volatility_Adjustment']
# Add this to your loop to build an audit trail
audit_log = []
if df['Position'].diff().abs().iloc[-1] > 0:
    audit_log.append(f"State Change at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: New Position = {current_state}")

# Display the audit log at the bottom of the dashboard
with st.expander("📜 Strategy Audit Trail"):
    st.write(audit_log)
