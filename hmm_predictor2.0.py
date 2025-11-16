import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from nselib import capital_market
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, time, timedelta
import warnings
import sys

warnings.filterwarnings("ignore")

# -------------------------------------------------
# STREAMLIT PAGE SETUP
# -------------------------------------------------
st.set_page_config(
    layout="wide", 
    page_title="Markov Chain Predictor", 
    initial_sidebar_state="expanded",
    page_icon="📊"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #60a5fa 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .logo {
        font-size: 2.5rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.5rem;
    }
    .tagline {
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        font-weight: 500;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <div class="logo">📊 Markov Chain Trend Probability Predictor</div>
    <div class="tagline">Pure Markov Chain Analysis with Brier Score Validation & Beta Calculation</div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD NSE SYMBOLS WITH ERROR HANDLING
# -------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_symbols():
    """Load NSE symbols with fallback"""
    try:
        df = capital_market.equity_list()
        symbols_list = sorted(df["SYMBOL"].unique().tolist())
        if len(symbols_list) > 0:
            return symbols_list
        else:
            return get_fallback_symbols()
    except Exception as e:
        st.warning(f"⚠️ Could not load NSE symbols from API. Using fallback list. Error: {str(e)}")
        return get_fallback_symbols()

def get_fallback_symbols():
    """Fallback list of popular NSE symbols"""
    return [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", 
        "SBIN", "BHARTIARTL", "HINDUNILVR", "ITC", "KOTAKBANK",
        "LT", "AXISBANK", "WIPRO", "ASIANPAINT", "MARUTI",
        "TITAN", "BAJFINANCE", "HCLTECH", "ULTRACEMCO", "NESTLEIND"
    ]

# Load symbols
with st.spinner("Loading symbols..."):
    symbols = load_symbols()

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration Panel")
    st.markdown("---")
    
    symbol = st.selectbox("📊 Select Symbol", symbols, index=0)
    
    st.markdown("##### Date Range")
    end_date = st.date_input("End Date", datetime.now().date())
    start_date = st.date_input("Start Date", end_date - timedelta(days=365))
    
    st.markdown("---")
    timeframe = st.selectbox("⏱️ Timeframe", ["1d", "1h", "2h", "1wk"], index=0)
    
    st.markdown("---")
    st.markdown("##### Markov Chain Parameters")
    lookback_period = st.slider("Lookback Period (n)", 5, 30, 14)
    atr_threshold = st.slider("ATR Threshold Multiplier", 0.1, 2.0, 0.5, 0.1)
    history_length = st.slider("Historical Periods", 10, 100, 33)
    atr_length = st.slider("ATR Length", 5, 50, 14)
    brier_lookback = st.slider("Brier Score Lookback", 10, 100, 20)
    
    st.markdown("---")
    st.markdown("##### Bollinger Bands Settings")
    bb_period = st.slider("BB Period", 10, 30, 20)
    bb_std = st.slider("BB Standard Deviation", 1.0, 3.0, 2.0, 0.5)
    
    st.markdown("---")
    run = st.button("🚀 Run Analysis")
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.info("""
    **Version:** 1.0  
    **Data Source:** Yahoo Finance  
    **Method:** Markov Chain  
    
    This tool uses statistical methods to analyze market trends. Not financial advice.
    """)

# -------------------------------------------------
# FETCH OHLC FUNCTION WITH RETRY LOGIC
# -------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlc(symbol, start, end, interval, max_retries=3):
    """Fetch OHLC data with retry logic"""
    for attempt in range(max_retries):
        try:
            if not isinstance(start, datetime):
                start = datetime.combine(start, time.min)
            if not isinstance(end, datetime):
                end = datetime.combine(end, time.min)
            
            now = datetime.now()
            if end > now:
                end = now
            
            start_ext = start - timedelta(days=180)
            yf_symbol = symbol + ".NS"
            
            df = yf.download(
                yf_symbol, 
                start=start_ext, 
                end=end + timedelta(days=1), 
                interval=interval, 
                progress=False,
                timeout=10
            )
            
            if df.empty:
                if attempt < max_retries - 1:
                    continue
                return None, f"❌ No data available for {symbol}"
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
            
            if "Datetime" in df.columns:
                df = df.rename(columns={"Datetime": "Date"})
            
            if "Date" not in df.columns:
                return None, "❌ No Date column returned."
            
            required = ["Open", "High", "Low", "Close", "Volume"]
            missing = [c for c in required if c not in df.columns]
            
            if missing:
                return None, f"❌ Missing OHLC columns: {missing}"
            
            for c in required:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            
            df = df.dropna(subset=["Close"]).reset_index(drop=True)
            
            if df.empty:
                return None, "❌ No valid OHLC rows after cleaning."
            
            return df, None
        
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            return None, f"❌ Fetch Exception: {str(e)}"
    
    return None, "❌ Failed after multiple retries"

# -------------------------------------------------
# FETCH NIFTY 50 DATA
# -------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_nifty50(start, end, interval, max_retries=3):
    """Fetch Nifty 50 data with retry logic"""
    for attempt in range(max_retries):
        try:
            if not isinstance(start, datetime):
                start = datetime.combine(start, time.min)
            if not isinstance(end, datetime):
                end = datetime.combine(end, time.min)
            
            now = datetime.now()
            if end > now:
                end = now
            
            start_ext = start - timedelta(days=180)
            
            df = yf.download(
                "^NSEI", 
                start=start_ext, 
                end=end + timedelta(days=1), 
                interval=interval, 
                progress=False,
                timeout=10
            )
            
            if df.empty:
                if attempt < max_retries - 1:
                    continue
                return None, "❌ No Nifty 50 data available"
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
            
            if "Datetime" in df.columns:
                df = df.rename(columns={"Datetime": "Date"})
            
            df = df[["Date", "Close"]].copy()
            df = df.rename(columns={"Close": "Nifty_Close"})
            df["Nifty_Return"] = df["Nifty_Close"].pct_change()
            
            return df.dropna().reset_index(drop=True), None
        
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            return None, f"❌ Nifty Fetch Error: {str(e)}"
    
    return None, "❌ Failed to fetch Nifty data after retries"

# -------------------------------------------------
# CALCULATE BETA
# -------------------------------------------------
def calculate_beta(stock_returns, market_returns):
    """Calculate beta coefficient"""
    try:
        merged = pd.DataFrame({'stock': stock_returns, 'market': market_returns}).dropna()
        
        if len(merged) < 30:
            return None, "Insufficient data for beta calculation"
        
        covariance = np.cov(merged['stock'], merged['market'])[0][1]
        market_variance = np.var(merged['market'])
        
        if market_variance == 0:
            return None, "Market variance is zero"
        
        beta = covariance / market_variance
        correlation = np.corrcoef(merged['stock'], merged['market'])[0][1]
        r_squared = correlation ** 2
        
        return {'beta': beta, 'r_squared': r_squared, 'correlation': correlation}, None
    
    except Exception as e:
        return None, f"Beta calculation error: {str(e)}"

# -------------------------------------------------
# BOLLINGER BANDS
# -------------------------------------------------
def calculate_bollinger_bands(df, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    df = df.copy()
    df['BB_Middle'] = df['Close'].rolling(window=period).mean()
    rolling_std = df['Close'].rolling(window=period).std()
    df['BB_Upper'] = df['BB_Middle'] + (rolling_std * std_dev)
    df['BB_Lower'] = df['BB_Middle'] - (rolling_std * std_dev)
    df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']) * 100
    df['BB_Position'] = ((df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])) * 100
    return df

# -------------------------------------------------
# MARKOV CHAIN STATE IDENTIFICATION
# -------------------------------------------------
def identify_states(df, lookback, atr_threshold, atr_length):
    """Identify market states using ATR-normalized price changes"""
    df = df.copy()
    
    # Calculate ATR
    df['H_L'] = df['High'] - df['Low']
    df['H_PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L_PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H_L', 'H_PC', 'L_PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=atr_length).mean()
    
    df['Price_Change'] = df['Close'] - df['Close'].shift(lookback)
    df['ATR_Normalized_Change'] = df['Price_Change'] / df['ATR']
    
    UPTREND = 1
    DOWNTREND = -1
    
    df['State'] = np.nan
    current_state = UPTREND
    
    for idx in df.index:
        atr_norm = df.loc[idx, 'ATR_Normalized_Change']
        
        if pd.notna(atr_norm):
            if atr_norm > atr_threshold:
                current_state = UPTREND
            elif atr_norm < -atr_threshold:
                current_state = DOWNTREND
        
        df.loc[idx, 'State'] = current_state
    
    df['State_Name'] = df['State'].map({1: 'Uptrend', -1: 'Downtrend'})
    
    return df

# -------------------------------------------------
# CALCULATE TRANSITION PROBABILITIES
# -------------------------------------------------
def calculate_transition_probabilities(state_history):
    """Calculate Markov chain transition probabilities"""
    if len(state_history) < 2:
        return {
            'up_to_up': 0.0,
            'up_to_down': 0.0,
            'down_to_up': 0.0,
            'down_to_down': 0.0
        }
    
    UPTREND = 1
    DOWNTREND = -1
    
    def calc_prob(from_state, to_state):
        transitions = 0
        total_from_state = 0
        
        for i in range(len(state_history) - 1):
            current = state_history[i]
            next_state = state_history[i + 1]
            
            if current == from_state:
                total_from_state += 1
                if next_state == to_state:
                    transitions += 1
        
        return transitions / total_from_state if total_from_state > 0 else 0.0
    
    return {
        'up_to_up': calc_prob(UPTREND, UPTREND),
        'up_to_down': calc_prob(UPTREND, DOWNTREND),
        'down_to_up': calc_prob(DOWNTREND, UPTREND),
        'down_to_down': calc_prob(DOWNTREND, DOWNTREND)
    }

# -------------------------------------------------
# CALCULATE STATE PROBABILITIES
# -------------------------------------------------
def calculate_state_probabilities(state_history):
    """Calculate steady-state probabilities"""
    if len(state_history) == 0:
        return {'uptrend': 0.5, 'downtrend': 0.5}
    
    UPTREND = 1
    uptrend_count = sum(1 for s in state_history if s == UPTREND)
    total = len(state_history)
    
    return {
        'uptrend': uptrend_count / total if total > 0 else 0.5,
        'downtrend': (total - uptrend_count) / total if total > 0 else 0.5
    }

# -------------------------------------------------
# CALCULATE BRIER SCORE
# -------------------------------------------------
def calculate_brier_score(predictions, actuals):
    """Calculate Brier score (lower is better)"""
    if len(predictions) == 0 or len(actuals) == 0:
        return None
    
    if len(predictions) != len(actuals):
        min_len = min(len(predictions), len(actuals))
        predictions = predictions[-min_len:]
        actuals = actuals[-min_len:]
    
    sum_squared_diff = sum((pred - actual) ** 2 for pred, actual in zip(predictions, actuals))
    return sum_squared_diff / len(predictions)

def get_brier_classification(score):
    """Classify Brier score"""
    if score is None:
        return "Calculating...", "gray"
    elif score < 0.10:
        return "Exceptional", "lime"
    elif score < 0.15:
        return "Excellent", "green"
    elif score < 0.20:
        return "Very Good", "blue"
    elif score < 0.25:
        return "Good", "orange"
    elif score < 0.30:
        return "Fair", "yellow"
    else:
        return "Poor", "red"

# -------------------------------------------------
# CALCULATE CONFIDENCE
# -------------------------------------------------
def calculate_confidence(prob_uptrend, prob_downtrend):
    """Calculate confidence"""
    return abs(prob_uptrend - prob_downtrend)

# -------------------------------------------------
# MAIN RUN
# -------------------------------------------------
if run:
    with st.spinner(f"🔄 Fetching data for {symbol} and Nifty 50..."):
        df, err = fetch_ohlc(symbol, start_date, end_date, timeframe)
        nifty_df, nifty_err = fetch_nifty50(start_date, end_date, timeframe)
    
    if err:
        st.error(err)
        st.info("💡 Try selecting a different symbol or date range.")
        st.stop()
    
    if nifty_err:
        st.warning(f"⚠️ {nifty_err} - Beta calculation will be skipped")
        nifty_df = None
    
    st.success(f"✅ Loaded {len(df)} data points")
    
    # Calculate returns
    df["Return"] = df["Close"].pct_change()
    
    # Identify states
    df = identify_states(df, lookback_period, atr_threshold, atr_length)
    
    # Calculate Bollinger Bands
    df = calculate_bollinger_bands(df, period=bb_period, std_dev=bb_std)
    
    # Beta calculation
    beta_info = None
    if nifty_df is not None:
        df_merged = df.merge(nifty_df[['Date', 'Nifty_Return']], on='Date', how='left')
        beta_info, beta_err = calculate_beta(df_merged['Return'], df_merged['Nifty_Return'])
        if beta_err:
            st.warning(f"⚠️ {beta_err}")
    
    df = df.dropna().reset_index(drop=True)
    
    if len(df) < 30:
        st.error("❌ Not enough data points after processing.")
        st.stop()
    
    # State history
    state_history = df['State'].tail(history_length).tolist()
    state_history.reverse()
    
    # Calculate probabilities
    trans_probs = calculate_transition_probabilities(state_history)
    state_probs = calculate_state_probabilities(state_history)
    
    # Brier Score
    predictions_list = []
    actuals_list = []
    
    for i in range(history_length + 1, len(df)):
        hist = df['State'].iloc[max(0, i-history_length-1):i-1].tolist()
        hist.reverse()
        
        if len(hist) >= 2:
            UPTREND = 1
            prob_up = sum(1 for s in hist if s == UPTREND) / len(hist)
            predictions_list.append(prob_up)
            actual = 1 if df['State'].iloc[i] == 1 else 0
            actuals_list.append(actual)
    
    brier_score = calculate_brier_score(predictions_list[-brier_lookback:], actuals_list[-brier_lookback:])
    brier_text, brier_color = get_brier_classification(brier_score)
    
    # Current state
    current_state = df.iloc[-1]['State']
    current_state_name = df.iloc[-1]['State_Name']
    prob_uptrend = state_probs['uptrend']
    prob_downtrend = state_probs['downtrend']
    confidence = calculate_confidence(prob_uptrend, prob_downtrend)
    
    st.markdown("---")
    
    # Display Dashboard
    st.markdown("### 🎯 Markov Chain Analysis Dashboard")
    
    if beta_info:
        col1, col2, col3, col4, col5 = st.columns(5)
    else:
        col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        state_emoji = "🟢" if current_state == 1 else "🔴"
        st.metric("📊 Current State", current_state_name, delta=f"{state_emoji} Active")
    
    with col2:
        confidence_pct = confidence * 100
        st.metric("🎲 Confidence", f"{confidence_pct:.1f}%")
        if confidence_pct >= 40:
            st.caption("🟢 High Confidence")
        elif confidence_pct >= 20:
            st.caption("🟡 Moderate Confidence")
        else:
            st.caption("🔴 Low Confidence")
    
    with col3:
        price_change = df.iloc[-1]['Return'] * 100
        st.metric("💰 Last Close", f"₹{df.iloc[-1]['Close']:.2f}", delta=f"{price_change:.2f}%")
    
    with col4:
        st.metric("📈 Model Accuracy", brier_text)
        if brier_score:
            st.caption(f"Brier Score: {brier_score:.3f}")
    
    if beta_info:
        with col5:
            beta_val = beta_info['beta']
            beta_delta = "High Risk" if abs(beta_val) > 1.5 else "Moderate" if abs(beta_val) > 0.7 else "Low Risk"
            st.metric("📊 Beta vs Nifty", f"{beta_val:.2f}", delta=beta_delta)
    
    # State Probabilities
    st.markdown("---")
    st.markdown("#### 📊 State Probabilities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🟢 P(Uptrend)", f"{prob_uptrend*100:.1f}%")
        st.caption("Probability of being in uptrend")
    
    with col2:
        st.metric("🔴 P(Downtrend)", f"{prob_downtrend*100:.1f}%")
        st.caption("Probability of being in downtrend")
    
    with col3:
        atr_change = df.iloc[-1]['ATR_Normalized_Change']
        st.metric("📈 ATR Normalized Change", f"{atr_change:.2f}")
        st.caption("Current momentum strength")
    
    # Transition Matrix
    st.markdown("---")
    st.markdown("#### 🔄 Transition Probability Matrix")
    
    trans_matrix = pd.DataFrame({
        'To Uptrend': [trans_probs['up_to_up'] * 100, trans_probs['down_to_up'] * 100],
        'To Downtrend': [trans_probs['up_to_down'] * 100, trans_probs['down_to_down'] * 100]
    }, index=['From Uptrend', 'From Downtrend'])
    
    styled_matrix = trans_matrix.style.background_gradient(cmap='RdYlGn', axis=None)\
        .format("{:.1f}%")\
        .set_properties(**{'text-align': 'center'})
    
    st.dataframe(styled_matrix, use_container_width=True)
    
    # Charts
    st.markdown("---")
    st.markdown("### 📈 Technical Analysis Charts")
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.45, 0.20, 0.20, 0.15],
        subplot_titles=("Price with States & Bollinger Bands", "State Probabilities", "Brier Score Evolution", "Volume")
    )
    
    # Price chart
    for state_name in df["State_Name"].unique():
        state_df = df[df["State_Name"] == state_name]
        color = "green" if state_name == "Uptrend" else "red"
        
        fig.add_trace(
            go.Scatter(
                x=state_df["Date"], y=state_df["Close"],
                mode="lines", name=f"{state_name}",
                line=dict(color=color, width=2), showlegend=True
            ), row=1, col=1
        )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=df["Date"], y=df["BB_Upper"],
            name=f"BB Upper ({bb_std}σ)",
            line=dict(color="rgba(59, 130, 246, 0.4)", width=1, dash="dash"),
            showlegend=True
        ), row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df["Date"], y=df["BB_Lower"],
            name=f"BB Lower ({bb_std}σ)",
            line=dict(color="rgba(59, 130, 246, 0.4)", width=1, dash="dash"),
            fill='tonexty', fillcolor="rgba(59, 130, 246, 0.1)",
            showlegend=True
        ), row=1, col=1
    )
    
    # State probabilities
    df['Prob_Up'] = 0.0
    df['Prob_Down'] = 0.0
    
    for i in range(history_length, len(df)):
        hist = df['State'].iloc[max(0, i-history_length):i].tolist()
        if len(hist) > 0:
            prob_up = sum(1 for s in hist if s == 1) / len(hist)
            df.loc[i, 'Prob_Up'] = prob_up * 100
            df.loc[i, 'Prob_Down'] = (1 - prob_up) * 100
    
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["Prob_Up"], name="P(Uptrend)",
                   line=dict(color="green", width=2), showlegend=True),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["Prob_Down"], name="P(Downtrend)",
                   line=dict(color="red", width=2), showlegend=True),
        row=2, col=1
    )
    
    # Brier Score
    if len(predictions_list) > 0:
        brier_dates = df["Date"].iloc[-len(predictions_list):].tolist()
        brier_scores_list = []
        
        for i in range(len(predictions_list)):
            if i >= brier_lookback:
                score = calculate_brier_score(
                    predictions_list[max(0, i-brier_lookback):i],
                    actuals_list[max(0, i-brier_lookback):i]
                )
                brier_scores_list.append(score * 100 if score else None)
            else:
                brier_scores_list.append(None)
        
        fig.add_trace(
            go.Scatter(x=brier_dates, y=brier_scores_list, name="Brier Score",
                       line=dict(color="orange", width=2), showlegend=True),
            row=3, col=1
        )
        
        fig.add_hline(y=15, line_dash="dash", line_color="lime", 
                      annotation_text="Excellent (15%)", row=3, col=1)
        fig.add_hline(y=25, line_dash="dash", line_color="yellow", 
                      annotation_text="Random (25%)", row=3, col=1)
    
    # Volume
    colors = ['green' if row['Return'] >= 0 else 'red' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(x=df["Date"], y=df["Volume"], name="Volume",
               marker_color=colors, showlegend=False),
        row=4, col=1
    )
    
    fig.update_layout(
        height=1100, hovermode='x unified', showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(title_text="Date", row=4, col=1)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Probability (%)", row=2, col=1)
    fig.update_yaxes(title_text="Brier Score (%)", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # -------------------------------------------------
    # STATE STATISTICS
    # -------------------------------------------------
    
    st.markdown("---")
    st.markdown("### 📊 State Statistics & Analysis")
    
    state_stats = df.groupby("State_Name").agg({
        "ATR": "mean",
        "Volume": "mean",
        "State": "count",
        "Return": ["mean", "std"]
    }).round(4)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### State Characteristics")
        for state_name in ["Uptrend", "Downtrend"]:
            if state_name in state_stats.index:
                count = int(state_stats.loc[state_name, ("State", "count")])
                pct = (count / len(df)) * 100
                avg_atr = state_stats.loc[state_name, ("ATR", "mean")]
                avg_vol = state_stats.loc[state_name, ("Volume", "mean")]
                avg_ret = state_stats.loc[state_name, ("Return", "mean")] * 100
                std_ret = state_stats.loc[state_name, ("Return", "std")] * 100
                
                color = "🟢" if state_name == "Uptrend" else "🔴"
                
                st.metric(
                    f"{color} {state_name}",
                    f"{pct:.1f}% of time",
                    f"{count} periods"
                )
                st.caption(f"Avg Return: {avg_ret:.3f}% | Volatility: {std_ret:.3f}%")
                st.caption(f"Avg ATR: {avg_atr:.2f} | Avg Volume: {avg_vol:,.0f}")
    
    with col2:
        st.markdown("#### Current Market State")
        
        # Calculate state changes
        state_changes = (df["State"] != df["State"].shift()).sum()
        
        st.info(f"**Current State:** {current_state_name}")
        st.info(f"**State Changes:** {state_changes} in {len(df)} periods")
        st.info(f"**State Stability:** {((1 - state_changes/len(df)) * 100):.1f}%")
        
        # Calculate average state duration
        state_durations = []
        current_duration = 1
        for i in range(1, len(df)):
            if df.iloc[i]["State"] == df.iloc[i-1]["State"]:
                current_duration += 1
            else:
                state_durations.append(current_duration)
                current_duration = 1
        state_durations.append(current_duration)
        
        avg_duration = np.mean(state_durations) if state_durations else 0
        st.info(f"**Avg State Duration:** {avg_duration:.1f} periods")
        
        # Current state duration
        current_state_duration = 1
        for i in range(len(df)-2, -1, -1):
            if df.iloc[i]["State"] == df.iloc[-1]["State"]:
                current_state_duration += 1
            else:
                break
        st.info(f"**Current State Duration:** {current_state_duration} periods")
    
    # -------------------------------------------------
    # PREDICTION SUMMARY
    # -------------------------------------------------
    
    st.markdown("---")
    st.markdown("### 🎯 Prediction Summary")
    
    # Next period prediction
    if current_state == 1:  # Currently in uptrend
        prob_stay = trans_probs['up_to_up'] * 100
        prob_switch = trans_probs['up_to_down'] * 100
        next_likely = "Uptrend" if prob_stay > prob_switch else "Downtrend"
        next_prob = max(prob_stay, prob_switch)
    else:  # Currently in downtrend
        prob_stay = trans_probs['down_to_down'] * 100
        prob_switch = trans_probs['down_to_up'] * 100
        next_likely = "Downtrend" if prob_stay > prob_switch else "Uptrend"
        next_prob = max(prob_stay, prob_switch)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "🔮 Most Likely Next State",
            next_likely,
            f"{next_prob:.1f}% probability"
        )
    
    with col2:
        if prob_stay > 70:
            stability = "🟢 High Persistence"
        elif prob_stay > 50:
            stability = "🟡 Moderate Persistence"
        else:
            stability = "🔴 Low Persistence"
        
        st.metric(
            "📊 State Persistence",
            stability,
            f"{prob_stay:.1f}% stay in current"
        )
    
    with col3:
        if brier_score and brier_score < 0.20:
            reliability = "🟢 Reliable"
        elif brier_score and brier_score < 0.30:
            reliability = "🟡 Moderate"
        else:
            reliability = "🔴 Use with Caution"
        
        st.metric(
            "✅ Model Reliability",
            reliability,
            brier_text
        )
    
    # Trading signals
    st.markdown("---")
    st.markdown("### 🎯 Trading Signals")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if prob_uptrend > 0.7:
            signal = "🟢 STRONG BUY SIGNAL"
            explanation = f"High uptrend probability ({prob_uptrend*100:.1f}%)"
        elif prob_uptrend > 0.6:
            signal = "🟢 BUY SIGNAL"
            explanation = f"Moderate uptrend probability ({prob_uptrend*100:.1f}%)"
        elif prob_downtrend > 0.7:
            signal = "🔴 STRONG SELL SIGNAL"
            explanation = f"High downtrend probability ({prob_downtrend*100:.1f}%)"
        elif prob_downtrend > 0.6:
            signal = "🔴 SELL SIGNAL"
            explanation = f"Moderate downtrend probability ({prob_downtrend*100:.1f}%)"
        else:
            signal = "🟡 NEUTRAL"
            explanation = "No clear directional bias"
        
        st.info(f"**{signal}**\n\n{explanation}")
    
    with col2:
        bb_pos = df.iloc[-1]['BB_Position']
        
        if pd.notna(bb_pos):
            if bb_pos > 80:
                bb_signal = "🔴 Overbought (BB)"
                bb_desc = f"Price at {bb_pos:.1f}% of BB range"
            elif bb_pos < 20:
                bb_signal = "🟢 Oversold (BB)"
                bb_desc = f"Price at {bb_pos:.1f}% of BB range"
            else:
                bb_signal = "⚪ Normal Range (BB)"
                bb_desc = f"Price at {bb_pos:.1f}% of BB range"
        else:
            bb_signal = "⚪ No BB Signal"
            bb_desc = "Insufficient data"
        
        st.info(f"**{bb_signal}**\n\n{bb_desc}")
    
    with col3:
        if confidence_pct > 40:
            conf_signal = "🟢 HIGH CONFIDENCE"
            conf_desc = f"Strong probability separation ({confidence_pct:.1f}%)"
        elif confidence_pct > 20:
            conf_signal = "🟡 MODERATE CONFIDENCE"
            conf_desc = f"Moderate separation ({confidence_pct:.1f}%)"
        else:
            conf_signal = "🔴 LOW CONFIDENCE"
            conf_desc = f"Weak separation ({confidence_pct:.1f}%) - uncertain market"
        
        st.info(f"**{conf_signal}**\n\n{conf_desc}")
    
    # -------------------------------------------------
    # RISK METRICS
    # -------------------------------------------------
    
    st.markdown("---")
    st.markdown("### ⚠️ Risk Metrics")
    
    # Calculate risk metrics
    returns = df['Return'].dropna()
    
    # Sharpe Ratio
    risk_free_rate = 0.06
    excess_returns = returns - (risk_free_rate / 252)
    sharpe = np.sqrt(252) * (excess_returns.mean() / returns.std()) if returns.std() != 0 else 0
    
    # Max Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # Volatility
    volatility = returns.std() * np.sqrt(252) * 100
    
    # VaR
    var_95 = np.percentile(returns, 5) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📈 Sharpe Ratio", f"{sharpe:.3f}")
        st.caption("Risk-adjusted return")
    
    with col2:
        st.metric("📉 Max Drawdown", f"{max_drawdown:.2f}%")
        st.caption("Largest decline")
    
    with col3:
        st.metric("📊 Volatility", f"{volatility:.2f}%")
        st.caption("Annualized")
    
    with col4:
        st.metric("⚠️ VaR (95%)", f"{var_95:.2f}%")
        st.caption("Daily risk")
    
    # -------------------------------------------------
    # ADDITIONAL QUANTITATIVE METRICS
    # -------------------------------------------------
    
    st.markdown("---")
    st.markdown("### 📈 Advanced Quantitative Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
        sortino = np.sqrt(252) * (excess_returns.mean() / downside_std) if downside_std != 0 else 0
        st.metric("📊 Sortino Ratio", f"{sortino:.3f}")
        st.caption("Downside risk-adjusted")
    
    with col2:
        # Calmar Ratio
        annual_return = returns.mean() * 252 * 100
        calmar = abs(annual_return / max_drawdown) if max_drawdown != 0 else 0
        st.metric("📊 Calmar Ratio", f"{calmar:.3f}")
        st.caption("Return/Max Drawdown")
    
    with col3:
        # Win Rate
        positive_returns = len(returns[returns > 0])
        total_returns = len(returns)
        win_rate = (positive_returns / total_returns * 100) if total_returns > 0 else 0
        st.metric("🎯 Win Rate", f"{win_rate:.1f}%")
        st.caption(f"{positive_returns}/{total_returns} periods")
    
    with col4:
        # Profit Factor
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = gains / losses if losses != 0 else 0
        st.metric("💰 Profit Factor", f"{profit_factor:.2f}")
        st.caption("Gains/Losses ratio")
    
    # Second row of metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Skewness
        skewness = returns.skew()
        st.metric("📐 Skewness", f"{skewness:.3f}")
        if skewness > 0:
            st.caption("🟢 Positive tail")
        elif skewness < 0:
            st.caption("🔴 Negative tail")
        else:
            st.caption("⚪ Symmetric")
    
    with col2:
        # Kurtosis
        kurtosis = returns.kurtosis()
        st.metric("📊 Kurtosis", f"{kurtosis:.3f}")
        if kurtosis > 3:
            st.caption("🔴 Fat tails (risky)")
        elif kurtosis < 3:
            st.caption("🟢 Thin tails")
        else:
            st.caption("⚪ Normal dist.")
    
    with col3:
        # Average Gain/Loss
        avg_gain = returns[returns > 0].mean() * 100 if len(returns[returns > 0]) > 0 else 0
        avg_loss = returns[returns < 0].mean() * 100 if len(returns[returns < 0]) > 0 else 0
        st.metric("📈 Avg Gain", f"{avg_gain:.3f}%")
        st.caption(f"Avg Loss: {avg_loss:.3f}%")
    
    with col4:
        # Gain/Loss Ratio
        gain_loss_ratio = abs(avg_gain / avg_loss) if avg_loss != 0 else 0
        st.metric("⚖️ Gain/Loss Ratio", f"{gain_loss_ratio:.2f}")
        if gain_loss_ratio > 1:
            st.caption("🟢 Gains > Losses")
        else:
            st.caption("🔴 Losses > Gains")
    
    # -------------------------------------------------
    # STOCK PROFILE & SECTOR INFORMATION
    # -------------------------------------------------
    
    st.markdown("---")
    st.markdown("### 🏢 Stock Profile & Sector Analysis")
    
    # Fetch stock info
    try:
        ticker = yf.Ticker(symbol + ".NS")
        info = ticker.info
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Company Information")
            
            company_name = info.get('longName', info.get('shortName', symbol))
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            market_cap = info.get('marketCap', 0)
            
            st.info(f"**Company:** {company_name}")
            st.info(f"**Sector:** {sector}")
            st.info(f"**Industry:** {industry}")
            
            if market_cap > 0:
                market_cap_cr = market_cap / 10000000  # Convert to Crores
                st.info(f"**Market Cap:** ₹{market_cap_cr:,.2f} Cr")
            
            # Additional info
            country = info.get('country', 'N/A')
            website = info.get('website', 'N/A')
            employees = info.get('fullTimeEmployees', 'N/A')
            
            st.caption(f"**Country:** {country}")
            if employees != 'N/A':
                st.caption(f"**Employees:** {employees:,}")
            if website != 'N/A':
                st.caption(f"**Website:** {website}")
        
        with col2:
            st.markdown("#### 💹 Key Financial Metrics")
            
            # Price metrics
            current_price = info.get('currentPrice', df.iloc[-1]['Close'])
            previous_close = info.get('previousClose', 0)
            fifty_two_week_high = info.get('fiftyTwoWeekHigh', 0)
            fifty_two_week_low = info.get('fiftyTwoWeekLow', 0)
            
            st.metric("💰 Current Price", f"₹{current_price:.2f}")
            
            if previous_close > 0:
                day_change = ((current_price - previous_close) / previous_close) * 100
                st.metric("📊 Day Change", f"{day_change:.2f}%")
            
            if fifty_two_week_high > 0 and fifty_two_week_low > 0:
                st.caption(f"**52W High:** ₹{fifty_two_week_high:.2f}")
                st.caption(f"**52W Low:** ₹{fifty_two_week_low:.2f}")
                
                # Calculate position in 52-week range
                range_position = ((current_price - fifty_two_week_low) / 
                                (fifty_two_week_high - fifty_two_week_low) * 100)
                st.caption(f"**52W Range Position:** {range_position:.1f}%")
            
            # Valuation metrics
            pe_ratio = info.get('trailingPE', None)
            pb_ratio = info.get('priceToBook', None)
            dividend_yield = info.get('dividendYield', None)
            
            if pe_ratio:
                st.caption(f"**P/E Ratio:** {pe_ratio:.2f}")
            if pb_ratio:
                st.caption(f"**P/B Ratio:** {pb_ratio:.2f}")
            if dividend_yield:
                st.caption(f"**Dividend Yield:** {dividend_yield*100:.2f}%")
        
        # Sector comparison
        st.markdown("---")
        st.markdown("#### 🎯 Sector Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if sector != 'N/A':
                st.info(f"**Primary Sector:** {sector}")
                
                # Provide sector context
                sector_description = {
                    'Technology': '💻 IT services, software, and tech products',
                    'Financial Services': '🏦 Banking, insurance, and financial products',
                    'Healthcare': '🏥 Pharmaceuticals, hospitals, and medical devices',
                    'Consumer Cyclical': '🛍️ Retail, automobiles, and discretionary goods',
                    'Consumer Defensive': '🛒 FMCG, food, and essential products',
                    'Energy': '⚡ Oil, gas, and renewable energy',
                    'Industrials': '🏭 Manufacturing, construction, and engineering',
                    'Basic Materials': '⚒️ Metals, mining, and chemicals',
                    'Communication Services': '📡 Telecom and media',
                    'Real Estate': '🏗️ Property development and REITs',
                    'Utilities': '💡 Power, water, and utilities'
                }
                
                st.caption(sector_description.get(sector, 'Sector information'))
        
        with col2:
            # Beta interpretation
            if beta_info:
                beta_val = beta_info['beta']
                st.info(f"**Beta:** {beta_val:.2f}")
                
                if beta_val > 1.5:
                    st.caption("🔴 Highly volatile vs market")
                elif beta_val > 1.0:
                    st.caption("🟠 More volatile than market")
                elif beta_val > 0.5:
                    st.caption("🟡 Moderately correlated")
                elif beta_val > 0:
                    st.caption("🟢 Less volatile than market")
                else:
                    st.caption("🔵 Inverse market correlation")
        
        with col3:
            # Volume analysis
            avg_volume = info.get('averageVolume', 0)
            current_volume = df.iloc[-1]['Volume']
            
            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                st.info(f"**Volume Ratio:** {volume_ratio:.2f}x")
                
                if volume_ratio > 2:
                    st.caption("🔴 Unusually high volume")
                elif volume_ratio > 1.5:
                    st.caption("🟠 Above average volume")
                elif volume_ratio > 0.5:
                    st.caption("🟢 Normal volume")
                else:
                    st.caption("🔵 Below average volume")
        
        # Business summary
        if 'longBusinessSummary' in info and info['longBusinessSummary']:
            with st.expander("📖 Business Summary"):
                st.write(info['longBusinessSummary'])
    
    except Exception as e:
        st.warning(f"⚠️ Could not fetch detailed stock information: {str(e)}")
        st.info("Continuing with available data...")
    
    # -------------------------------------------------
    # KEY INSIGHTS
    # -------------------------------------------------
    
    st.markdown("---")
    st.markdown("### 💡 Key Insights")
    
    insights = []
    
    # Insight 1: State prediction
    if next_prob > 70:
        insights.append(f"✅ **Strong prediction**: Next state likely to be {next_likely} with {next_prob:.1f}% probability")
    else:
        insights.append(f"⚠️ **Uncertain prediction**: Next state could be {next_likely} ({next_prob:.1f}%), but confidence is moderate")
    
    # Insight 2: Model accuracy
    if brier_score and brier_score < 0.15:
        insights.append(f"✅ **Excellent model accuracy**: Brier score of {brier_score:.3f} indicates reliable predictions")
    elif brier_score and brier_score > 0.30:
        insights.append(f"⚠️ **Poor model accuracy**: Brier score of {brier_score:.3f} - use predictions with caution")
    
    # Insight 3: State stability
    if prob_stay > 75:
        insights.append(f"✅ **Stable regime**: Current {current_state_name} state shows high persistence ({prob_stay:.1f}%)")
    elif prob_stay < 50:
        insights.append(f"⚠️ **Unstable regime**: Current state may change soon (only {prob_stay:.1f}% persistence)")
    
    # Insight 4: Confidence
    if confidence_pct > 40:
        insights.append(f"✅ **Clear market direction**: High confidence ({confidence_pct:.1f}%) indicates strong trend")
    else:
        insights.append(f"⚠️ **Unclear direction**: Low confidence ({confidence_pct:.1f}%) suggests ranging market")
    
    # Insight 5: Beta
    if beta_info:
        if beta_info['beta'] > 1.2:
            insights.append(f"⚠️ **High market sensitivity**: Beta of {beta_info['beta']:.2f} means stock amplifies market movements")
        elif beta_info['beta'] < 0.8:
            insights.append(f"✅ **Lower risk**: Beta of {beta_info['beta']:.2f} indicates less volatile than market")
    
    # Insight 6: Risk metrics
    if sharpe > 1.5:
        insights.append(f"✅ **Excellent risk-adjusted returns**: Sharpe ratio of {sharpe:.2f} indicates strong performance")
    elif sharpe < 0.5:
        insights.append(f"⚠️ **Poor risk-adjusted returns**: Sharpe ratio of {sharpe:.2f} suggests underperformance")
    
    # Insight 7: Win rate
    if win_rate > 60:
        insights.append(f"✅ **High win rate**: {win_rate:.1f}% of periods showed positive returns")
    elif win_rate < 40:
        insights.append(f"⚠️ **Low win rate**: Only {win_rate:.1f}% of periods showed positive returns")
    
    for insight in insights:
        st.markdown(insight)
    
    # -------------------------------------------------
    # METHODOLOGY COMPARISON
    # -------------------------------------------------
    
    st.markdown("---")
    st.markdown("### 📚 Methodology: Markov Chain vs HMM")
    
    with st.expander("🔍 Why Markov Chain is Better Than HMM for This Use Case"):
        st.markdown("""
        **Markov Chain Advantages:**
        1. **Transparency**: States are clearly defined by observable metrics (ATR-normalized changes)
        2. **No Overfitting**: No complex parameter optimization that can memorize data
        3. **Interpretability**: Transition probabilities are easy to understand and validate
        4. **Stability**: No convergence issues or numerical instabilities
        5. **Validation**: Brier score directly measures prediction accuracy
        
        **HMM Disadvantages:**
        1. **Black Box**: Hidden states are not directly observable
        2. **Overfitting**: Complex covariance matrices can memorize training data
        3. **Convergence**: Often fails to converge or gets stuck in local minima
        4. **Parameter Sensitivity**: Results vary wildly with initialization
        5. **False Confidence**: Can show 100% confidence even when overfitting
        
        **When to Use Each:**
        - **Markov Chain**: When you can define states based on observable features (price, volume, ATR)
        - **HMM**: When underlying states are truly hidden and not directly measurable
        
        **This Implementation:**
        - Uses ATR-normalized price changes to define states
        - Validates predictions with Brier score
        - Shows honest confidence based on probability separation
        - Provides transition matrix for regime analysis
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Disclaimer:</strong> This tool is for educational and research purposes only. 
        Not financial advice. Past performance does not guarantee future results.</p>
        <p><small>Markov Chain Predictor with Brier Score Validation & Beta Analysis | Powered by Streamlit & yfinance</small></p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("👈 Configure parameters in the sidebar and click **Run Analysis** to start!")
    
    st.markdown("### 🚀 Quick Start Guide")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **1️⃣ Select Symbol**
        - Choose from NSE stocks
        - Popular: RELIANCE, TCS, INFY
        """)
    
    with col2:
        st.markdown("""
        **2️⃣ Set Parameters**
        - Date range (default: 1 year)
        - Timeframe (1d, 1h, etc.)
        - Markov chain settings
        """)
    
    with col3:
        st.markdown("""
        **3️⃣ Run Analysis**
        - Click "Run Analysis"
        - View predictions & charts
        - Get trading signals
        """)
    
    st.markdown("---")
    st.markdown("### 📊 What You'll Get")
    
    st.markdown("""
    - **State Identification**: Uptrend/Downtrend classification using ATR
    - **Probability Analysis**: Transition probabilities between states
    - **Model Accuracy**: Brier score validation (lower is better)
    - **Beta Analysis**: Stock volatility vs Nifty 50
    - **Trading Signals**: Data-driven buy/sell recommendations
    - **Technical Charts**: Price, Bollinger Bands, probabilities
    """)
    
    st.markdown("---")
    st.markdown("### ⚡ Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **✅ Advantages**
        - Pure Markov Chain (no HMM complexity)
        - Transparent state definitions
        - Validated with Brier score
        - No overfitting issues
        - Easy to interpret
        """)
    
    with col2:
        st.warning("""
        **⚠️ Important Notes**
        - Not financial advice
        - For educational purposes
        - Past ≠ Future performance
        - Use with risk management
        - Combine with other analysis
        """)