import yfinance as yf
import numpy as np
import pandas as pd
from google import genai
import os
import re
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Secure API Key Handling
def get_api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass
    return os.getenv("GEMINI_API_KEY")

api_key = get_api_key()
client = genai.Client(api_key=api_key) if api_key else None

# Global Benchmark Mapping for auto-detection
BENCHMARK_MAP = {
    ".AT": "GD.AT",      # Greece
    ".L": "^FTSE",       # UK
    ".DE": "^GDAXI",     # Germany
    ".PA": "^FCHI",      # France
    ".AS": "^AEX",       # Amsterdam
    ".MI": "FTSEMIB.MI", # Italy
    ".MC": "^IBEX",      # Spain
    ".SW": "SMI.SW",     # Swiss
    ".TO": "^GSPTSE"     # Toronto
}

def sanitize_ticker(ticker):
    """Strips malicious or invalid characters from inputs."""
    return re.sub(r'[^A-Z0-9.\-=^]', '', ticker.strip().upper())

def get_default_benchmark(ticker):
    for suffix, bench in BENCHMARK_MAP.items():
        if ticker.endswith(suffix): 
            return bench
    return "^GSPC" # Default to S&P 500

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(tickers, period="1y", interval="1d"):
    """Cached data fetcher to prevent rate limiting and speed up the UI."""
    data = yf.download(tickers, period=period, interval=interval, progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data.xs('Close', level=0, axis=1)
        except KeyError:
            data = data.droplevel(0, axis=1)
    return data

def get_market_analysis(ticker, custom_benchmark=""):
    """Audited quantitative risk engine for a single asset."""
    ticker = sanitize_ticker(ticker)
    benchmark = sanitize_ticker(custom_benchmark) if custom_benchmark else get_default_benchmark(ticker)

    # Data Acquisition
    data = fetch_market_data([ticker, benchmark, "^TNX"], period="1y", interval="1d")
    
    if ticker not in data.columns or data[ticker].dropna().empty:
        raise ValueError(f"No data for '{ticker}'. Check the symbol or suffix.")

    # Cross-Market Alignment
    data.ffill(inplace=True)
    data.dropna(subset=[ticker], inplace=True) 

    # Minimum data point check (require at least 3 months / ~60 trading days)
    if len(data) < 60:
        raise ValueError(f"Insufficient data points ({len(data)}) for {ticker}. Need at least 60 days for valid metrics.")

    # Statistical Calculations
    returns = np.log(data / data.shift(1)).dropna()
    daily_rf = (data["^TNX"] / 100 / 252).reindex(returns.index).ffill()
    
    # Static Metrics
    vol = returns[ticker].std() * np.sqrt(252) * 100
    covariance = returns[[ticker, benchmark]].cov().iloc[0, 1]
    market_variance = returns[benchmark].var()
    beta = covariance / market_variance
    excess_return = returns[ticker] - daily_rf
    sharpe = (excess_return.mean() / excess_return.std()) * np.sqrt(252)
    
    rolling_vol = returns[ticker].rolling(window=21).std() * np.sqrt(252) * 100
    cumulative_returns = returns[ticker].cumsum() * 100
    
    return {
        "ticker": ticker,
        "benchmark": benchmark,
        "volatility": vol,
        "beta": beta,
        "sharpe": sharpe,
        "rolling_vol": rolling_vol.dropna(),
        "cumulative_returns": cumulative_returns
    }

def get_comparative_analysis(t1, t2, b1="", b2=""):
    """Synchronized multi-asset matrix calculation."""
    t1, t2 = sanitize_ticker(t1), sanitize_ticker(t2)
    bench1 = sanitize_ticker(b1) if b1 else get_default_benchmark(t1)
    bench2 = sanitize_ticker(b2) if b2 else get_default_benchmark(t2)
    
    tickers_to_fetch = list(set([t1, t2, bench1, bench2, "^TNX"]))
    data = fetch_market_data(tickers_to_fetch, period="1y", interval="1d")
            
    data.ffill(inplace=True)
    data.dropna(subset=[t1, t2], inplace=True)
    
    if len(data) < 60:
        raise ValueError("Insufficient overlapping data points for comparison.")
    
    returns = np.log(data / data.shift(1)).dropna()
    daily_rf = (data["^TNX"] / 100 / 252).reindex(returns.index).ffill()
    
    def calc_metrics(ticker, benchmark):
        vol = returns[ticker].std() * np.sqrt(252) * 100
        cov = returns[[ticker, benchmark]].cov().iloc[0, 1]
        m_var = returns[benchmark].var()
        beta = cov / m_var
        
        excess = returns[ticker] - daily_rf
        sharpe = (excess.mean() / excess.std()) * np.sqrt(252)
        
        rolling_vol = returns[ticker].rolling(window=21).std() * np.sqrt(252) * 100
        cum_ret = returns[ticker].cumsum() * 100
        
        return {
            "ticker": ticker,
            "benchmark": benchmark,
            "volatility": vol,
            "beta": beta,
            "sharpe": sharpe,
            "rolling_vol": rolling_vol.dropna(),
            "cumulative_returns": cum_ret
        }
        
    r1 = calc_metrics(t1, bench1)
    r2 = calc_metrics(t2, bench2)
    
    aligned_vol_df = pd.DataFrame({
        r1['ticker']: r1['rolling_vol'], 
        r2['ticker']: r2['rolling_vol']
    }).dropna()
    
    aligned_cum_df = pd.DataFrame({
        r1['ticker']: r1['cumulative_returns'], 
        r2['ticker']: r2['cumulative_returns']
    })
    
    return r1, r2, aligned_vol_df, aligned_cum_df

def get_weekly_movers():
    """Calculates 7-day market heat segmented by asset class using global data."""
    # Expanded global ticker lists for diverse volatility profiles
    asset_classes = {
        "EQUITIES": {
            "tickers": [
                "NVDA", "TSLA", "COIN", "MSTR",  # High Volatility US
                "JNJ", "PG", "KO", "PEP",       # Low Volatility US
                "ALPHA.AT", "ASML.AS", "SAP.DE", "MC.PA" # Global/European
            ], 
            "annual_factor": 1638 # ~6.5h/day * 252 days
        },
        "CRYPTO": {
            "tickers": [
                "BTC-USD", "ETH-USD", "SOL-USD", # Majors
                "DOGE-USD", "SHIB-USD", "PEPE-USD", # High Volatility Alts
                "ADA-USD", "XRP-USD", "LINK-USD" # Mid-Caps
            ], 
            "annual_factor": 8760 # 24/7 * 365 days
        },
        "MACRO": {
            "tickers": [
                "GLD", "SLV", "USO",            # Commodities (Gold, Silver, Oil)
                "EURUSD=X", "USDJPY=X", "GBPUSD=X", # Major Currencies
                "^TNX", "^VIX", "BITO"          # Yields, Volatility Index, Bitcoin ETF
            ], 
            "annual_factor": 5796 # ~23h/day * 252 days
        }
    }
    
    # Flatten all tickers for a single vectorized fetch
    all_tickers = [t for ac in asset_classes.values() for t in ac["tickers"]]
    data = fetch_market_data(all_tickers, period="7d", interval="1h")
    
    results = {}
    for category, config in asset_classes.items():
        cat_results = []
        for col in config["tickers"]:
            if col in data.columns:
                # Calculate log returns for hourly data
                series = data[col].dropna()
                if not series.empty and len(series) > 10:
                    ret = np.log(series / series.shift(1)).dropna()
                    # Annualize volatility based on asset-class specific trading hours
                    vol = ret.std() * np.sqrt(config["annual_factor"]) * 100
                    cat_results.append({
                        "ASSET": col, 
                        "HEAT (ANN.% VOL)": round(vol, 2)
                    })
        
        # Create DataFrame and sort to find the most/least volatile
        if cat_results:
            df = pd.DataFrame(cat_results).sort_values(by="HEAT (ANN.% VOL)", ascending=False)
            results[category] = df
        
    return results


def get_gemini_report(r1, r2=None):
    if not client: return "AI SERVICE OFFLINE. CHECK API KEY SECRETS."
    
    prompt = f"RISK REPORT: {r1['ticker']} vs benchmark {r1['benchmark']} (Total Vol: {r1['volatility']:.2f}%, Systemic Beta: {r1['beta']:.2f}, Ann. Sharpe: {r1['sharpe']:.2f})."
    if r2: 
        prompt += f" COMPARE WITH {r2['ticker']} vs benchmark {r2['benchmark']} (Total Vol: {r2['volatility']:.2f}%, Systemic Beta: {r2['beta']:.2f}, Ann. Sharpe: {r2['sharpe']:.2f}). CRITICAL NOTE: If benchmarks differ, explicitly factor cross-benchmark limitations into your systemic risk comparison."
        
    prompt += " WRITE 3 FORMAL SENTENCES ANALYZING SYSTEMIC SENSITIVITY AND RISK-ADJUSTED EFFICIENCY. DO NOT USE EMOJIS. MAINTAIN AN ACADEMIC FINANCIAL TONE."
    
    try:
        return client.models.generate_content(model="gemini-2.0-flash", contents=prompt).text
    except Exception as e: 
        return f"AI DIAGNOSTIC EXCEPTION: {str(e)}"