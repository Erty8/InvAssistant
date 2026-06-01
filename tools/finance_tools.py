import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from typing import Dict, Any, List

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI) for a given series of prices.
    Uses Wilder's smoothed moving average method.
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).copy()
    loss = (-delta.where(delta < 0, 0)).copy()

    # Exponential moving average (Wilder's smoothing uses alpha = 1/period)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_stock_metrics(ticker_symbol: str) -> Dict[str, Any]:
    """
    Fetches real-time and historical price data for a ticker using yfinance.
    Calculates key metrics: Daily % change, volume spikes, RSI, and SMA crossovers.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Fetch daily data for the last 90 calendar days to compute technical indicators
        # (needs at least ~50 trading days for 50-day SMA)
        hist = ticker.history(period="6mo")
        
        if hist.empty:
            return {
                "ticker": ticker_symbol,
                "error": "No historical data found. Ticker may be invalid or delisted."
            }

        # Last trading day info
        latest_data = hist.iloc[-1]
        prev_data = hist.iloc[-2] if len(hist) > 1 else latest_data
        
        current_price = latest_data["Close"]
        prev_close = prev_data["Close"]
        
        # Calculations
        daily_change_pct = ((current_price - prev_close) / prev_close) * 100 if prev_close else 0.0
        
        # Volume analysis
        current_volume = latest_data["Volume"]
        # Calculate 20-day average volume (excluding the current day to see standard base volume)
        avg_volume_20d = hist["Volume"].iloc[-21:-1].mean() if len(hist) > 21 else hist["Volume"].mean()
        volume_spike_ratio = (current_volume / avg_volume_20d) if avg_volume_20d > 0 else 1.0

        # Technical indicators
        # 1. RSI (14 days)
        rsi_series = calculate_rsi(hist["Close"], period=14)
        latest_rsi = rsi_series.iloc[-1] if not rsi_series.empty else None

        # 2. SMA/EMA (20-day and 50-day)
        sma20 = hist["Close"].rolling(window=20).mean()
        sma50 = hist["Close"].rolling(window=50).mean()
        
        latest_sma20 = sma20.iloc[-1] if len(sma20) >= 20 else None
        latest_sma50 = sma50.iloc[-1] if len(sma50) >= 50 else None
        
        prev_sma20 = sma20.iloc[-2] if len(sma20) >= 21 else None
        prev_sma50 = sma50.iloc[-2] if len(sma50) >= 51 else None
        
        # Crossover state
        crossover_signal = "Neutral"
        if latest_sma20 and latest_sma50 and prev_sma20 and prev_sma50:
            if latest_sma20 > latest_sma50 and prev_sma20 <= prev_sma50:
                crossover_signal = "Bullish Crossover (Golden Cross / 20-day crossed above 50-day)"
            elif latest_sma20 < latest_sma50 and prev_sma20 >= prev_sma50:
                crossover_signal = "Bearish Crossover (Death Cross / 20-day crossed below 50-day)"
            elif latest_sma20 > latest_sma50:
                crossover_signal = "Bullish (20-day is above 50-day)"
            else:
                crossover_signal = "Bearish (20-day is below 50-day)"

        # Extra metadata: High, Low, Open, 52-week High/Low
        info = ticker.info
        name = info.get("longName", ticker_symbol)
        sector = info.get("sector", "Unknown Sector")
        industry = info.get("industry", "Unknown Industry")
        fifty_two_week_high = info.get("fiftyTwoWeekHigh", None)
        fifty_two_week_low = info.get("fiftyTwoWeekLow", None)

        return {
            "ticker": ticker_symbol,
            "company_name": name,
            "sector": sector,
            "industry": industry,
            "current_price": float(current_price),
            "previous_close": float(prev_close),
            "daily_change_percent": float(daily_change_pct),
            "volume": int(current_volume),
            "average_volume_20d": float(avg_volume_20d),
            "volume_spike_ratio": float(volume_spike_ratio),
            "rsi_14": float(latest_rsi) if latest_rsi is not None and not np.isnan(latest_rsi) else None,
            "sma_20": float(latest_sma20) if latest_sma20 is not None and not np.isnan(latest_sma20) else None,
            "sma_50": float(latest_sma50) if latest_sma50 is not None and not np.isnan(latest_sma50) else None,
            "sma_crossover_status": crossover_signal,
            "fifty_two_week_high": fifty_two_week_high,
            "fifty_two_week_low": fifty_two_week_low,
            "success": True
        }
    except Exception as e:
        return {
            "ticker": ticker_symbol,
            "success": False,
            "error": f"Failed to fetch finance metrics: {str(e)}"
        }

def get_portfolio_metrics(tickers: List[str]) -> List[Dict[str, Any]]:
    """Fetches metrics for a list of tickers and returns a list of results."""
    results = []
    for ticker in tickers:
        results.append(fetch_stock_metrics(ticker))
    return results
