import yfinance as yf
import datetime
from typing import Dict, Any, List

def fetch_ticker_news(ticker_symbol: str, max_articles: int = 5) -> List[Dict[str, Any]]:
    """
    Fetches the latest news articles for a given ticker symbol using yfinance.
    Formats the output for the Financial Analyst agent.
    """
    formatted_news = []
    try:
        ticker = yf.Ticker(ticker_symbol)
        raw_news = ticker.news
        
        if not raw_news:
            return []
            
        for article in raw_news[:max_articles]:
            content = article.get("content", {}) if "content" in article else article
            
            title = content.get("title", "No Title")
            
            provider = content.get("provider", {})
            if isinstance(provider, dict):
                publisher = provider.get("displayName", "Unknown Publisher")
            else:
                publisher = content.get("publisher", "Unknown Publisher")
                
            link = ""
            if "clickThroughUrl" in content and isinstance(content["clickThroughUrl"], dict):
                link = content["clickThroughUrl"].get("url", "")
            elif "canonicalUrl" in content and isinstance(content["canonicalUrl"], dict):
                link = content["canonicalUrl"].get("url", "")
            else:
                link = content.get("link", "")
                
            publish_date_str = "Unknown Date"
            if "pubDate" in content:
                publish_date_str = content.get("pubDate", "Unknown Date")
            elif "providerPublishTime" in content:
                publish_time_stamp = content.get("providerPublishTime")
                if publish_time_stamp:
                    try:
                        publish_date = datetime.datetime.fromtimestamp(publish_time_stamp)
                        publish_date_str = publish_date.strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        pass
            
            formatted_news.append({
                "title": title,
                "publisher": publisher,
                "link": link,
                "publish_time": publish_date_str
            })
            
    except Exception as e:
        # Log error in returned list so agent is aware of failure
        formatted_news.append({
            "title": f"Error fetching news for {ticker_symbol}",
            "publisher": "System Error",
            "link": "",
            "publish_time": str(e)
        })
        
    return formatted_news

def get_portfolio_news(tickers: List[str], max_articles_per_ticker: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """Fetches and maps latest news articles to each ticker in the portfolio."""
    portfolio_news = {}
    for ticker in tickers:
        portfolio_news[ticker] = fetch_ticker_news(ticker, max_articles_per_ticker)
    return portfolio_news
