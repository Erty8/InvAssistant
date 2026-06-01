import sys
import argparse
import datetime
import json
from config import Config
from market_calendar import is_trading_day, get_current_ny_time
from tools.finance_tools import get_portfolio_metrics
from tools.news_tools import get_portfolio_news
from tools.email_tool import send_portfolio_email, save_report_locally
from agents import FinancialDataAnalystAgent, PortfolioManagerAgent

def run_portfolio_pipeline(force: bool = False) -> bool:
    """
    Executes the main portfolio monitoring pipeline:
    1. Validates configuration.
    2. Checks trading calendar (unless forced).
    3. Fetches quantitative data & news.
    4. Invokes Financial Analyst agent for structured aggregation.
    5. Invokes Portfolio Manager agent for strategist summary.
    6. Dispatches HTML report via Email (SMTP) and saves local copy.
    """
    print("=" * 60)
    print(f"Starting Portfolio Pipeline Run at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. Validate Config
    errors, warnings = Config.validate()
    for warning in warnings:
        print(f"[Config Warning] {warning}")
    if errors:
        for error in errors:
            print(f"[Config Error] {error}")
        print("\nPipeline execution halted due to configuration errors.")
        return False

    # 2. Check Calendar
    ny_time = get_current_ny_time()
    today_date = ny_time.date()
    
    print(f"Current NY Time: {ny_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    if not force:
        trading_day = is_trading_day(today_date)
        if not trading_day:
            print(f"[Calendar Bypass] Today ({today_date}) is a weekend or NYSE holiday. Skipping scheduled run.")
            return True
        print(f"[Calendar check] Today ({today_date}) is a valid US trading day. Proceeding...")
    else:
        print(f"[Calendar Bypass] Force execution flag is active. Ignoring trading day check.")

    # 3. Fetch Stock Data
    tickers = Config.PORTFOLIO_TICKERS
    print(f"\n[Data Gathering] Fetching quantitative metrics for: {', '.join(tickers)}")
    metrics_list = get_portfolio_metrics(tickers)
    
    # 4. Fetch News
    print(f"[Data Gathering] Scraping latest stock headlines...")
    news_dict = get_portfolio_news(tickers)
    
    # Check if we retrieved data
    valid_metrics = [m for m in metrics_list if m.get("success", False)]
    if not valid_metrics:
        print("[Error] No stock data could be retrieved for the specified tickers. Pipeline aborted.")
        return False

    # 5. Run Analyst Agent
    print(f"\n[Agent Execution] Running Financial Data Analyst (Quantitative Expert)...")
    analyst = FinancialDataAnalystAgent()
    try:
        analyst_output_json = analyst.run(valid_metrics, news_dict)
        # Verify output is valid JSON
        try:
            parsed_json = json.loads(analyst_output_json)
            print(f"[Analyst Output] Successfully generated structured analytical data.")
        except json.JSONDecodeError:
            print("[Warning] Analyst output is not strictly valid JSON. Feeding raw output to Portfolio Manager.")
            parsed_json = None
    except Exception as e:
        print(f"[Error] Financial Data Analyst Agent execution failed: {e}")
        return False

    # 6. Run Portfolio Manager Agent
    print(f"\n[Agent Execution] Running Portfolio Manager & Strategist (The Communicator)...")
    manager = PortfolioManagerAgent()
    try:
        html_report = manager.run(analyst_output_json)
        print(f"[Portfolio Manager Output] Generated newsletter HTML report.")
    except Exception as e:
        print(f"[Error] Portfolio Manager Agent execution failed: {e}")
        
        # Fallback basic HTML generation if LLM fails
        print("[Fallback] Creating basic fallback report with raw metrics...")
        html_report = generate_fallback_html(valid_metrics, today_date)

    # 7. Dispatch Report via Email / Local Copy
    date_str = today_date.strftime("%m/%d/%Y")
    subject = f"Daily Portfolio Insights - {date_str} - US Market Open Update"
    
    email_sent = send_portfolio_email(html_report, subject)
    
    if not email_sent:
        print("\nReport could not be sent via email (check configuration).")
    else:
        print("\nReport sent successfully to recipient.")

    print("\n" + "=" * 60)
    print("Pipeline execution completed successfully.")
    print("=" * 60)
    return True

def generate_fallback_html(metrics: list, date_obj: datetime.date) -> str:
    """Generates a simple table-based HTML report if the LLM agents fail to complete."""
    rows = ""
    for stock in metrics:
        change = stock.get('daily_change_percent', 0.0)
        color = "green" if change >= 0 else "red"
        sign = "+" if change >= 0 else ""
        
        rows += f"""
        <tr>
            <td style="padding: 10px; border-bottom: 1px solid #ddd;"><b>{stock.get('ticker')}</b> ({stock.get('company_name')})</td>
            <td style="padding: 10px; border-bottom: 1px solid #ddd;">${stock.get('current_price', 0.0):.2f}</td>
            <td style="padding: 10px; border-bottom: 1px solid #ddd; color: {color};"><b>{sign}{change:.2f}%</b></td>
            <td style="padding: 10px; border-bottom: 1px solid #ddd;">{stock.get('rsi_14', 'N/A')}</td>
            <td style="padding: 10px; border-bottom: 1px solid #ddd;">{stock.get('sma_crossover_status', 'N/A')}</td>
        </tr>
        """
        
    fallback_template = f"""
    <html>
    <head>
        <title>Portfolio Insights Fallback Report</title>
    </head>
    <body style="font-family: Arial, sans-serif; background-color: #f9f9f9; padding: 20px; color: #333;">
        <div style="max-width: 600px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; border: 1px solid #eee;">
            <h2>Daily Portfolio Insights (Fallback Report)</h2>
            <p>Date: {date_obj.strftime('%Y-%m-%d')}</p>
            <p style="color: #666; font-style: italic;">Note: This fallback report was automatically generated because the LLM Agent processing encountered a runtime error.</p>
            <table style="width: 100%; border-collapse: collapse; text-align: left;">
                <thead>
                    <tr style="background-color: #f2f2f2;">
                        <th style="padding: 10px; border-bottom: 2px solid #ddd;">Ticker</th>
                        <th style="padding: 10px; border-bottom: 2px solid #ddd;">Price</th>
                        <th style="padding: 10px; border-bottom: 2px solid #ddd;">Change</th>
                        <th style="padding: 10px; border-bottom: 2px solid #ddd;">RSI(14)</th>
                        <th style="padding: 10px; border-bottom: 2px solid #ddd;">Crossover Signal</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    return fallback_template

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portfolio Monitoring AI Agent System")
    parser.add_argument(
        "--run-now", 
        action="store_true", 
        help="Execute the pipeline immediately, ignoring trading calendar constraints"
    )
    args = parser.parse_args()
    
    # Run the pipeline
    success = run_portfolio_pipeline(force=args.run_now)
    sys.exit(0 if success else 1)
