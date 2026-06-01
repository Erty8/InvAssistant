import json
import datetime
from openai import OpenAI
from config import Config

def get_client() -> OpenAI:
    """Returns an initialized OpenAI client with custom base URL if configured."""
    api_key = Config.OPENAI_API_KEY
    base_url = Config.OPENAI_API_BASE
    
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)

class FinancialDataAnalystAgent:
    """
    Persona: A data-driven quantitative financial analyst.
    Aggregates technical metrics and raw news headlines into a clean, structured JSON report.
    """
    def __init__(self):
        self.client = get_client()
        self.model = Config.OPENAI_MODEL

    def run(self, raw_data: list, raw_news: dict) -> str:
        """
        Processes stock metrics and news feeds.
        Returns a structured JSON string.
        """
        system_prompt = (
            "You are a quantitative financial data analyst specializing in US equity markets.\n"
            "Your objective is to aggregate and structure stock technical metrics and news headlines.\n"
            "You must synthesize technical metrics (RSI, SMA Crossovers, Volume Spikes) and raw news into a factual, clean JSON report.\n\n"
            "Crucial Rules:\n"
            "1. Output ONLY a valid JSON object. No markdown block wraps (like ```json), no leading or trailing text. Just the raw JSON string.\n"
            "2. Be factual. Do not make speculative investment calls or write narrative paragraphs. Leave strategy to the Portfolio Manager.\n"
            "3. Identify warning indicators:\n"
            "   - RSI >= 70 is 'Overbought'.\n"
            "   - RSI <= 30 is 'Oversold'.\n"
            "   - Volume Spike Ratio >= 1.5 indicates a 'High Volume Spike'.\n"
            "   - SMA crossovers (e.g. Golden Cross, Death Cross).\n"
        )

        user_content = {
            "analysis_date": datetime.date.today().strftime("%Y-%m-%d"),
            "portfolio_technical_data": raw_data,
            "portfolio_news_data": raw_news
        }

        prompt = (
            f"Here is the raw stock price and technical data:\n{json.dumps(user_content['portfolio_technical_data'], indent=2)}\n\n"
            f"Here are the recent news headlines for these tickers:\n{json.dumps(user_content['portfolio_news_data'], indent=2)}\n\n"
            "Process this information and return a JSON object with the following structure:\n"
            "{\n"
            "  \"date\": \"YYYY-MM-DD\",\n"
            "  \"tickers\": {\n"
            "    \"TICKER\": {\n"
            "      \"name\": \"Company Name\",\n"
            "      \"price\": 123.45,\n"
            "      \"change_pct\": 1.25,\n"
            "      \"volume_ratio\": 1.2,\n"
            "      \"rsi\": 58.5,\n"
            "      \"technical_signals\": [\"List of indicators like RSI warning, Volume spike, SMA status\"],\n"
            "      \"top_news_summary\": [\"List of 2-3 summarized bullet points connecting headlines to market events\"]\n"
            "    }\n"
            "  },\n"
            "  \"macro_summary\": \"A short 2-3 sentence overview of major technical indicators and volume behaviors observed across the portfolio\"\n"
            "}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean markdown wraps if LLM accidentally outputs them
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()
            
        return content


class PortfolioManagerAgent:
    """
    Persona: A seasoned Wall Street portfolio manager and investment strategist.
    Consumes structured JSON data, synthesizes macro/sector outlooks, and drafts a beautifully designed HTML summary.
    """
    def __init__(self):
        self.client = get_client()
        self.model = Config.OPENAI_MODEL

    def run(self, analyst_json_str: str) -> str:
        """
        Takes structured JSON data from the analyst and writes the final summary.
        Returns a beautifully formatted HTML document.
        """
        # Ensure we have valid JSON to print or debug if needed
        try:
            parsed_data = json.loads(analyst_json_str)
            readable_data = json.dumps(parsed_data, indent=2)
        except Exception:
            readable_data = analyst_json_str

        system_prompt = (
            "You are a seasoned Wall Street Portfolio Manager and Senior Investment Strategist.\n"
            "You write highly personalized, professional, and strategic market reports for high-net-worth clients.\n"
            "Your writing style is sophisticated, authoritative, yet accessible and actionable. You excel at connecting micro stock technicals with macro sector trends.\n\n"
            "Your output must be a single, fully-styled HTML document representing a 'Daily Portfolio Executive Summary' email newsletter.\n"
            "Include inline CSS styling (no externals, no <style> blocks that get stripped by email clients if possible, or use standard safe <style> blocks in <head>).\n"
            "Make the visual layout look modern, professional, premium, and clean (Navy/Indigo accents, clean typography, colored badges, structured tables/cards, responsive layout).\n\n"
            "Rules for email newsletter content:\n"
            "1. Output ONLY the raw HTML string (starts with <!DOCTYPE html> or <html>). Do not wrap in ```html block markdown. No conversational introduction/outro in the response. Just the HTML document.\n"
            "2. Read the structured quantitative analysis from the Analyst.\n"
            "3. Structure your report into:\n"
            "   - **Strategic Daily Overview**: A brief high-level summary of the day's market tone.\n"
            "   - **Portfolio Asset Summary Table**: A beautiful table showing stock names, ticker, price, daily % change (styled green for positive, red for negative), RSI, and technical warning flags.\n"
            "   - **Detailed Ticker Spotlights & Catalyst Analysis**: Dive deeper into specific portfolio stocks experiencing significant news triggers, volume spikes, or key technical boundaries (overbought/oversold, SMA crosses).\n"
            "   - **Macro & Sector Risk Assessment**: Synthesize risk elements for Tech, Semiconductors, Rate Dependencies, or retail based on the stocks in the report.\n"
            "   - **Strategic Recommendations**: What to watch, potential hedge ideas, or key price levels for support/resistance.\n"
            "4. Ensure ALL ticker symbols mentioned are linked to their Yahoo Finance pages using standard links (e.g. https://finance.yahoo.com/quote/AAPL).\n"
        )

        prompt = (
            f"Here is the structured quantitative and news data prepared by the Financial Analyst:\n\n{readable_data}\n\n"
            "Draft the Daily Portfolio Executive Summary HTML email. Make it visual, clean, and highly professional. Ensure it matches all requirements."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        content = response.choices[0].message.content.strip()
        
        # Clean markdown wraps if LLM accidentally outputs them
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()
            
        return content
