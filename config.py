import os
from dotenv import load_dotenv

# Load env variables from .env file if it exists
load_dotenv()

class Config:
    # LLM Settings
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()  # 'openai', 'gemini', or 'anthropic'
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", None)
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

    # Portfolio settings
    PORTFOLIO_TICKERS = [
        t.strip().upper() 
        for t in os.getenv("PORTFOLIO_TICKERS", "AAPL,MSFT,NVDA,AMZN,GOOGL").split(",") 
        if t.strip()
    ]

    # SMTP Settings
    SENDER_EMAIL = os.getenv("SENDER_EMAIL", "")
    RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL", "")
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
    SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "True").lower() in ("true", "1", "yes")

    # Local storage for summaries
    SAVE_LOCAL_COPY = os.getenv("SAVE_LOCAL_COPY", "True").lower() in ("true", "1", "yes")
    REPORTS_DIR = "reports"

    @classmethod
    def validate(cls):
        """Validates critical configurations and prints warning messages."""
        warnings = []
        errors = []

        if cls.LLM_PROVIDER == "gemini":
            if not cls.GEMINI_API_KEY:
                errors.append("GEMINI_API_KEY environment variable is missing. LLM agents will fail to execute.")
        elif cls.LLM_PROVIDER == "anthropic":
            if not cls.ANTHROPIC_API_KEY:
                errors.append("ANTHROPIC_API_KEY environment variable is missing. LLM agents will fail to execute.")
        else:  # Default is openai
            if not cls.OPENAI_API_KEY:
                errors.append("OPENAI_API_KEY environment variable is missing. LLM agents will fail to execute.")

        if not cls.PORTFOLIO_TICKERS:
            errors.append("PORTFOLIO_TICKERS environment variable is empty. No tickers to analyze.")

        if not cls.SENDER_EMAIL or not cls.RECEIVER_EMAIL:
            warnings.append("SENDER_EMAIL or RECEIVER_EMAIL is not set. Email notifications will be skipped.")

        if not cls.SMTP_PASSWORD:
            warnings.append("SMTP_PASSWORD is not set. Email delivery might be bypassed or fail.")

        return errors, warnings
