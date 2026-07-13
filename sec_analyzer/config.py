"""Environment-driven configuration for the sec_analyzer package.

All runtime configuration is sourced from environment variables (optionally
loaded from a local ``.env`` file via ``python-dotenv``). Nothing sensitive
or SEC-identity-related is hardcoded: callers must supply their own values
via the environment.
"""

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv is not installed. Degrade gracefully and rely solely on
    # whatever is already present in the process environment (os.environ).
    pass


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


class Config:
    """Centralized, class-level configuration for sec_analyzer.

    All values are read once at import time from the environment. Use the
    provided classmethods to access values that require validation
    (``get_user_agent``, ``require_anthropic_key``) or that have filesystem
    side effects (``ensure_dirs``).
    """

    # Directory of the sec_analyzer package itself.
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Local cache directory for raw JSON payloads pulled from SEC EDGAR.
    RAW_DIR = os.path.join(BASE_DIR, "raw")

    # SQLite database path. Overridable via SEC_DB_PATH.
    DB_PATH = os.getenv("SEC_DB_PATH", os.path.join(BASE_DIR, "sec_data.sqlite3"))

    # Path to the methodology document. Overridable via METODOLOJI_PATH.
    METODOLOJI_PATH = os.getenv(
        "METODOLOJI_PATH", os.path.join(BASE_DIR, "METODOLOJI.md")
    )

    # Anthropic API configuration.
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    # Model is read from SEC_ANTHROPIC_MODEL (not the generic ANTHROPIC_MODEL)
    # on purpose: this package may share a .env with other tools that set
    # ANTHROPIC_MODEL to a different model, and we must not let that silently
    # downgrade the required interpretation model. Defaults to claude-opus-4-8.
    ANTHROPIC_MODEL = os.getenv("SEC_ANTHROPIC_MODEL", "claude-opus-4-8")

    # SEC EDGAR fair-access limit is 10 requests/sec. Default to 8 to stay
    # safely under that ceiling. Overridable via SEC_MAX_RPS.
    SEC_MAX_REQUESTS_PER_SEC = int(os.getenv("SEC_MAX_RPS", "8"))

    # Analyzer LLM backend selection. This chooses which model the
    # `interpret` layer talks to for fundamental analysis:
    #   * "ollama"    -- a local Gemma model served by Ollama (DEFAULT; free,
    #                    private, no API key required, requires Ollama to be
    #                    running locally).
    #   * "anthropic" -- the hosted Claude API (requires ANTHROPIC_API_KEY).
    # Overridable via ANALYZER_PROVIDER.
    ANALYZER_PROVIDER = os.getenv("ANALYZER_PROVIDER", "ollama").lower()

    # Local Ollama settings (used when ANALYZER_PROVIDER == "ollama").
    # OLLAMA_HOST is the base URL of the Ollama server's REST API (the
    # `/api/chat` endpoint is appended by the client). OLLAMA_MODEL is the
    # name of a model already pulled locally, e.g. via `ollama pull gemma4`.
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:latest")

    # Per-hold-horizon (fundamental_weight, technical_weight) pairs, used by
    # the interpret layer to tell the LLM/rule-based analyzer how much to
    # lean on fundamentals vs. technicals for a given investment horizon.
    # Short horizons (3m) lean technical; long horizons (5y) lean
    # fundamental. Weights sum to 1.0 for each horizon.
    HORIZON_WEIGHTS = {"3m": (0.3, 0.7), "1y": (0.5, 0.5), "5y": (0.8, 0.2)}

    # Path to an optional investor-profile document (risk tolerance,
    # investment style, position-size limits, behavioral notes, etc.). When
    # present, its contents are merged into the interpret layer's LLM system
    # prompt so the analysis can be judged for "profile fit". Overridable
    # via PROFIL_PATH. Not required -- if missing, a neutral-default
    # investor profile is assumed instead.
    PROFIL_PATH = os.getenv("PROFIL_PATH", os.path.join(os.getcwd(), "PROFIL.md"))

    # Directory where generated analysis reports are written. Overridable
    # via REPORTS_DIR.
    REPORTS_DIR = os.getenv("REPORTS_DIR", os.path.join(os.getcwd(), "reports"))

    # Path to the valuation-methodology document (DCF/reverse-DCF/multiples
    # conventions, scenario philosophy, etc.) merged into the two-phase
    # interpret flow's phase-1 (assumption proposal) system prompt, after
    # METODOLOJI.md and before PROFIL.md. Overridable via VALUATION_PATH.
    VALUATION_PATH = os.getenv("VALUATION_PATH", os.path.join(BASE_DIR, "VALUATION.md"))

    # Directory holding local Damodaran sector-multiple/ERP reference CSVs
    # (see sec_analyzer/valuation/damodaran.py) -- optional, not fetched
    # over the network. Overridable via DAMODARAN_DIR.
    DAMODARAN_DIR = os.getenv("DAMODARAN_DIR", os.path.join(os.getcwd(), "data", "damodaran"))

    @classmethod
    def get_user_agent(cls) -> str:
        """Return the User-Agent string SEC requires for all EDGAR requests.

        SEC mandates that every request identify a real requester, e.g.
        ``"Name Surname email@example.com"``. There is no safe default for
        this value, so it must be supplied explicitly via the ``SEC_USER_AGENT``
        environment variable (typically in a local ``.env`` file).

        Raises:
            ConfigError: If ``SEC_USER_AGENT`` is unset or blank.
        """
        user_agent = os.getenv("SEC_USER_AGENT", "").strip()
        if not user_agent:
            raise ConfigError(
                "SEC_USER_AGENT is not set. SEC EDGAR requires every request "
                "to identify a real requester via the User-Agent header, in "
                "the form 'Name Surname email@example.com'. Set it in your "
                "environment or in a .env file, e.g.:\n"
                '  SEC_USER_AGENT="Jane Doe jane.doe@example.com"\n'
                "No default identity is provided on purpose -- SEC can block "
                "or rate-limit generic/anonymous User-Agent strings."
            )
        return user_agent

    @classmethod
    def ensure_dirs(cls) -> None:
        """Create any directories required by sec_analyzer if missing."""
        os.makedirs(cls.RAW_DIR, exist_ok=True)

    @classmethod
    def require_anthropic_key(cls) -> str:
        """Return the configured Anthropic API key.

        Raises:
            ConfigError: If ``ANTHROPIC_API_KEY`` is unset or blank.
        """
        if not cls.ANTHROPIC_API_KEY:
            raise ConfigError(
                "ANTHROPIC_API_KEY is not set. Set it in your environment or "
                "in a .env file to use Anthropic-powered features."
            )
        return cls.ANTHROPIC_API_KEY
