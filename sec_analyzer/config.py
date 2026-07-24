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
    (``get_user_agent``) or that have filesystem side effects
    (``ensure_dirs``).
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

    # SEC EDGAR fair-access limit is 10 requests/sec. Default to 8 to stay
    # safely under that ceiling. Overridable via SEC_MAX_RPS.
    SEC_MAX_REQUESTS_PER_SEC = int(os.getenv("SEC_MAX_RPS", "8"))

    # LLM backend selection (transport + WHO gets billed). This is the
    # high-level knob for how an LLM-based interpretation runs:
    #   * "claude_code" (DEFAULT) -- drive the local `claude` CLI (`claude -p`)
    #     as a subprocess. Billed to your Claude *subscription*, not an API
    #     account. Requires `claude` installed and logged in, and
    #     ANTHROPIC_API_KEY UNSET -- a set key would silently redirect
    #     `claude -p` to API billing, so the claude_code backend refuses to run
    #     in that case (see interpret/backends/claude_code.py).
    #   * "none" -- the deterministic, rule-based analyzer (no AI, no key, free,
    #     fully offline and reproducible); maps to the "script" provider.
    # The hosted HTTP API backend ("api"/"anthropic") has been REMOVED; any
    # legacy "api"/"anthropic" value routes to "claude_code" below.
    # Overridable via LLM_BACKEND.
    LLM_BACKEND = os.getenv("LLM_BACKEND", "claude_code").lower()

    #: Maps the billing-oriented LLM_BACKEND onto the low-level analyzer
    #: provider the interpret layer dispatches on. Unknown/legacy backends
    #: (including the removed "api") fall through to "claude_code".
    _LLM_BACKEND_TO_PROVIDER = {
        "claude_code": "claude_code",
        "none": "script",
    }

    # Analyzer backend (low-level provider) the `interpret` layer dispatches on:
    #   * "claude_code" -- local `claude -p` subprocess (subscription billing).
    #   * "script"      -- deterministic, rule-based analyzer (no AI, offline).
    #   * "ollama"      -- a local Gemma model served by Ollama.
    # Resolution: an explicit ANALYZER_PROVIDER env var wins (advanced /
    # backward-compatible override); otherwise it is derived from LLM_BACKEND
    # (default provider "claude_code"). The hosted-API backend is gone, so a
    # legacy "anthropic"/"api" selection (from either source) is remapped to
    # "claude_code" rather than crashing.
    _RAW_ANALYZER_PROVIDER = (
        os.environ.get("ANALYZER_PROVIDER")
        or _LLM_BACKEND_TO_PROVIDER.get(LLM_BACKEND, "claude_code")
    ).lower()
    ANALYZER_PROVIDER = (
        "claude_code"
        if _RAW_ANALYZER_PROVIDER in ("anthropic", "api")
        else _RAW_ANALYZER_PROVIDER
    )

    # Claude Code CLI settings (used when the resolved provider is
    # "claude_code"). CLAUDE_CODE_BIN is the CLI binary name/path (looked up on
    # PATH via shutil.which); CLAUDE_CODE_TIMEOUT is the per-call subprocess
    # timeout in seconds. Overridable via CLAUDE_CODE_BIN / CLAUDE_CODE_TIMEOUT.
    CLAUDE_CODE_BIN = os.getenv("CLAUDE_CODE_BIN", "claude")
    CLAUDE_CODE_TIMEOUT = int(os.getenv("CLAUDE_CODE_TIMEOUT", "120"))

    # Per-phase model for the claude_code backend, passed to `claude --model`.
    # The two phases have different needs, so they get different tiers:
    #   * phase 1 (assumption proposal) -- the numeric/judgment step that drives
    #     the fair-value numbers -- uses the STRONGER model (Opus).
    #   * phase 2 (commentary) -- narrative synthesis that never touches the
    #     numbers -- uses a cheaper/faster model (Sonnet).
    # Values are Claude Code model *aliases* ("opus"/"sonnet"/"haiku"), which the
    # CLI resolves to the current model of that tier on your subscription plan;
    # a pinned full id (e.g. "claude-opus-4-8") also works. An explicit --model
    # (e.g. `assumptions propose --model ...`) overrides these. Overridable via
    # CLAUDE_CODE_MODEL_ASSUMPTIONS / CLAUDE_CODE_MODEL_COMMENTARY.
    CLAUDE_CODE_MODEL_ASSUMPTIONS = os.getenv("CLAUDE_CODE_MODEL_ASSUMPTIONS", "opus")
    CLAUDE_CODE_MODEL_COMMENTARY = os.getenv("CLAUDE_CODE_MODEL_COMMENTARY", "sonnet")

    @classmethod
    def claude_code_model_for_phase(cls, phase: str) -> str:
        """Default claude_code model for a dispatch phase.

        ``phase == "assumptions"`` -> the strong model (Opus); anything else
        (commentary) -> the cheaper model (Sonnet).
        """
        if phase == "assumptions":
            return cls.CLAUDE_CODE_MODEL_ASSUMPTIONS
        return cls.CLAUDE_CODE_MODEL_COMMENTARY

    # Local Ollama settings (used when ANALYZER_PROVIDER == "ollama").
    # OLLAMA_HOST is the base URL of the Ollama server's REST API (the
    # `/api/chat` endpoint is appended by the client). OLLAMA_MODEL is the
    # name of a model already pulled locally, e.g. via `ollama pull gemma4`.
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:latest")

    # Context window (tokens) requested per Ollama call. Ollama's server-side
    # default (4096) silently truncates the long methodology system prompt
    # (~14k tokens alone) + financial JSON payload, which makes local models
    # emit empty or degenerate output. Overridable via OLLAMA_NUM_CTX.
    OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "32768"))

    # Per-request timeout (seconds) for Ollama calls. Cold-start evaluation of
    # the ~20k-token analyzer prompt on consumer hardware can take ~10 minutes
    # (subsequent calls hit Ollama's prompt cache and finish in ~1 minute), so
    # this is deliberately generous. Overridable via OLLAMA_TIMEOUT.
    OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "1200"))

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

    # Benchmark-based mature-state FCF-margin ceiling for the hyper-grower
    # revenue-first DCF (valuation engine "B-prime", staged). The mature target
    # FCF margin is FLAGGED up to this ceiling and CAPPED beyond it -- replacing
    # the earlier arbitrary flat cap. The value is the empirical upper-decile
    # sustained FCF margin of mature software companies (Damodaran "Software
    # (System & Application)"; MSFT/ADBE realized margins sit at ~35-40%). This
    # is a hardcoded placeholder: once the per-sector upper-decile FCF margin is
    # available in DAMODARAN_DIR, read it from there instead. Overridable via
    # MATURE_SOFTWARE_FCF_MARGIN_CEILING.
    MATURE_SOFTWARE_FCF_MARGIN_CEILING = float(
        os.getenv("MATURE_SOFTWARE_FCF_MARGIN_CEILING", "0.37")
    )

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

