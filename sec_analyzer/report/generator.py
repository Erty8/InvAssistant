"""Render a standalone, self-contained HTML "verdict card" report.

This module has two public entry points: :func:`render_report_html`, which
builds the report HTML as a string, and :func:`generate_report`, which calls
it and writes the result to disk. Both take the same pieces of data the
CLI's terminal verdict card and the web UI's verdict section render (the
``interpret()`` result, valuation metrics, merged technical
indicators/verdict, red flags, price, and as-of date), serialize them into a
single compact JSON payload, and inject that payload into ``template.html``
in place of the ``__DATA_JSON__`` placeholder. All rendering logic lives in
the template's inline ``<script>`` -- this module does no HTML
string-building of its own beyond that substitution, so the two stay in
sync by construction (one placeholder, one payload shape).

``result`` may be either the "classic" shape (``fair_value_range``,
``fundamental_verdict``, ``technical_verdict``, ``profile_fit``, ...) or the
richer two-phase shape described in ``sec_analyzer/valuation/SPEC.md``
Sec.11-12, which additionally carries a full ``result["valuation"]`` dict
(sector type, DCF/P-B×ROE scenarios, reverse-DCF, multiples, sensitivity,
triangulation) plus top-level ``confidence`` and ``reverse_dcf_comment``
fields. The template renders the triangulation row, sensitivity table, and
reverse-DCF comment line only when ``result["valuation"]`` is present,
degrading gracefully to the simpler classic card otherwise -- this module
itself doesn't need to know which shape it got; it always just passes
``result`` through verbatim.

No external resources are used anywhere (no CDN scripts/fonts/styles): the
template is fully self-contained, so a generated report can be opened
straight from disk (``file://...``) or emailed as a single attachment.
"""

import json
import logging
import os
from datetime import date
from typing import List, Optional, Tuple

from sec_analyzer.config import Config

logger = logging.getLogger(__name__)

#: Directory this module lives in -- used to locate template.html regardless
#: of the caller's current working directory.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

#: Path to the HTML template used for every generated report -- the same
#: file backs both the baked CLI/``/report`` card and the interactive
#: ``/`` search page (see ``render_search_page``); the two are told apart
#: by the injected payload's ``"mode"`` field.
_TEMPLATE_PATH = os.path.join(_PACKAGE_DIR, "template.html")

#: Placeholder in template.html that gets replaced with the JSON data blob.
#: Must appear exactly once in the template, inside a
#: ``<script type="application/json">`` element.
_DATA_PLACEHOLDER = "__DATA_JSON__"


def _load_template() -> str:
    """Read template.html from disk.

    Raises:
        OSError: If the template file is missing or unreadable -- this is a
            packaging error, not a runtime data issue, so it's allowed to
            propagate rather than being swallowed.
    """
    with open(_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _inject_payload(payload: dict) -> str:
    """Serialize ``payload`` and inject it into ``template.html`` in place
    of the ``__DATA_JSON__`` placeholder.

    Shared by :func:`render_report_html` (payload ``mode == "report"``) and
    :func:`render_search_page` (payload ``mode == "search"``) so both stay
    byte-for-byte in sync with the same load/escape/substitute logic.

    Raises:
        ValueError: If ``template.html`` is missing the ``__DATA_JSON__``
            placeholder (a packaging error).
    """
    data_json = json.dumps(payload, ensure_ascii=False)
    # Defensively guard against a literal "</script" sequence anywhere
    # inside the payload (e.g. inside a free-text LLM-generated field) from
    # prematurely closing the <script type="application/json"> element the
    # data is embedded in.
    data_json_safe = data_json.replace("</", "<\\/")

    template = _load_template()
    if _DATA_PLACEHOLDER not in template:
        raise ValueError(
            f"report template is missing the {_DATA_PLACEHOLDER!r} placeholder"
        )
    return template.replace(_DATA_PLACEHOLDER, data_json_safe)


def _safe_filename_component(value: str) -> str:
    """Keep a string safe to use as (part of) a Windows/POSIX filename.

    Strips everything except alphanumerics, ``-``, and ``_``. Falls back to
    ``"UNKNOWN"`` if that leaves nothing (e.g. a ticker made entirely of
    punctuation), so a generated path is never empty/malformed.
    """
    cleaned = "".join(c for c in str(value) if c.isalnum() or c in ("-", "_"))
    return cleaned or "UNKNOWN"


def render_report_html(
    ticker: str,
    horizon: str,
    result: dict,
    metrics: Optional[dict] = None,
    technical: Optional[dict] = None,
    flags: Optional[List[dict]] = None,
    price: Optional[float] = None,
    as_of: Optional[str] = None,
    entity_name: Optional[str] = None,
    analyst: Optional[dict] = None,
) -> str:
    """Build the standalone verdict-card report HTML as a string.

    This is the string-building half of :func:`generate_report` (which
    additionally writes the result to disk), factored out so a caller that
    wants to *serve* the report -- e.g. ``sec_analyzer.web.app``'s
    ``GET /report`` route -- doesn't have to write a temp file and read it
    back just to reuse this card. ``generate_report`` calls this function
    and then persists its return value verbatim, so the two never drift.

    Works whether ``result`` is a successful interpretation (matching the
    schema documented in ``sec_analyzer.interpret.analyzer.interpret``) or an
    error dict (``{"error": ..., "summary": ...}``) -- the template renders
    a simple error card in the latter case.

    Args:
        ticker: Stock ticker symbol, e.g. ``"NVDA"``. Upper-cased for
            display.
        horizon: One of ``"3m"``, ``"1y"``, ``"5y"``.
        result: The dict returned by
            :func:`sec_analyzer.interpret.analyzer.interpret`.
        metrics: The dict returned by
            :func:`sec_analyzer.normalize.metrics.compute_metrics`, or
            ``None`` if unavailable.
        technical: The merged indicators + technical-verdict dict (i.e.
            ``{**indicators, **technical_verdict_result}``), or ``None`` if
            price data was unavailable.
        flags: The list returned by
            :func:`sec_analyzer.normalize.red_flags.detect_red_flags`, or
            ``None``/empty if none fired.
        price: The latest market price per share, or ``None`` if unavailable.
        as_of: The date that ``price`` is as of (``"YYYY-MM-DD"``), or
            ``None``.
        entity_name: The filer's resolved company name (e.g. ``"Apple
            Inc."``), or ``None`` if unavailable -- shown beside the ticker
            in the header when present, omitted otherwise.
        analyst: The dict returned by
            :func:`sec_analyzer.fetch.analyst.get_analyst_targets`, or
            ``None`` if unavailable. Display-only consensus analyst-target
            cross-check -- never feeds the valuation engine.

    Returns:
        The complete, self-contained report HTML as a string.

    Raises:
        ValueError: If ``template.html`` is missing the ``__DATA_JSON__``
            placeholder (a packaging error).
    """
    ticker_upper = str(ticker).strip().upper()
    generated_on = date.today().isoformat()

    payload = {
        "mode": "report",
        "ticker": ticker_upper,
        "horizon": horizon,
        "price": price,
        "analyst": analyst,
        "as_of": as_of,
        "generated_on": generated_on,
        "result": result or {},
        "metrics": metrics or {},
        "technical": technical or {},
        "red_flags": flags or [],
        "entity_name": entity_name,
    }

    return _inject_payload(payload)


def render_search_page(
    horizons: List[Tuple[str, str]],
    providers: List[Tuple[str, str]],
    default_horizon: str,
    default_provider: str,
    default_model: str,
) -> str:
    """Build the interactive "Verdict Terminal" search page HTML.

    This is the ``GET /`` counterpart to :func:`render_report_html`: it
    loads the exact same ``template.html`` shell, but injects a
    ``mode: "search"`` payload instead of a baked ``result``. The
    template's client-side script reads this payload to render a live
    ticker/horizon/provider search box in the top nav; submitting it POSTs
    to ``/api/analyze`` and reshapes the JSON response into the same
    payload shape :func:`render_report_html` embeds, then renders it with
    the identical body-rendering code -- so the two entry points never
    visually drift.

    Args:
        horizons: ``(value, label)`` pairs for the horizon selector, e.g.
            ``[("3m", "3 ay"), ("1y", "1 yıl"), ("5y", "5 yıl")]``.
        providers: ``(value, label)`` pairs for the analysis-provider
            selector.
        default_horizon: The horizon value pre-selected on page load.
        default_provider: The provider value pre-selected on page load.
        default_model: Display-only label for the default provider's model
            (e.g. the configured Ollama model name).

    Returns:
        The complete, self-contained search-page HTML as a string.

    Raises:
        ValueError: If ``template.html`` is missing the ``__DATA_JSON__``
            placeholder (a packaging error).
    """
    payload = {
        "mode": "search",
        "horizons": [list(pair) for pair in horizons],
        "providers": [list(pair) for pair in providers],
        "default_horizon": default_horizon,
        "default_provider": default_provider,
        "default_model": default_model,
    }

    return _inject_payload(payload)


def generate_report(
    ticker: str,
    horizon: str,
    result: dict,
    metrics: Optional[dict] = None,
    technical: Optional[dict] = None,
    flags: Optional[List[dict]] = None,
    price: Optional[float] = None,
    as_of: Optional[str] = None,
    out_dir: Optional[str] = None,
    entity_name: Optional[str] = None,
    analyst: Optional[dict] = None,
) -> str:
    """Render and save the HTML verdict-card report for one ticker/horizon.

    Works whether ``result`` is a successful interpretation (matching the
    schema documented in ``sec_analyzer.interpret.analyzer.interpret``) or an
    error dict (``{"error": ..., "summary": ...}``) -- the template renders
    a simple error card in the latter case, but a file is written either
    way, so a caller never has to special-case an analysis failure just to
    get *some* report on disk.

    Args:
        ticker: Stock ticker symbol, e.g. ``"NVDA"``. Upper-cased for
            display and for the saved filename.
        horizon: One of ``"3m"``, ``"1y"``, ``"5y"``.
        result: The dict returned by
            :func:`sec_analyzer.interpret.analyzer.interpret`.
        metrics: The dict returned by
            :func:`sec_analyzer.normalize.metrics.compute_metrics`, or
            ``None`` if unavailable.
        technical: The merged indicators + technical-verdict dict (i.e.
            ``{**indicators, **technical_verdict_result}``), or ``None`` if
            price data was unavailable.
        flags: The list returned by
            :func:`sec_analyzer.normalize.red_flags.detect_red_flags`, or
            ``None``/empty if none fired.
        price: The latest market price per share, or ``None`` if unavailable.
        as_of: The date that ``price`` is as of (``"YYYY-MM-DD"``), or
            ``None``.
        out_dir: Directory to write the report into. Defaults to
            ``Config.REPORTS_DIR``; created if it doesn't already exist.
        entity_name: The filer's resolved company name, or ``None`` if
            unavailable -- see :func:`render_report_html`.
        analyst: The dict returned by
            :func:`sec_analyzer.fetch.analyst.get_analyst_targets`, or
            ``None`` if unavailable -- see :func:`render_report_html`.

    Returns:
        The path the report was saved to.
    """
    ticker_upper = str(ticker).strip().upper()
    generated_on = date.today().isoformat()

    html = render_report_html(
        ticker, horizon, result,
        metrics=metrics, technical=technical, flags=flags, price=price, as_of=as_of,
        entity_name=entity_name, analyst=analyst,
    )

    target_dir = out_dir or Config.REPORTS_DIR
    os.makedirs(target_dir, exist_ok=True)

    filename = f"{_safe_filename_component(ticker_upper)}_{generated_on}_{horizon}.html"
    path = os.path.join(target_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info("Wrote HTML verdict report for %s (%s) to %s", ticker_upper, horizon, path)
    return path
