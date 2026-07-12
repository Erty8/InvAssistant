"""Shared HTTP client for talking to SEC EDGAR.

Built on top of the plain ``requests`` library only (no third-party SEC
wrappers). Handles the two things SEC EDGAR access requires in practice:

1. A well-formed, identifying ``User-Agent`` header on every request.
2. Polite, throttled request pacing plus resilient retry/backoff behavior
   for rate-limiting (429), transient blocks (403), and server errors (5xx).
"""

import logging
import time

import requests

from .config import Config, ConfigError

logger = logging.getLogger(__name__)

# Status codes that warrant a retry with exponential backoff.
_RETRYABLE_STATUS_CODES = {403, 429}

# Base delay (seconds) for exponential backoff: base_delay * (2 ** attempt).
_BASE_DELAY = 1.0

# Upper bound (seconds) on any single backoff sleep.
_MAX_DELAY = 30.0


class SecHttpClient:
    """A throttled, retrying HTTP client for SEC EDGAR endpoints.

    Example:
        client = SecHttpClient()
        data = client.get_json("https://data.sec.gov/submissions/CIK0000320193.json")
    """

    def __init__(self, user_agent: str = None, max_rps: int = None):
        """Initialize the client.

        Args:
            user_agent: Override for the SEC-required User-Agent string. If
                not provided, it is read from configuration (which raises
                ``ConfigError`` if unset).
            max_rps: Override for the maximum requests-per-second throttle.
                If not provided, uses ``Config.SEC_MAX_REQUESTS_PER_SEC``.
        """
        self.user_agent = user_agent or Config.get_user_agent()
        self.max_rps = max_rps or Config.SEC_MAX_REQUESTS_PER_SEC
        self._min_interval = 1.0 / self.max_rps
        self._last_request_ts = 0.0

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
            }
        )

    def _throttle(self) -> None:
        """Block, if necessary, to keep the request rate under ``max_rps``."""
        now = time.monotonic()
        elapsed = now - self._last_request_ts
        remaining = self._min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_ts = time.monotonic()

    @staticmethod
    def _compute_backoff(attempt: int, retry_after: str = None) -> float:
        """Compute the sleep duration for a given retry attempt.

        Combines exponential backoff with any server-provided ``Retry-After``
        header, taking the larger of the two, capped at ``_MAX_DELAY``.
        """
        backoff = _BASE_DELAY * (2 ** attempt)
        if retry_after is not None:
            try:
                backoff = max(backoff, float(retry_after))
            except (TypeError, ValueError):
                pass
        return min(backoff, _MAX_DELAY)

    def _request(self, url: str, timeout: int, max_retries: int, binary: bool):
        """Shared GET implementation for both ``get_json`` and ``get_bytes``.

        Args:
            url: The URL to fetch.
            timeout: Per-request timeout in seconds.
            max_retries: Maximum number of retry attempts after the initial try.
            binary: If True, return ``response.content``; otherwise
                ``response.json()``.

        Returns:
            The parsed JSON body (dict) or raw bytes, depending on ``binary``.

        Raises:
            requests.HTTPError: On a 404 (immediately, no retry) or after
                retries are exhausted on other error statuses.
            requests.RequestException: After retries are exhausted on
                connection/timeout errors.
        """
        last_exc = None

        for attempt in range(max_retries + 1):
            self._throttle()

            try:
                response = self.session.get(url, timeout=timeout)
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= max_retries:
                    logger.error(
                        "Request to %s failed after %d attempts: %s",
                        url,
                        attempt + 1,
                        exc,
                    )
                    raise
                delay = self._compute_backoff(attempt)
                logger.warning(
                    "Request to %s raised %s (attempt %d/%d); retrying in %.1fs",
                    url,
                    exc,
                    attempt + 1,
                    max_retries + 1,
                    delay,
                )
                time.sleep(delay)
                continue

            status = response.status_code

            if status == 404:
                # The resource genuinely doesn't exist -- do not retry.
                response.raise_for_status()

            if status == 200:
                return response.content if binary else response.json()

            if status in _RETRYABLE_STATUS_CODES or status >= 500:
                last_exc = requests.HTTPError(
                    f"{status} error for url: {url}", response=response
                )
                if attempt >= max_retries:
                    logger.error(
                        "Request to %s failed with status %d after %d attempts",
                        url,
                        status,
                        attempt + 1,
                    )
                    raise last_exc
                delay = self._compute_backoff(
                    attempt, retry_after=response.headers.get("Retry-After")
                )
                logger.warning(
                    "Request to %s returned status %d (attempt %d/%d); "
                    "retrying in %.1fs",
                    url,
                    status,
                    attempt + 1,
                    max_retries + 1,
                    delay,
                )
                time.sleep(delay)
                continue

            # Any other non-success status: surface it immediately.
            response.raise_for_status()

        # Should be unreachable, but guard against falling through silently.
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Exhausted retries for {url} with no recorded error")

    def get_json(self, url: str, timeout: int = 30, max_retries: int = 5) -> dict:
        """Fetch a URL and return its JSON-decoded body.

        Throttles before every attempt and retries on 429/403/5xx responses
        and on connection errors, using exponential backoff (capped at 30s
        per sleep, honoring ``Retry-After`` when present). A 404 is raised
        immediately without retrying.

        Args:
            url: The URL to fetch.
            timeout: Per-request timeout in seconds.
            max_retries: Maximum number of retries after the initial attempt.

        Returns:
            The decoded JSON body as a dict.

        Raises:
            requests.HTTPError: On 404, or after retries are exhausted.
            requests.RequestException: After retries are exhausted on
                connection/timeout errors.
        """
        return self._request(url, timeout=timeout, max_retries=max_retries, binary=False)

    def get_bytes(self, url: str, timeout: int = 30, max_retries: int = 5) -> bytes:
        """Fetch a URL and return its raw response body as bytes.

        Same throttling and retry semantics as ``get_json``, but returns raw
        bytes instead of decoding JSON. Useful for fetching binary artifacts
        (e.g. PDFs, XBRL zip archives) from SEC EDGAR.

        Args:
            url: The URL to fetch.
            timeout: Per-request timeout in seconds.
            max_retries: Maximum number of retries after the initial attempt.

        Returns:
            The raw response body as bytes.

        Raises:
            requests.HTTPError: On 404, or after retries are exhausted.
            requests.RequestException: After retries are exhausted on
                connection/timeout errors.
        """
        return self._request(url, timeout=timeout, max_retries=max_retries, binary=True)
