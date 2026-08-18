"""Shared, always-on safety fixtures for the sec_analyzer test suite.

Both fixtures here are ``autouse``: they apply to every test in the package
without anyone having to remember them. That is deliberate. Both guarantees
below were previously enforced per-test, opt-in -- and both were breached
exactly because opt-in isolation only protects the tests whose author thought
of it.

**1. No test may write into the real on-disk cache.**
``Config.RAW_DIR`` is the user's production cache (SEC companyfacts,
submissions, Form 4 documents, price history, FRED/Treasury series -- roughly
a gigabyte of it). A single test fixture once leaked a five-row 2022 ``DGS10``
CSV into it. Because the risk-free rate feeds both the CAPM cost of equity and
the terminal-growth anchor, every *live* valuation on that machine would then
have been priced off a 2022 yield -- silently, with no error anywhere. Running
the test suite must never be able to corrupt real data.

**2. No test may reach the network.**
The same leak masked a second problem: with a cache file present, a test that
called straight through to a live fetch looked fast and green. Once the leak
was cleaned up, that test took tens of seconds (unreachable host -> retry
backoff -> fallback provider) and the suite went from ~40 seconds to over ten
minutes. A test that silently depends on the network is not just slow, it is
non-deterministic -- it passes or fails on someone else's uptime. This fixture
makes any unstubbed HTTP call fail immediately and loudly, naming the URL, so
the fix is obvious: stub at the boundary.

Tests that genuinely need to exercise HTTP behaviour stub ``requests`` (or the
fetch function above it) themselves; those stubs replace the attribute and are
therefore unaffected by the block installed here.
"""

import pytest

import requests

from sec_analyzer.config import Config


class NetworkAccessError(RuntimeError):
    """Raised when a test attempts a real HTTP request.

    Seeing this in a test run is not a flaky failure -- it means that test has
    an unstubbed dependency on a live remote host. Stub the fetch function (or
    ``requests``) at the module boundary the code under test imports it from.
    """


@pytest.fixture(autouse=True)
def _isolate_cache_dir(tmp_path, monkeypatch):
    """Point ``Config.RAW_DIR`` at a per-test temporary directory.

    Every cache-writing code path in ``sec_analyzer.fetch`` derives its path
    from this attribute, so redirecting it is enough to contain all of them --
    including ones added later, which is the point of doing it here rather
    than in each test module.
    """
    cache_dir = tmp_path / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Config, "RAW_DIR", str(cache_dir))


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    """Make any unstubbed real HTTP request fail immediately.

    Patches the two entry points every caller in this package funnels through:
    ``requests.Session.request`` (used by ``SecHttpClient`` and yfinance) and
    the module-level ``requests.get``/``requests.request`` helpers (used by the
    FRED/Treasury fetchers). Failing fast beats hanging: an unreachable host
    otherwise costs a retry-backoff cycle per call.
    """

    def _blocked(*args, **kwargs):
        url = kwargs.get("url")
        if url is None:
            # Positional form: (self, method, url) on Session.request,
            # (method, url) on requests.request, (url,) on requests.get.
            for candidate in args:
                if isinstance(candidate, str) and "://" in candidate:
                    url = candidate
                    break
        raise NetworkAccessError(
            f"Real network access attempted in a test (url={url!r}). "
            "Stub the fetch function or `requests` at the boundary instead."
        )

    monkeypatch.setattr(requests.Session, "request", _blocked)
    monkeypatch.setattr(requests, "request", _blocked)
    monkeypatch.setattr(requests, "get", _blocked)
