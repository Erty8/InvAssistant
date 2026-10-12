"""Point-in-time S&P 500 membership history.

Source: the MIT-licensed ``fja05680/sp500`` dataset on GitHub
(``S&P 500 Historical Components & Changes (Updated).csv``) -- the 1996-2019
list distributed with Andreas Clenow's *Trading Evolved*, extended since from
the S&P 500 Wikipedia page. Each row is ``date,"T1,T2,..."``: the full
membership in effect from that date until the next row. Tickers are the
symbols in use *on that date* (a later rename shows up as a remove + add).

The file is downloaded once into ``Config.RAW_DIR`` and read locally after
that, so studies that use it stay deterministic; a ``.source`` sidecar
records the URL and SHA-256 of the bytes actually used. Backtest-only data:
nothing on the live analysis path reads it.
"""

import bisect
import csv
import hashlib
import logging
import os
from datetime import date
from typing import Optional

import requests

from sec_analyzer.config import Config

logger = logging.getLogger(__name__)

SOURCE_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
_FILENAME = "sp500_membership_history.csv"


class MembershipHistory:
    """Sorted change dates with the member set in effect from each one."""

    def __init__(self, rows: "list[tuple[date, frozenset]]", sha256: Optional[str] = None):
        rows = sorted(rows, key=lambda r: r[0])
        self._dates = [d for d, _ in rows]
        self._sets = [s for _, s in rows]
        self.sha256 = sha256

    @property
    def first_date(self) -> Optional[date]:
        return self._dates[0] if self._dates else None

    @property
    def last_date(self) -> Optional[date]:
        return self._dates[-1] if self._dates else None

    def members_on(self, d: date) -> frozenset:
        """Members in effect on ``d`` (the last change on/before it); empty
        before the first row."""
        i = bisect.bisect_right(self._dates, d) - 1
        return self._sets[i] if i >= 0 else frozenset()

    def all_tickers(self, start: Optional[date] = None, end: Optional[date] = None) -> "set[str]":
        """Every ticker that was a member at any point in ``[start, end]``."""
        out: set = set()
        for i, d in enumerate(self._dates):
            nxt = self._dates[i + 1] if i + 1 < len(self._dates) else None
            if end is not None and d > end:
                break
            if start is not None and nxt is not None and nxt <= start:
                continue
            out |= self._sets[i]
        return out


def parse_membership_csv(text: str) -> "list[tuple[date, frozenset]]":
    rows = []
    for rec in csv.DictReader(text.splitlines()):
        try:
            d = date.fromisoformat(rec["date"].strip())
        except (KeyError, ValueError, AttributeError):
            continue
        tickers = frozenset(t.strip().upper() for t in (rec.get("tickers") or "").split(",") if t.strip())
        if tickers:
            rows.append((d, tickers))
    return rows


def _path() -> str:
    return os.path.join(Config.RAW_DIR, _FILENAME)


def load_membership_history(refresh: bool = False, timeout: int = 60) -> MembershipHistory:
    """Load the membership history, downloading it once if not cached.

    Raises ``OSError``/``requests.RequestException`` when there is no local
    copy and the download fails -- callers in the backtest layer convert that
    into an error result.
    """
    path = _path()
    if refresh or not os.path.exists(path):
        resp = requests.get(SOURCE_URL, timeout=timeout)
        resp.raise_for_status()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(resp.content)
        digest = hashlib.sha256(resp.content).hexdigest()
        with open(path + ".source", "w", encoding="utf-8") as f:
            f.write(f"{SOURCE_URL}\nsha256={digest}\n")
        logger.info("Downloaded S&P 500 membership history (%d bytes, sha256 %s)", len(resp.content), digest)
    with open(path, "rb") as f:
        raw = f.read()
    return MembershipHistory(parse_membership_csv(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest())
