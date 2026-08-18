"""Unit tests for the Form 4 insider-trading signal layer:

* ``sec_analyzer.fetch.insider`` -- network + immutable on-disk cache + XML
  parsing of a filer's recent Form 4 filings.
* ``sec_analyzer.signals.insider`` -- pure, deterministic aggregation of the
  parsed transactions into a buy/sell verdict.

No real network access is used anywhere in this module: a small fake HTTP
client stands in for ``SecHttpClient`` (mirroring ``test_earnings.py``'s
fake ``yfinance.Ticker``), and ``Config.RAW_DIR`` is pointed at pytest's
``tmp_path`` so nothing touches the package's real cache directory. A fixed
``today`` is passed everywhere so lookback windows and point-in-time guards
are deterministic and independent of the wall clock.
"""

from datetime import date

import pytest

from sec_analyzer.config import Config
from sec_analyzer.fetch import insider as insider_fetch
from sec_analyzer.signals import insider as insider_signal

_TODAY = date(2026, 7, 16)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _submissions(rows):
    """Build a submissions dict from ``rows`` of
    ``(form, filingDate, accession, primaryDoc)`` tuples.
    """
    return {
        "filings": {
            "recent": {
                "form": [r[0] for r in rows],
                "filingDate": [r[1] for r in rows],
                "accessionNumber": [r[2] for r in rows],
                "primaryDocument": [r[3] for r in rows],
            }
        }
    }


def _form4_xml(
    *,
    period="2026-06-12",
    person="Cook Timothy D",
    is_director=True,
    is_officer=True,
    is_ten_percent=False,
    officer_title="Chief Executive Officer",
    nd_date="2026-06-12",
    nd_code="P",
    nd_shares=1000,
    nd_price=210.5,
    nd_direction="A",
    nd_owned_after=None,
    d_date="2026-06-10",
    d_code="M",
    d_shares=500,
    d_price=0,
    d_direction="A",
    d_owned_after=None,
):
    """Build a realistic inline Form 4 ownershipDocument XML string with one
    nonDerivativeTransaction and one derivativeTransaction.
    """
    nd_post = (
        f"<postTransactionAmounts><sharesOwnedFollowingTransaction>"
        f"<value>{nd_owned_after}</value></sharesOwnedFollowingTransaction>"
        f"</postTransactionAmounts>"
        if nd_owned_after is not None
        else ""
    )
    d_post = (
        f"<postTransactionAmounts><sharesOwnedFollowingTransaction>"
        f"<value>{d_owned_after}</value></sharesOwnedFollowingTransaction>"
        f"</postTransactionAmounts>"
        if d_owned_after is not None
        else ""
    )
    return f"""<?xml version="1.0"?>
<ownershipDocument>
  <periodOfReport>{period}</periodOfReport>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0000012345</rptOwnerCik>
      <rptOwnerName>{person}</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>{1 if is_director else 0}</isDirector>
      <isOfficer>{1 if is_officer else 0}</isOfficer>
      <isTenPercentOwner>{1 if is_ten_percent else 0}</isTenPercentOwner>
      <isOther>0</isOther>
      <officerTitle>{officer_title}</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>{nd_date}</value></transactionDate>
      <transactionCoding><transactionCode>{nd_code}</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>{nd_shares}</value></transactionShares>
        <transactionPricePerShare><value>{nd_price}</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>{nd_direction}</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      {nd_post}
    </nonDerivativeTransaction>
  </nonDerivativeTable>
  <derivativeTable>
    <derivativeTransaction>
      <transactionDate><value>{d_date}</value></transactionDate>
      <transactionCoding><transactionCode>{d_code}</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>{d_shares}</value></transactionShares>
        <transactionPricePerShare><value>{d_price}</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>{d_direction}</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      {d_post}
    </derivativeTransaction>
  </derivativeTable>
</ownershipDocument>""".encode("utf-8")


class _FakeClient:
    """Minimal stand-in for ``SecHttpClient``. ``responses`` maps a URL
    substring match (the accession-stripped segment) to the bytes to return;
    ``raise_for`` is a set of such substrings that raise instead."""

    def __init__(self, responses=None, raise_for=None):
        self.responses = responses or {}
        self.raise_for = raise_for or set()
        self.calls = []

    def get_bytes(self, url, timeout=30, max_retries=5):
        self.calls.append(url)
        for key in self.raise_for:
            if key in url:
                raise RuntimeError(f"simulated network failure for {url}")
        for key, payload in self.responses.items():
            if key in url:
                return payload
        raise AssertionError(f"_FakeClient: no canned response for {url}")


# ---------------------------------------------------------------------------
# fetch.insider: XML parsing
# ---------------------------------------------------------------------------


def test_parse_form4_xml_extracts_fields():
    raw = _form4_xml()
    transactions = insider_fetch._parse_form4_xml(raw, filed="2026-06-14", accession="0000320193-26-000012")

    assert len(transactions) == 2
    nd = next(t for t in transactions if not t["derivative"])
    d = next(t for t in transactions if t["derivative"])

    assert nd["date"] == "2026-06-12"
    assert nd["filed"] == "2026-06-14"
    assert nd["person"] == "Cook Timothy D"
    assert nd["is_director"] is True
    assert nd["is_officer"] is True
    assert nd["is_ten_percent"] is False
    assert nd["officer_title"] == "Chief Executive Officer"
    assert nd["code"] == "P"
    assert nd["shares"] == 1000.0
    assert nd["price"] == pytest.approx(210.5)
    assert nd["direction"] == "A"
    assert nd["derivative"] is False
    assert nd["accession"] == "0000320193-26-000012"

    assert d["code"] == "M"
    assert d["derivative"] is True
    assert d["price"] == 0.0  # explicit zero must survive, not become None

    # No postTransactionAmounts supplied in the default fixture.
    assert nd["shares_owned_after"] is None
    assert d["shares_owned_after"] is None


def test_shares_owned_after_parsed_from_post_transaction_amounts():
    raw = _form4_xml(nd_owned_after=48000, d_owned_after=1200)
    transactions = insider_fetch._parse_form4_xml(raw, filed="2026-06-14", accession="acc")

    nd = next(t for t in transactions if not t["derivative"])
    d = next(t for t in transactions if t["derivative"])
    assert nd["shares_owned_after"] == pytest.approx(48000.0)
    assert d["shares_owned_after"] == pytest.approx(1200.0)


def test_parse_form4_xml_drops_rows_without_code_or_date():
    raw = b"""<ownershipDocument>
      <nonDerivativeTable>
        <nonDerivativeTransaction>
          <transactionAmounts>
            <transactionShares><value>100</value></transactionShares>
          </transactionAmounts>
        </nonDerivativeTransaction>
      </nonDerivativeTable>
    </ownershipDocument>"""
    assert insider_fetch._parse_form4_xml(raw, filed="2026-06-14", accession="acc") == []


def test_parse_form4_xml_malformed_returns_empty_not_raise():
    assert insider_fetch._parse_form4_xml(b"not xml at all <<<", "2026-06-14", "acc") == []
    assert insider_fetch._parse_form4_xml(b"", "2026-06-14", "acc") == []


def test_namespaced_xml_still_parses_via_strip_namespace():
    raw = b"""<ns:ownershipDocument xmlns:ns="urn:example">
      <ns:reportingOwner>
        <ns:reportingOwnerId><ns:rptOwnerName>Jane Doe</ns:rptOwnerName></ns:reportingOwnerId>
        <ns:reportingOwnerRelationship><ns:isOfficer>1</ns:isOfficer></ns:reportingOwnerRelationship>
      </ns:reportingOwner>
      <ns:nonDerivativeTable>
        <ns:nonDerivativeTransaction>
          <ns:transactionDate><ns:value>2026-06-01</ns:value></ns:transactionDate>
          <ns:transactionCoding><ns:transactionCode>S</ns:transactionCode></ns:transactionCoding>
          <ns:transactionAmounts>
            <ns:transactionShares><ns:value>10</ns:value></ns:transactionShares>
          </ns:transactionAmounts>
        </ns:nonDerivativeTransaction>
      </ns:nonDerivativeTable>
    </ns:ownershipDocument>"""
    transactions = insider_fetch._parse_form4_xml(raw, "2026-06-02", "acc")
    assert len(transactions) == 1
    assert transactions[0]["person"] == "Jane Doe"
    assert transactions[0]["code"] == "S"


# ---------------------------------------------------------------------------
# get_insider_transactions: URL construction / filtering / caching
# ---------------------------------------------------------------------------


def test_malformed_filing_counts_as_failed_others_still_parsed(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    subs = _submissions(
        [
            ("4", "2026-07-01", "0000000001-26-000001", "good.xml"),
            ("4", "2026-06-25", "0000000001-26-000002", "bad.xml"),
        ]
    )
    client = _FakeClient(
        responses={"000000000126000001": _form4_xml()},
        raise_for={"000000000126000002"},
    )

    result = insider_fetch.get_insider_transactions("1", subs, client, today=_TODAY)

    assert result is not None
    assert result["filings_scanned"] == 1
    assert result["filings_failed"] == 1
    assert len(result["transactions"]) == 2  # the two rows from the good filing


def test_xsl_prefixed_primary_document_resolves_to_basename(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    subs = _submissions(
        [("4", "2026-07-01", "0000320193-26-000012", "xslF345X03/wk-form4_1.xml")]
    )
    client = _FakeClient(responses={"000032019326000012": _form4_xml()})

    result = insider_fetch.get_insider_transactions("320193", subs, client, today=_TODAY)

    assert result is not None
    assert len(client.calls) == 1
    assert client.calls[0].endswith("/wk-form4_1.xml")
    assert "xslF345X03" not in client.calls[0]
    assert "/320193/" in client.calls[0]  # unpadded CIK segment


def test_non_xml_primary_document_is_skipped_without_network_call(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    subs = _submissions(
        [
            ("4", "2026-07-01", "0000000001-26-000001", "form4.htm"),
            ("4", "2026-06-25", "0000000001-26-000002", "good.xml"),
        ]
    )
    client = _FakeClient(responses={"000000000126000002": _form4_xml()})

    result = insider_fetch.get_insider_transactions("1", subs, client, today=_TODAY)

    assert result is not None
    assert result["filings_failed"] == 1
    assert result["filings_scanned"] == 1
    # The .htm filing must never have triggered a network call.
    assert all("000000000126000001" not in url for url in client.calls)


def test_filing_dated_after_today_is_excluded(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    subs = _submissions([("4", "2026-07-20", "0000000001-26-000001", "future.xml")])
    client = _FakeClient(responses={"000000000126000001": _form4_xml()})

    result = insider_fetch.get_insider_transactions("1", subs, client, today=_TODAY)

    assert result is None
    assert client.calls == []


def test_lookback_days_excludes_older_filings(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    subs = _submissions(
        [
            ("4", "2026-07-10", "0000000001-26-000001", "recent.xml"),
            ("4", "2025-01-01", "0000000001-26-000002", "old.xml"),  # >180d back
        ]
    )
    client = _FakeClient(responses={"000000000126000001": _form4_xml()})

    result = insider_fetch.get_insider_transactions(
        "1", subs, client, lookback_days=180, today=_TODAY
    )

    assert result is not None
    assert result["filings_scanned"] == 1
    assert all("000000000126000002" not in url for url in client.calls)


def test_max_filings_caps_fetch_count(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    rows = [
        ("4", f"2026-07-{10 - i:02d}", f"0000000001-26-00000{i}", f"f{i}.xml")
        for i in range(5)
    ]
    subs = _submissions(rows)
    responses = {row[2].replace("-", ""): _form4_xml() for row in rows}
    client = _FakeClient(responses=responses)

    result = insider_fetch.get_insider_transactions(
        "1", subs, client, max_filings=2, today=_TODAY
    )

    assert result is not None
    assert len(client.calls) <= 2
    assert result["truncated"] is True


def test_truncated_is_false_when_all_qualifying_filings_fetched(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    subs = _submissions([("4", "2026-07-01", "0000000001-26-000001", "f.xml")])
    client = _FakeClient(responses={"000000000126000001": _form4_xml()})

    result = insider_fetch.get_insider_transactions(
        "1", subs, client, max_filings=10, today=_TODAY
    )

    assert result is not None
    assert result["truncated"] is False


def test_get_insider_transactions_none_for_missing_or_empty_submissions():
    client = _FakeClient()
    assert insider_fetch.get_insider_transactions("1", None, client, today=_TODAY) is None
    assert insider_fetch.get_insider_transactions("1", {}, client, today=_TODAY) is None


def test_get_insider_transactions_none_when_no_form4_filings(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    subs = _submissions([("10-Q", "2026-07-01", "acc", "q.htm")])
    client = _FakeClient()
    assert insider_fetch.get_insider_transactions("1", subs, client, today=_TODAY) is None


def test_get_insider_transactions_never_raises_on_garbage_input(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    client = _FakeClient()
    garbage_inputs = [
        {"filings": None},
        {"filings": {"recent": None}},
        {"filings": {"recent": {"form": ["4", "4"], "filingDate": ["2026-07-01"]}}},  # ragged
        "not-a-dict",
        123,
    ]
    for garbage in garbage_inputs:
        assert insider_fetch.get_insider_transactions("1", garbage, client, today=_TODAY) is None
    # An unparseable CIK must not raise either.
    subs = _submissions([("4", "2026-07-01", "0000000001-26-000001", "f.xml")])
    assert insider_fetch.get_insider_transactions("not-a-cik", subs, client, today=_TODAY) is None


# ---------------------------------------------------------------------------
# get_insider_transactions: per-filing cache
# ---------------------------------------------------------------------------


def test_second_call_hits_cache_no_network_call(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    subs = _submissions([("4", "2026-07-01", "0000000001-26-000001", "f.xml")])
    client = _FakeClient(responses={"000000000126000001": _form4_xml()})

    first = insider_fetch.get_insider_transactions("1", subs, client, today=_TODAY)
    assert len(client.calls) == 1

    second = insider_fetch.get_insider_transactions("1", subs, client, today=_TODAY)
    assert len(client.calls) == 1  # served entirely from cache
    assert second["transactions"] == first["transactions"]


def test_no_cache_true_forces_refetch(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    subs = _submissions([("4", "2026-07-01", "0000000001-26-000001", "f.xml")])
    client = _FakeClient(responses={"000000000126000001": _form4_xml()})

    insider_fetch.get_insider_transactions("1", subs, client, today=_TODAY)
    assert len(client.calls) == 1

    insider_fetch.get_insider_transactions("1", subs, client, no_cache=True, today=_TODAY)
    assert len(client.calls) == 2


def test_cache_file_written_under_form4_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "RAW_DIR", str(tmp_path))
    subs = _submissions([("4", "2026-07-01", "0000000001-26-000001", "f.xml")])
    client = _FakeClient(responses={"000000000126000001": _form4_xml()})

    insider_fetch.get_insider_transactions("1", subs, client, today=_TODAY)

    cache_file = tmp_path / "form4" / "form4_000000000126000001.json"
    assert cache_file.exists()


# ---------------------------------------------------------------------------
# signals.insider: verdict rules
# ---------------------------------------------------------------------------


def _txn(date_str, code, person, shares=100.0, price=10.0, derivative=False,
         is_director=False, is_officer=False, is_ten_percent=False, officer_title=None,
         shares_owned_after=None):
    return {
        "date": date_str,
        "filed": date_str,
        "person": person,
        "is_director": is_director,
        "is_officer": is_officer,
        "is_ten_percent": is_ten_percent,
        "officer_title": officer_title,
        "code": code,
        "shares": shares,
        "price": price,
        "direction": "A" if code == "P" else "D",
        "derivative": derivative,
        "accession": "acc",
        "shares_owned_after": shares_owned_after,
    }


def _fetched(transactions, truncated=False):
    return {"transactions": transactions, "filings_scanned": 1, "filings_failed": 0,
            "lookback_days": 180, "source": "SEC Form 4", "truncated": truncated}


def test_cluster_buy_is_strong_alim():
    txns = [
        _txn("2026-07-01", "P", "Alice", shares=1000, price=10.0),
        _txn("2026-07-02", "P", "Bob", shares=500, price=12.0),
    ]
    activity = insider_signal.detect_insider_activity(_fetched(txns), today=_TODAY)
    assert activity["verdict"] == "GÜÇLÜ ALIM"
    assert activity["severity"] == "positive"
    assert activity["cluster_buy"] is True
    assert activity["buyers"] == ["Alice", "Bob"]


def test_single_buy_is_alim():
    txns = [_txn("2026-07-01", "P", "Alice", shares=1000, price=10.0)]
    activity = insider_signal.detect_insider_activity(_fetched(txns), today=_TODAY)
    assert activity["verdict"] == "ALIM"
    assert activity["severity"] == "positive"
    assert activity["cluster_buy"] is False


# ---------------------------------------------------------------------------
# signals.insider: derivative P/S rows must never enter the buy/sell
# aggregates, only the separately-reported derivative_buy_count/
# derivative_sell_count -- see the module docstring's "derivative rows never
# enter the buy/sell aggregates" section.
# ---------------------------------------------------------------------------


def test_derivative_buy_does_not_affect_buy_tallies():
    txns = [_txn("2026-07-01", "P", "Alice", shares=500, price=100.0, derivative=True)]
    activity = insider_signal.detect_insider_activity(_fetched(txns), today=_TODAY)

    assert activity["buy_count"] == 0
    assert activity["buy_shares"] == 0.0
    assert activity["buy_value"] == 0.0
    assert activity["net_value"] == 0.0
    assert activity["buyers"] == []
    assert activity["derivative_buy_count"] == 1
    assert activity["derivative_sell_count"] == 0


def test_two_derivative_buyers_do_not_trigger_cluster_buy():
    """The regression that matters most: two distinct people each with only
    a derivative P row must NOT read as a cluster buy."""
    txns = [
        _txn("2026-07-01", "P", "Alice", shares=500, price=100.0, derivative=True),
        _txn("2026-07-02", "P", "Bob", shares=300, price=100.0, derivative=True),
    ]
    activity = insider_signal.detect_insider_activity(_fetched(txns), today=_TODAY)

    assert activity["cluster_buy"] is False
    assert activity["verdict"] == "NÖTR"
    assert activity["severity"] == "neutral"
    assert activity["derivative_buy_count"] == 2


def test_derivative_sell_does_not_affect_sell_tallies():
    txns = [_txn("2026-07-01", "S", "Alice", shares=500, price=100.0, derivative=True)]
    activity = insider_signal.detect_insider_activity(_fetched(txns), today=_TODAY)

    assert activity["sell_count"] == 0
    assert activity["sell_value"] == 0.0
    assert activity["cluster_sell"] is False
    assert activity["derivative_sell_count"] == 1
    assert activity["derivative_buy_count"] == 0


def test_derivative_counts_are_zero_when_no_derivative_open_market_rows():
    txns = [_txn("2026-07-01", "P", "Alice", shares=100, price=10.0)]
    activity = insider_signal.detect_insider_activity(_fetched(txns), today=_TODAY)
    assert activity["derivative_buy_count"] == 0
    assert activity["derivative_sell_count"] == 0
    assert "türev" not in activity["note"]


def test_transaction_count_and_recent_still_include_derivative_rows():
    txns = [
        _txn("2026-07-01", "P", "Alice", shares=100, price=10.0),
        _txn("2026-07-02", "P", "Bob", shares=500, price=100.0, derivative=True),
    ]
    activity = insider_signal.detect_insider_activity(_fetched(txns), today=_TODAY)
    assert activity["transaction_count"] == 2  # coverage figure, not signal
    derivative_rows = [r for r in activity["recent"] if r["derivative"]]
    assert len(derivative_rows) == 1
    assert derivative_rows[0]["person"] == "Bob"


def test_note_carries_derivative_exclusion_clause_when_present():
    txns = [
        _txn("2026-07-01", "P", "Alice", shares=100, price=10.0),
        _txn("2026-07-02", "P", "Bob", shares=500, price=100.0, derivative=True),
        _txn("2026-07-03", "S", "Carol", shares=200, price=100.0, derivative=True),
    ]
    activity = insider_signal.detect_insider_activity(_fetched(txns), today=_TODAY)
    assert "2 türev" in activity["note"]
    assert "toplamlara dahil edilmedi" in activity["note"]


def test_mixed_non_derivative_and_derivative_buy_counts_separately():
    txns = [
        _txn("2026-07-01", "P", "Alice", shares=100, price=10.0, derivative=False),
        _txn("2026-07-02", "P", "Alice", shares=500, price=100.0, derivative=True),
    ]
    activity = insider_signal.detect_insider_activity(_fetched(txns), today=_TODAY)
    assert activity["buy_count"] == 1
    assert activity["derivative_buy_count"] == 1


def test_sells_only_is_satis_agirlikli_neutral():
    """The bug this whole recalibration fixes: a single seller (or several)
    with no stake-disposal evidence must NOT read as bearish."""
    txns = [_txn("2026-07-01", "S", "Alice", shares=1000, price=10.0)]
    activity = insider_signal.detect_insider_activity(_fetched(txns), today=_TODAY)
    assert activity["verdict"] == "SATIŞ AĞIRLIKLI"
    assert activity["severity"] == "neutral"


def test_regression_zero_buys_several_sells_small_stake_is_satis_agirlikli_not_yogun():
    """The exact NVDA/SOFI/JPM-shaped regression: 0 buys, several sells, but
    each seller only let go of a small (~3%) slice of their position. Must
    stay neutral, not escalate to the bearish label."""
    txns = [
        _txn("2026-07-01", "S", "Alice", shares=3000, price=100.0, shares_owned_after=97000),  # 3.0%
        _txn("2026-07-02", "S", "Bob", shares=3000, price=100.0, shares_owned_after=97000),      # 3.0%
        _txn("2026-07-03", "S", "Carol", shares=3000, price=100.0, shares_owned_after=97000),    # 3.0%
    ]
    activity = insider_signal.detect_insider_activity(_fetched(txns), today=_TODAY)
    assert activity["buy_count"] == 0
    assert activity["cluster_sell"] is True
    assert activity["median_stake_sold_pct"] == pytest.approx(3.0)
    assert activity["verdict"] == "SATIŞ AĞIRLIKLI"
    assert activity["severity"] == "neutral"


def test_cluster_sell_median_stake_at_threshold_is_yogun_satis():
    # sold=250, remaining=750 -> stake_sold_pct = 25.0 == _HEAVY_SELL_STAKE_PCT
    txns = [
        _txn("2026-07-01", "S", "Alice", shares=250, price=100.0, shares_owned_after=750),
        _txn("2026-07-02", "S", "Bob", shares=250, price=100.0, shares_owned_after=750),
    ]
    activity = insider_signal.detect_insider_activity(_fetched(txns), today=_TODAY)
    assert activity["median_stake_sold_pct"] == pytest.approx(25.0)
    assert activity["verdict"] == "YOĞUN SATIŞ"
    assert activity["severity"] == "negative"


def test_cluster_sell_median_stake_above_threshold_is_yogun_satis():
    # sold=410, remaining=590 -> stake_sold_pct = 41.0
    txns = [
        _txn("2026-07-01", "S", "Alice", shares=410, price=100.0, shares_owned_after=590),
        _txn("2026-07-02", "S", "Bob", shares=410, price=100.0, shares_owned_after=590),
    ]
    activity = insider_signal.detect_insider_activity(_fetched(txns), today=_TODAY)
    assert activity["median_stake_sold_pct"] == pytest.approx(41.0)
    assert activity["verdict"] == "YOĞUN SATIŞ"
    assert activity["severity"] == "negative"
    assert "%41,0" in activity["note"]


def test_cluster_sell_median_stake_just_below_threshold_is_satis_agirlikli():
    # sold=249, remaining=751 -> stake_sold_pct = 24.9, just under the 25.0 bar
    txns = [
        _txn("2026-07-01", "S", "Alice", shares=249, price=100.0, shares_owned_after=751),
        _txn("2026-07-02", "S", "Bob", shares=249, price=100.0, shares_owned_after=751),
    ]
    activity = insider_signal.detect_insider_activity(_fetched(txns), today=_TODAY)
    assert activity["median_stake_sold_pct"] == pytest.approx(24.9)
    assert activity["verdict"] == "SATIŞ AĞIRLIKLI"
    assert activity["severity"] == "neutral"


def test_cluster_sell_with_no_parseable_stake_falls_through_to_satis_agirlikli():
    """Materiality can't be established without a known median -- must not
    guess bearish."""
    txns = [
        _txn("2026-07-01", "S", "Alice", shares=100000, price=100.0),  # huge $ value, no stake data
        _txn("2026-07-02", "S", "Bob", shares=100000, price=100.0),
    ]
    activity = insider_signal.detect_insider_activity(_fetched(txns), today=_TODAY)
    assert activity["cluster_sell"] is True
    assert activity["median_stake_sold_pct"] is None
    assert activity["verdict"] == "SATIŞ AĞIRLIKLI"
    assert activity["severity"] == "neutral"


def test_buy_outsold_by_value_is_karisik_neutral():
    txns = [
        _txn("2026-07-01", "P", "Alice", shares=10, price=10.0),     # value 100
        _txn("2026-07-02", "S", "Bob", shares=1000, price=100.0),    # value 100,000
    ]
    activity = insider_signal.detect_insider_activity(_fetched(txns), today=_TODAY)
    assert activity["buy_count"] == 1
    assert activity["buy_value"] < activity["sell_value"]
    assert activity["verdict"] == "KARIŞIK"
    assert activity["severity"] == "neutral"


def test_stake_sold_pct_hand_computed():
    txns = [_txn("2026-07-01", "S", "Alice", shares=250, price=10.0, shares_owned_after=750)]
    activity = insider_signal.detect_insider_activity(_fetched(txns), today=_TODAY)
    detail = next(d for d in activity["sellers_detail"] if d["person"] == "Alice")
    assert detail["sold_shares"] == pytest.approx(250.0)
    assert detail["shares_owned_after"] == pytest.approx(750.0)
    assert detail["stake_sold_pct"] == pytest.approx(25.0)


def test_stake_sold_pct_none_when_shares_owned_after_missing():
    txns = [_txn("2026-07-01", "S", "Alice", shares=250, price=10.0, shares_owned_after=None)]
    activity = insider_signal.detect_insider_activity(_fetched(txns), today=_TODAY)
    detail = next(d for d in activity["sellers_detail"] if d["person"] == "Alice")
    assert detail["stake_sold_pct"] is None


def test_stake_added_pct_hand_computed():
    # bought 300, remaining (post-buy) 1200 -> stake_added_pct = 300/1200*100 = 25.0
    txns = [_txn("2026-07-01", "P", "Alice", shares=300, price=10.0, shares_owned_after=1200)]
    activity = insider_signal.detect_insider_activity(_fetched(txns), today=_TODAY)
    detail = next(d for d in activity["buyers_detail"] if d["person"] == "Alice")
    assert detail["bought_shares"] == pytest.approx(300.0)
    assert detail["stake_added_pct"] == pytest.approx(25.0)


def test_median_stake_sold_pct_ignores_unknown_sellers():
    txns = [
        _txn("2026-07-01", "S", "Alice", shares=100, price=10.0, shares_owned_after=900),   # 10.0%
        _txn("2026-07-02", "S", "Bob", shares=300, price=10.0, shares_owned_after=700),      # 30.0%
        _txn("2026-07-03", "S", "Carol", shares=100, price=10.0, shares_owned_after=None),   # unknown
    ]
    activity = insider_signal.detect_insider_activity(_fetched(txns), today=_TODAY)
    # Median of the two KNOWN sellers only (10.0, 30.0) -> 20.0, Carol excluded.
    assert activity["median_stake_sold_pct"] == pytest.approx(20.0)


def test_truncated_propagates_from_fetched_to_activity_and_summary():
    txns = [_txn("2026-07-01", "P", "Alice", shares=100, price=10.0)]
    truncated_activity = insider_signal.detect_insider_activity(_fetched(txns, truncated=True), today=_TODAY)
    complete_activity = insider_signal.detect_insider_activity(_fetched(txns, truncated=False), today=_TODAY)

    assert truncated_activity["truncated"] is True
    assert complete_activity["truncated"] is False

    assert insider_signal.summarize_insider(truncated_activity).endswith(" (kısmi)")
    assert not insider_signal.summarize_insider(complete_activity).endswith(" (kısmi)")


def test_awards_only_is_notr_and_compensation_only():
    txns = [_txn("2026-07-01", "A", "Alice", shares=500, price=0.0)]
    activity = insider_signal.detect_insider_activity(_fetched(txns), today=_TODAY)
    assert activity["verdict"] == "NÖTR"
    assert activity["severity"] == "neutral"
    assert activity["compensation_only"] is True
    assert activity["buy_count"] == 0
    assert activity["sell_count"] == 0


def test_unpriced_transactions_skip_value_but_count_shares():
    txns = [
        _txn("2026-07-01", "P", "Alice", shares=1000, price=None),
        _txn("2026-07-02", "P", "Alice", shares=500, price=10.0),
    ]
    activity = insider_signal.detect_insider_activity(_fetched(txns), today=_TODAY)
    assert activity["buy_shares"] == pytest.approx(1500.0)
    assert activity["buy_value"] == pytest.approx(5000.0)  # only the priced row


def test_point_in_time_filter_excludes_future_transaction():
    txns = [
        _txn("2026-07-01", "P", "Alice", shares=100, price=10.0),
        _txn("2026-07-20", "P", "Bob", shares=100, price=10.0),  # after _TODAY
    ]
    activity = insider_signal.detect_insider_activity(_fetched(txns), today=_TODAY)
    assert activity["buy_count"] == 1
    assert activity["buyers"] == ["Alice"]


def test_lookback_days_window_excludes_old_transactions():
    txns = [
        _txn("2026-07-01", "P", "Alice", shares=100, price=10.0),
        _txn("2024-01-01", "P", "Bob", shares=100, price=10.0),  # way outside window
    ]
    activity = insider_signal.detect_insider_activity(_fetched(txns), lookback_days=180, today=_TODAY)
    assert activity["buy_count"] == 1


def test_unknown_code_is_surfaced_not_dropped():
    txns = [_txn("2026-07-01", "Z", "Alice", shares=100, price=10.0)]
    activity = insider_signal.detect_insider_activity(_fetched(txns), today=_TODAY)
    assert activity is not None
    assert activity["recent"][0]["category"] == "other"
    assert "Z" in activity["recent"][0]["label"]


def test_detect_insider_activity_none_for_empty_or_missing():
    assert insider_signal.detect_insider_activity(None) is None
    assert insider_signal.detect_insider_activity({}) is None
    assert insider_signal.detect_insider_activity({"transactions": []}) is None


def test_role_label_precedence():
    officer_with_title = _txn("2026-07-01", "P", "A", is_officer=True, officer_title="CFO")
    officer_no_title = _txn("2026-07-01", "P", "B", is_officer=True)
    director = _txn("2026-07-01", "P", "C", is_director=True)
    ten_pct = _txn("2026-07-01", "P", "D", is_ten_percent=True)
    nobody = _txn("2026-07-01", "P", "E")

    activity = insider_signal.detect_insider_activity(
        _fetched([officer_with_title, officer_no_title, director, ten_pct, nobody]),
        today=_TODAY,
    )
    roles = {r["person"]: r["role"] for r in activity["recent"]}
    assert roles["A"] == "CFO"
    assert roles["B"] == "Yönetici"
    assert roles["C"] == "Yönetim Kurulu Üyesi"
    assert roles["D"] == "%10 Ortak"
    assert roles["E"] == "İçeriden"


def test_summarize_insider_none_is_yok():
    assert insider_signal.summarize_insider(None) == "yok"
    assert insider_signal.summarize_insider({}) == "yok"


def test_summarize_insider_never_contains_none_or_nan():
    cases = [
        None,
        {},
        insider_signal.detect_insider_activity(_fetched([_txn("2026-07-01", "P", "Alice", shares=100, price=10.0)]), today=_TODAY),
        insider_signal.detect_insider_activity(_fetched([_txn("2026-07-01", "S", "Alice", shares=100, price=None)]), today=_TODAY),
        insider_signal.detect_insider_activity(_fetched([_txn("2026-07-01", "A", "Alice", shares=100, price=0.0)]), today=_TODAY),
        insider_signal.detect_insider_activity(
            _fetched(
                [
                    _txn("2026-07-01", "S", "Alice", shares=100, price=10.0, shares_owned_after=900),
                    _txn("2026-07-02", "S", "Bob", shares=300, price=10.0, shares_owned_after=700),
                ]
            ),
            today=_TODAY,
        ),
        insider_signal.detect_insider_activity(
            _fetched(
                [
                    _txn("2026-07-01", "P", "Alice", shares=10, price=10.0),
                    _txn("2026-07-02", "S", "Bob", shares=1000, price=100.0),
                ]
            ),
            today=_TODAY,
        ),
    ]
    for case in cases:
        summary = insider_signal.summarize_insider(case)
        assert "None" not in summary
        assert "nan" not in summary.lower()


def test_note_never_contains_none_or_nan():
    cases = [
        _fetched([_txn("2026-07-01", "S", "Alice", shares=100, price=None)]),
        _fetched(
            [
                _txn("2026-07-01", "S", "Alice", shares=100, price=10.0, shares_owned_after=900),
                _txn("2026-07-02", "S", "Bob", shares=300, price=10.0, shares_owned_after=700),
            ]
        ),
        _fetched(
            [
                _txn("2026-07-01", "S", "Alice", shares=250, price=10.0, shares_owned_after=750),
                _txn("2026-07-02", "S", "Bob", shares=250, price=10.0, shares_owned_after=750),
            ]
        ),
        _fetched(
            [
                _txn("2026-07-01", "P", "Alice", shares=10, price=10.0),
                _txn("2026-07-02", "S", "Bob", shares=1000, price=100.0),
            ]
        ),
    ]
    for fetched in cases:
        activity = insider_signal.detect_insider_activity(fetched, today=_TODAY)
        assert "None" not in activity["note"]
        assert "nan" not in activity["note"].lower()
