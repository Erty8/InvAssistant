"""sec_analyzer.normalize: turn raw SEC companyfacts JSON into tidy time series.

This package has no dependency on network I/O, caching, or configuration --
it operates purely on already-parsed Python dicts (the JSON returned by
``sec_analyzer.fetch.companyfacts.get_company_facts``), which keeps it fully
unit-testable without touching SEC EDGAR.
"""
