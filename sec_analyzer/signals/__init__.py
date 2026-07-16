"""Deterministic qualitative signals derived from SEC filing metadata.

Unlike ``sec_analyzer.normalize.red_flags`` (which reasons over *numeric*
XBRL facts), this package extracts *event/qualitative* signals from filing
metadata that SEC already provides in structured form -- no filing document
is ever downloaded or parsed, and no LLM is involved.
"""
