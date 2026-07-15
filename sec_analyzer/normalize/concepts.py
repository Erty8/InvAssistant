"""Canonical financial concepts and their us-gaap XBRL tag fallbacks.

SEC filers do not all report the same XBRL tag for the same underlying line
item -- companies change tags across fiscal years (e.g. when adopting ASC
606 revenue recognition), and different filers pick different tags for
economically-equivalent figures. To get a consistent time series per
concept, each canonical concept below maps to an *ordered* list of us-gaap
tags to try, from most to least preferred. The extraction logic in
``normalizer.py`` walks this list in order and uses the first tag that is
actually present (with at least one ``USD`` fact) in a given filer's
companyfacts document -- it does not merge values across tags, since mixing
tags for the same concept within a single time series can silently combine
incompatible definitions.

Concepts are also classified as either "flow" (income-statement /
cash-flow items that describe an activity over a period, with both a
``start`` and an ``end`` date) or "stock" (balance-sheet items that are a
snapshot at a single point in time, with only an ``end`` date). This
distinction drives period-selection heuristics in the normalizer, e.g.
verifying that an annual "flow" figure actually spans ~12 months rather
than being a stray quarter reported inside a 10-K.

Most concepts are reported in plain USD in SEC's companyfacts documents,
but per-share and share-count concepts (e.g. ``EPS``, ``SharesOutstanding``)
are reported under different XBRL unit keys (``USD/shares``, ``shares``
respectively) rather than ``USD``. ``CONCEPT_UNITS`` records, per concept,
which unit keys the normalizer should look for -- see its docstring below.
"""

from typing import Dict, List

#: Canonical concept name -> ordered list of us-gaap tags to try, most
#: preferred first. The first tag with usable data wins; later tags in the
#: list are only used as a fallback when earlier ones are absent.
CONCEPTS: Dict[str, List[str]] = {
    "Revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "NetIncome": [
        "NetIncomeLoss",
    ],
    "TotalAssets": [
        "Assets",
    ],
    "TotalLiabilities": [
        "Liabilities",
    ],
    "StockholdersEquity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "OperatingCashFlow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "Cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "CurrentAssets": [
        "AssetsCurrent",
    ],
    "CurrentLiabilities": [
        "LiabilitiesCurrent",
    ],
    "GrossProfit": [
        "GrossProfit",
    ],
    "OperatingIncome": [
        "OperatingIncomeLoss",
    ],
    "LongTermDebt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
    ],
    "CapEx": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        # Some filers (e.g. NVIDIA after ~2013) report capex combined with
        # intangible purchases under this broader tag instead.
        "PaymentsToAcquireProductiveAssets",
    ],
    "DividendsPaid": [
        "PaymentsOfDividendsCommonStock",
        "PaymentsOfDividends",
    ],
    "EPS": [
        "EarningsPerShareDiluted",
        "EarningsPerShareBasic",
    ],
    "SharesOutstanding": [
        # The dei-taxonomy cover-page tag is preferred: it's an actual
        # point-in-time share count (as of the filing's cover page), which
        # is what a market-cap calculation wants, whereas the us-gaap
        # fallbacks below are period-weighted averages. See TAG_TAXONOMY.
        "EntityCommonStockSharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "CommonStockSharesOutstanding",
    ],
    "LongTermDebtCurrent": [
        "LongTermDebtCurrent",
    ],
    "Buyback": [
        "PaymentsForRepurchaseOfCommonStock",
    ],
    "RnD": [
        "ResearchAndDevelopmentExpense",
    ],
    "SBC": [
        "ShareBasedCompensation",
    ],
    "Receivables": [
        "AccountsReceivableNetCurrent",
    ],
    "Depreciation": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
        "Depreciation",
    ],
}

#: Concepts that describe an activity over a period (both ``start`` and
#: ``end`` dates are meaningful). Income-statement and cash-flow-statement
#: line items fall in this bucket.
FLOW_CONCEPTS = {
    "Revenue", "NetIncome", "OperatingCashFlow",
    "GrossProfit", "OperatingIncome", "CapEx", "DividendsPaid",
    "EPS", "SharesOutstanding", "Buyback", "RnD", "SBC", "Depreciation",
}

#: Concepts that are a point-in-time snapshot (only ``end`` is meaningful).
#: Balance-sheet line items fall in this bucket. Defined as "everything in
#: CONCEPTS that isn't a flow concept" so new concepts default to "stock"
#: unless explicitly added to FLOW_CONCEPTS.
STOCK_CONCEPTS = set(CONCEPTS) - FLOW_CONCEPTS

#: Canonical concept name -> ordered list of acceptable XBRL unit keys, most
#: preferred first. Concepts absent from this dict default to ``["USD"]``
#: (the common case for monetary income-statement/balance-sheet/cash-flow
#: items). Per-share figures and share counts are reported under different
#: unit keys in companyfacts, so they need an explicit entry here.
CONCEPT_UNITS: Dict[str, List[str]] = {
    "EPS": ["USD/shares"],
    "SharesOutstanding": ["shares"],
}

#: Tag name -> XBRL taxonomy it's reported under, for tags that live outside
#: ``us-gaap``. Every tag not listed here defaults to ``"us-gaap"`` (the
#: common case for essentially all financial-statement line items).
#: ``EntityCommonStockSharesOutstanding`` is a "dei" (Document and Entity
#: Information) taxonomy tag -- it's the point-in-time share count SEC
#: filers report on their cover page, not a financial-statement fact, so
#: SEC's companyfacts document files it under ``facts["dei"]`` rather than
#: ``facts["us-gaap"]``. The normalizer consults this map to look each tag
#: up in the right taxonomy sub-dict (see ``normalizer._extract_concept``).
TAG_TAXONOMY: Dict[str, str] = {
    "EntityCommonStockSharesOutstanding": "dei",
}
