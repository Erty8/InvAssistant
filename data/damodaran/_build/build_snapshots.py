"""Build per-year Damodaran sector snapshots for point-in-time (as-of) runs.

Offline, operator-run tool -- never imported by sec_analyzer at runtime.
Reads Damodaran's archived industry files (``pedataNN.xls`` / ``psdataNN.xls``
from https://pages.stern.nyu.edu/~adamodar/pc/archives/, downloaded by the
operator) and writes ``data/damodaran/{YEAR}/multiples.csv``, which
``valuation.damodaran.load_sector_data`` picks up for ``as_of`` dates in or
after ``YEAR`` (SPEC.md Sec.18).

Release year vs file suffix: ``pedataNN.xls`` is the January release of year
``2000 + NN + 1`` (``pedata19.xls`` -> ``2020/``), built on data to the end of
year ``NN``.

Columns, matching the top-level ``multiples.csv`` definitions:

* ``pe``  -- "Aggregate Mkt Cap/ Trailing Net Income (only money making
  firms)". The 2015-2017 releases label the same column "Aggregate Mkt Cap/
  Trailing Net Income"; the "(all firms)" column is never used.
* ``ps``  -- "Price/Sales".
* ``pfcf`` -- blank, as in the current file (Damodaran publishes none).
* ``unlevered_beta`` / ``capex_sales`` -- copied by industry name from the
  current top-level ``multiples.csv``. They are hand-set seed values, not a
  Damodaran snapshot, so every snapshot carries the same ones the live run
  uses; the as-of engine then differs from the live one only in the
  market-priced columns.

The current top-level ``multiples.csv`` is also copied to its own release
year (``--current-year``) so an as-of date in that year doesn't fall back to
the previous January's snapshot.

Usage (needs ``xlrd`` for the legacy ``.xls`` format; dev-only, not a
runtime requirement)::

    python data/damodaran/_build/build_snapshots.py ARCHIVE_DIR \\
        [--out data/damodaran] [--first 2015] [--last 2025] [--current-year 2026]

Releases before 2015 are deliberately not built: 2005-2014 use a different
P/E definition (all firms, or an older industry taxonomy the SIC aliases
don't match), so those years stay on the loader's "withhold P/E and P/S"
path instead.
"""

import argparse
import csv
import os
import shutil
import sys

_HEADER = ["industry", "pe", "ps", "pfcf", "unlevered_beta", "capex_sales"]
_TOTAL_MARKET = "total market"


def _clean(text) -> str:
    return " ".join(str(text).split())


def _read_industry_table(path: str) -> "tuple[list[str], list[list]]":
    """(header, data rows) of the "Industry Averages" sheet (or the only
    sheet in older files), starting at the row whose first cell is the
    "Industry Name" header."""
    import xlrd  # dev-only dependency, imported lazily

    book = xlrd.open_workbook(path, on_demand=True)
    names = book.sheet_names()
    sheet = book.sheet_by_name("Industry Averages") if "Industry Averages" in names else book.sheet_by_index(0)
    for r in range(min(sheet.nrows, 20)):
        row = [_clean(c) for c in sheet.row_values(r)]
        if row and row[0].lower().startswith("industry"):
            return row, [sheet.row_values(i) for i in range(r + 1, sheet.nrows)]
    raise ValueError(f"{path}: no 'Industry Name' header row found")


def _column(header: "list[str]", path: str, match) -> int:
    hits = [i for i, h in enumerate(header) if match(h)]
    if len(hits) != 1:
        raise ValueError(f"{path}: expected one matching column, found {[header[i] for i in hits]}")
    return hits[0]


def _number(value) -> "str":
    """A positive numeric cell as a 2dp string; anything else (NA, blank,
    negative/zero multiple) as an empty cell."""
    if isinstance(value, (int, float)) and value > 0:
        return f"{value:.2f}"
    return ""


def _industry_values(path: str, match) -> "dict[str, str]":
    header, rows = _read_industry_table(path)
    col = _column(header, path, match)
    values = {}
    for row in rows:
        name = _clean(row[0]) if row else ""
        if not name or _TOTAL_MARKET in name.lower():
            continue
        values[name] = _number(row[col])
    return values


def _is_pe_column(h: str) -> bool:
    return "Trailing Net Income" in h and "all firms" not in h


def _is_ps_column(h: str) -> bool:
    return h == "Price/Sales"


def build_year(archive_dir: str, year: int, seeds: "dict[str, dict]") -> "list[list[str]]":
    """Rows for the January ``year`` release, sorted by industry name."""
    suffix = f"{(year - 1) % 100:02d}"
    pe = _industry_values(os.path.join(archive_dir, f"pedata{suffix}.xls"), _is_pe_column)
    ps = _industry_values(os.path.join(archive_dir, f"psdata{suffix}.xls"), _is_ps_column)
    rows = []
    for industry in sorted(set(pe) | set(ps)):
        seed = seeds.get(industry, {})
        rows.append([
            industry, pe.get(industry, ""), ps.get(industry, ""), "",
            seed.get("unlevered_beta", ""), seed.get("capex_sales", ""),
        ])
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("archive_dir")
    parser.add_argument("--out", default=os.path.join("data", "damodaran"))
    parser.add_argument("--first", type=int, default=2015)
    parser.add_argument("--last", type=int, default=2025)
    parser.add_argument("--current-year", type=int, default=2026)
    args = parser.parse_args(argv)

    current_path = os.path.join(args.out, "multiples.csv")
    with open(current_path, encoding="utf-8", newline="") as fh:
        seeds = {row["industry"]: row for row in csv.DictReader(fh)}

    for year in range(args.first, args.last + 1):
        rows = build_year(args.archive_dir, year, seeds)
        year_dir = os.path.join(args.out, str(year))
        os.makedirs(year_dir, exist_ok=True)
        with open(os.path.join(year_dir, "multiples.csv"), "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(_HEADER)
            writer.writerows(rows)
        unseeded = sum(1 for r in rows if r[0] not in seeds)
        print(f"{year}: {len(rows)} industries, pe {sum(1 for r in rows if r[1])}, "
              f"ps {sum(1 for r in rows if r[2])}, not in current file {unseeded}")

    current_dir = os.path.join(args.out, str(args.current_year))
    os.makedirs(current_dir, exist_ok=True)
    shutil.copyfile(current_path, os.path.join(current_dir, "multiples.csv"))
    print(f"{args.current_year}: copied current multiples.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
