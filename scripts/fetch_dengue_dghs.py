"""Phases H9/H10 -- snapshots the DGHS HEOC dengue dashboard's division / city-
corporation admitted-case counts (H9) and its division-by-week series (H10)
to small, dated CSVs.

The dashboard reports year-to-date totals that change daily and keeps no
history, so an unrecorded run isn't reproducible: the snapshot is committed
(tiny public government statistics) under data/snapshots/, named by date.
DGHS returns 403 to requests without a browser-like User-Agent.

Usage:
    python scripts/fetch_dengue_dghs.py
"""

from __future__ import annotations

import sys
import urllib.request
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dghs import parse_unit_counts, parse_weekly_division  # noqa: E402

DASHBOARD_URL = "https://dashboard.dghs.gov.bd/pages/heoc_dengue_v1.php"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data" / "snapshots"


def fetch_dashboard_html(url: str = DASHBOARD_URL, timeout: int = 60) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="ignore")


def save_snapshots(html: str, stamp: str, out_dir: Path = SNAPSHOT_DIR) -> list[Path]:
    """Both tables come from the same page fetch, so they always agree."""
    out_dir.mkdir(parents=True, exist_ok=True)
    units_path = out_dir / f"dghs_dengue_units_{stamp}.csv"
    weekly_path = out_dir / f"dghs_dengue_weekly_division_{stamp}.csv"
    parse_unit_counts(html).to_csv(units_path)
    parse_weekly_division(html).to_csv(weekly_path)
    return [units_path, weekly_path]


def main() -> None:
    html = fetch_dashboard_html()
    paths = save_snapshots(html, date.today().isoformat())
    counts = parse_unit_counts(html)
    weekly = parse_weekly_division(html)
    print(counts.to_string())
    print(f"total admitted cases (units): {counts['cases'].sum():,}; weekly table: {weekly.shape[0]} weeks, {weekly.values.sum():,} cases")
    for path in paths:
        print(f"saved {path}")


if __name__ == "__main__":
    main()
