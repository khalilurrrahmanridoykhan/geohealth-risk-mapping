"""Phase H9 -- snapshots the DGHS HEOC dengue dashboard's division / city-
corporation admitted-case counts to a small, dated CSV.

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

from src.dghs import parse_unit_counts  # noqa: E402

DASHBOARD_URL = "https://dashboard.dghs.gov.bd/pages/heoc_dengue_v1.php"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data" / "snapshots"


def fetch_dashboard_html(url: str = DASHBOARD_URL, timeout: int = 60) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="ignore")


def main() -> None:
    counts = parse_unit_counts(fetch_dashboard_html())
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SNAPSHOT_DIR / f"dghs_dengue_units_{date.today().isoformat()}.csv"
    counts.to_csv(out_path)
    print(counts.to_string())
    print(f"total admitted cases: {counts['cases'].sum():,}\nsaved {out_path}")


if __name__ == "__main__":
    main()
