"""Phase H9 -- parse the DGHS HEOC dengue dashboard's embedded Highcharts data.

The dashboard (https://dashboard.dghs.gov.bd/pages/heoc_dengue_v1.php) has no
API and no download button: the numbers live inside inline `Highcharts.chart(
'<id>', {...})` JavaScript blocks. The finest geography it publishes is 10
reporting units -- 7 divisions, with Dhaka division split into "Dhaka (Out of
CC)", DNCC and DSCC -- so that is the finest unit any H9 model can use. There is
no district, upazila or ward breakdown anywhere on the page.

These are *admitted* (hospitalised) cases, not infections, and are most likely
attributed to the reporting hospital's location rather than the patient's home.
"""

from __future__ import annotations

import re

import pandas as pd

_CASES_CHART = "div_city_cor_case_in_year"
_DEATHS_CHART = "div_city_cor_death_in_year"

# DGHS spellings -> geoBoundaries spellings (ADM1 shapeName), including
# geoBoundaries' own "Rajshani" typo.
_TO_GEOBOUNDARIES = {"Barishal": "Barisal", "Chattogram": "Chittagong", "Rajshahi": "Rajshani"}

_DHAKA_UNITS = ("Dhaka (Out of CC)", "DNCC", "DSCC")


def extract_chart(html: str, chart_id: str) -> dict:
    """Categories and named series of one `Highcharts.chart('<chart_id>', ...)`
    block. Returns empty lists if the chart is not on the page."""
    block = re.search(
        rf"Highcharts\.chart\(['\"]?{re.escape(chart_id)}['\"]?.*?(?=Highcharts\.chart\(|</script>)",
        html,
        re.DOTALL,
    )
    if block is None:
        return {"categories": [], "series": []}
    text = block.group(0)

    categories_match = re.search(r"categories\s*:\s*(\[[^\]]*\])", text)
    categories = re.findall(r'["\']([^"\']*)["\']', categories_match.group(1)) if categories_match else []

    series = [
        {"name": name, "data": [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", data)]}
        for name, data in re.findall(
            r'\{[^{}]*name\s*:\s*["\']([^"\']+)["\'][^{}]*data\s*:\s*(\[[^\]]*\])[^{}]*\}', text, re.DOTALL
        )
    ]
    return {"categories": categories, "series": series}


def _clean_unit_name(name: str) -> str:
    """DGHS pads some names with trailing spaces and puts a double space
    inside "Dhaka  (Out of CC)"."""
    return re.sub(r"\s+", " ", name).strip()


def _series_by_unit(html: str, chart_id: str) -> pd.Series:
    chart = extract_chart(html, chart_id)
    if not chart["categories"] or not chart["series"]:
        raise ValueError(f"chart {chart_id!r} not found or empty -- the dashboard layout may have changed")
    units = [_clean_unit_name(c) for c in chart["categories"]]
    values = chart["series"][0]["data"]
    if len(units) != len(values):
        raise ValueError(f"chart {chart_id!r}: {len(units)} categories but {len(values)} values")
    return pd.Series([int(v) for v in values], index=units)


def parse_unit_counts(html: str) -> pd.DataFrame:
    """Year-to-date admitted dengue cases and deaths for each of the 10 DGHS
    reporting units (index = unit name, columns = cases, deaths)."""
    counts = pd.DataFrame(
        {"cases": _series_by_unit(html, _CASES_CHART), "deaths": _series_by_unit(html, _DEATHS_CHART)}
    )
    counts.index.name = "unit"
    return counts


def to_divisions(counts: pd.DataFrame) -> pd.DataFrame:
    """Collapse the 10 reporting units to the 8 administrative divisions
    (Dhaka = Out of CC + DNCC + DSCC) and use geoBoundaries' spellings, so the
    result joins to the ADM1 boundary file."""
    missing = [u for u in _DHAKA_UNITS if u not in counts.index]
    if missing:
        raise ValueError(f"cannot merge Dhaka division, missing units: {missing}")
    dhaka = counts.loc[list(_DHAKA_UNITS)].sum()
    divisions = counts.drop(index=list(_DHAKA_UNITS))
    divisions = pd.concat([divisions, dhaka.to_frame("Dhaka").T])
    divisions.index = [_TO_GEOBOUNDARIES.get(name, name) for name in divisions.index]
    divisions.index.name = "division"
    return divisions.sort_index().astype(int)
