import pytest

from src.dghs import extract_chart, parse_unit_counts, to_divisions

_UNITS = ["Barishal ", "Chattogram ", "Dhaka  (Out of CC)", "DNCC", "DSCC", "Khulna ", "Mymensingh ", "Rajshahi ", "Rangpur ", "Sylhet "]


def _chart(chart_id: str, categories: list[str], name: str, data: list[int]) -> str:
    cats = ",".join(f'"{c}"' for c in categories)
    return f"""<script>
    Highcharts.chart('{chart_id}', {{
        chart: {{ zoomType: 'xy' }},
        xAxis: {{ categories: [{cats}], crosshair: true }},
        series: [
            {{color: '#337ab7', type: 'column', name: '{name}', data:{data}}}
        ]
    }});
</script>"""


def _page(cases: list[int], deaths: list[int]) -> str:
    return (
        _chart("div_city_cor_case_in_year", _UNITS, "Admitted", cases)
        + _chart("div_city_cor_death_in_year", _UNITS, "Death", deaths)
    )


CASES = [8467, 8672, 9997, 6980, 7317, 9903, 3366, 3638, 1762, 277]
DEATHS = [13, 16, 10, 23, 53, 32, 18, 8, 4, 1]


def test_extract_chart_reads_categories_and_series():
    chart = extract_chart(_page(CASES, DEATHS), "div_city_cor_case_in_year")
    assert chart["categories"][:3] == ["Barishal ", "Chattogram ", "Dhaka  (Out of CC)"]
    assert chart["series"] == [{"name": "Admitted", "data": [float(c) for c in CASES]}]


def test_extract_chart_does_not_bleed_into_the_next_chart():
    # the deaths chart follows the cases chart on the page; its numbers must not leak in
    chart = extract_chart(_page(CASES, DEATHS), "div_city_cor_case_in_year")
    assert len(chart["series"]) == 1


def test_extract_chart_returns_empty_for_an_unknown_chart():
    assert extract_chart(_page(CASES, DEATHS), "nope") == {"categories": [], "series": []}


def test_parse_unit_counts_normalises_padded_names():
    counts = parse_unit_counts(_page(CASES, DEATHS))
    assert list(counts.index) == [u.strip().replace("  ", " ") for u in _UNITS]
    assert counts.loc["Dhaka (Out of CC)", "cases"] == 9997
    assert counts.loc["DSCC", "deaths"] == 53
    assert counts["cases"].sum() == 60379


def test_parse_unit_counts_raises_when_the_dashboard_layout_changes():
    with pytest.raises(ValueError, match="not found"):
        parse_unit_counts("<html>no charts here</html>")


def test_parse_unit_counts_raises_on_length_mismatch():
    page = _page(CASES[:-1], DEATHS)
    with pytest.raises(ValueError, match="categories"):
        parse_unit_counts(page)


def test_to_divisions_merges_dhaka_and_uses_geoboundaries_spellings():
    divisions = to_divisions(parse_unit_counts(_page(CASES, DEATHS)))
    assert len(divisions) == 8
    assert divisions.loc["Dhaka", "cases"] == 9997 + 6980 + 7317
    assert divisions.loc["Dhaka", "deaths"] == 10 + 23 + 53
    assert set(divisions.index) == {
        "Barisal", "Chittagong", "Dhaka", "Khulna", "Mymensingh", "Rajshani", "Rangpur", "Sylhet"
    }  # exactly geoBoundaries' ADM1 shapeName values
    assert divisions["cases"].sum() == sum(CASES)  # merging must not lose or add cases


def test_to_divisions_raises_if_a_dhaka_unit_is_missing():
    counts = parse_unit_counts(_page(CASES, DEATHS)).drop(index="DNCC")
    with pytest.raises(ValueError, match="DNCC"):
        to_divisions(counts)
