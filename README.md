# Geospatial AI for Public Health — Dengue/Flood Risk Mapping (Bangladesh)

Turning free satellite imagery into a ward-level dengue and flood risk map for
Bangladesh: an AI model (U-Net / EO foundation models) segments water, built-up area,
and flooding from Sentinel-1/2, the result is vectorized and joined with real DGHS
dengue case data, and a spatial model turns that into a risk score a health department
can act on.

This is a phased learning-and-build project, not a single script — see
[`PLAN.md`](PLAN.md) for the full roadmap (13 phases, H0 through H13), the model/skill
coverage matrix, data sources, and the ethics rules every phase follows.

## Why this exists

Bangladesh has severe annual dengue outbreaks and monsoon flooding. Satellite imagery
(free, public, updated every few days) can show standing water, built-up density, and
land-surface temperature at ward level — real predictors of disease risk — but turning
pixels into a decision-ready map takes the same model-selection skill demonstrated in
[`best-model-for-satellite-imagery`](https://github.com/khalilurrrahmanridoykhan/best-model-for-satellite-imagery)
(segmentation vs. detection), now pointed at a real public-health question instead of a
benchmark dataset.

## Current status

**Phase H10 done (reformulated to what the public data supports)** — a division-level,
two-week-ahead dengue admissions outlook, back-tested with a rolling-origin scheme inside
2026 (real DGHS division-by-week counts + NASA POWER weather; **not** ward-level, **not** a
held-out season — neither exists publicly). Ridge + XGBoost (SHAP) beat "no change" at 2 and
4 weeks ahead (RMSE −24% and −39%) but not at 1 week; the static density/water layers added
nothing; weather helps only at 4 weeks and is confounded with season in a single year; the
ranking is essentially last week's ranking; and the nominal-80% intervals covered only 61%.
The result is three loose tiers of divisions, not eight ranks. See
[`docs/HOW_TO_READ.md`](docs/HOW_TO_READ.md), [`RESULTS.md`](RESULTS.md) and
[`notebooks/11_risk_scoring.ipynb`](notebooks/11_risk_scoring.ipynb).

**Phase H9 done** — spatial epidemiology on the 8 divisions (the finest public dengue
geography): Poisson was invalid (dispersion 2,555–4,545), and neither population density nor
surface-water occurrence is distinguishable from no effect at n = 8. See
[`notebooks/10_spatial_epidemiology.ipynb`](notebooks/10_spatial_epidemiology.ipynb).

Earlier phases (H0–H8) — see [`RESULTS.md`](RESULTS.md); [`PLAN.md`](PLAN.md) section 6
covers what's next.

## Repo layout

```
PLAN.md           # the full phased roadmap -- start here
data/
├── snapshots/    # small dated public statistics (DGHS dengue counts, NASA POWER weather), committed
├── raw/          # downloaded satellite scenes + boundary files (gitignored)
└── processed/    # derived rasters/vectors (gitignored, reproducible from notebooks/)
notebooks/        # one notebook per phase, in order
src/              # shared, unit-tested code the notebooks import
tests/            # pytest tests, using synthetic/small data (no download needed)
scripts/          # data-fetching and pipeline scripts
docs/             # plain-language guides (how to read the dengue outlook)
```

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest tests/ -v
```

Notebooks pull real Sentinel-1/2 imagery from the Microsoft Planetary Computer STAC
catalog — free, no account or API key needed for search or download.

## Rules this project follows

One branch → one PR (`Phase X — <name>`) → merge → tag `phase-x`. `main` always runs
end-to-end on at least a small sample AOI. Free tiers only (Planetary Computer / Google
Earth Engine for data, Colab/Kaggle for GPU training). Spatial (block) cross-validation
always, never a plain random split. See [`PLAN.md`](PLAN.md) section 3 and 10 for the
full rules and the ethics guardrails around any settlement/poverty-adjacent output.

## License

Apache-2.0
