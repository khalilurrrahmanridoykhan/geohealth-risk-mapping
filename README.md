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

**Phase H3 done** — a real U-Net (ResNet34 encoder) trained for water + built-up
segmentation at 10m resolution, with the same spatial-block holdout discipline as every
prior phase. It doesn't hit PLAN.md's target IoU, and the real reason is the phase's
most useful result: the training region is ~30% built-up while the held-out eastern
block is only ~4-5% built-up, a genuine spatial distribution shift that a random pixel
split would have hidden entirely — precisely the failure mode spatial cross-validation
exists to catch. See [`RESULTS.md`](RESULTS.md) for the numbers and
[`PLAN.md`](PLAN.md) section 6 for what's next (H4 — vectorize + zonal stats pipeline).

## Repo layout

```
PLAN.md           # the full phased roadmap -- start here
data/
├── raw/          # downloaded satellite scenes + boundary files (gitignored)
└── processed/    # derived rasters/vectors (gitignored, reproducible from notebooks/)
notebooks/        # one notebook per phase, in order
src/              # shared, unit-tested code the notebooks import
tests/            # pytest tests, using synthetic/small data (no download needed)
scripts/          # data-fetching and pipeline scripts
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
