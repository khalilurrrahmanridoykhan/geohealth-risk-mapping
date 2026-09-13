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

**Phase H6 done** — a real YOLO11n trained on a real ~10cm/px drone survey of a
canal-side Dhaka neighborhood, against real Microsoft Building Footprints. This phase
makes a documented, upfront scope call: PLAN.md names specific breeding-site classes
(containers, tyres, stagnant pools, rooftop tanks) that no labeled dataset covers for
this real imagery — hand-labeling them would mean fabricating "ground truth" with
nothing real to check it against, which this project avoids at every phase. Building/
structure detection is used instead, as the real precursor signal (dense settlement near
standing water) the breeding-site classes would eventually plug into. A real, algorithmic
stagnant-water flag caught and fixed its own bug along the way — the honest kind of
result this repo tries to surface rather than hide. See [`RESULTS.md`](RESULTS.md) for
the numbers and [`PLAN.md`](PLAN.md) section 6 for what's next (H7 — EO foundation
models: Prithvi-EO-2.0, TerraMind, Clay via TerraTorch).

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
