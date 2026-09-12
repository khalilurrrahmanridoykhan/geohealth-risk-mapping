# Results

One row per phase, per [`PLAN.md`](PLAN.md) section 3's rule: every phase ends with a
reproducible notebook and a metrics row here — updated as each phase actually runs
against real data, not projected in advance.

| Phase | What was proven | Real numbers | Notebook |
| :--- | :--- | :--- | :--- |
| H0 | Cloud STAC search, real Sentinel-2 load in native UTM, NDVI/NDWI/NDBI, both raster<->vector directions, real per-district zonal stats | Smoke tile: Dhaka/Gazipur/Narayanganj, 0.001% cloud scene (2026-02-14). Mean NDVI 0.24 / NDWI −0.26 / NDBI −0.02. 1,931 water polygons (15.1 km²) after sliver filtering. | [`01_fundamentals.ipynb`](notebooks/01_fundamentals.ipynb) |
| H1 | Reusable `get_imagery(aoi, date_range, sensor)` cloud-masked composite (SCL-based), one call for any AOI/season | Dry season (Jan-Feb): 3 scenes, always_cloudy_fraction 0.01 (clean). Monsoon (Jul-Aug): 5 scenes at 60-70% cloud each, always_cloudy_fraction **0.65** — real cloud contamination confirmed (mean NDVI came out lower than dry season, backwards from real phenology). All ~46 scenes across Jun-Sep still left ~0.49 always-cloudy: persistent regional cloud, not scene count, is the limit. Confirms PLAN.md's own stated monsoon-cloud pitfall from real data. | [`02_cloud_data_access.ipynb`](notebooks/02_cloud_data_access.ipynb) |
| H2 | RandomForest land-cover baseline (7 features: 4 bands + NDVI/NDWI/NDBI) trained against real ESA WorldCover 2021 reference labels, real spatial block holdout; SAM label-assist tested for real | Accuracy 0.85 (vegetation F1 0.92, water 0.47, built-up 0.39, bare 0.02) — real confusion likely from the 2021-label vs. 2026-imagery gap in a fast-urbanizing city, not a classifier bug. `class_weight='balanced'`: accuracy 0.70 but much better minority recall (bare 0.01→0.67, water 0.36→0.66) — a real precision/recall tradeoff, not a fix. SAM (`samgeo`): default settings took ~955s for one small chip on CPU; a fast setting (9s) found only 9 scattered segments — real evidence 20m Sentinel-2 is the wrong resolution for SAM, deferred to H6's higher-res imagery. | [`03_landcover_baseline.ipynb`](notebooks/03_landcover_baseline.ipynb) |

## Notes

- H0's numbers describe one scene, one small AOI — they're a pipeline-correctness proof,
  not a characterization of Dhaka's land cover. See the notebook's own "Notes / CRS
  pitfalls" section for two real bugs (NaN-propagation in `.mean()`/`np.percentile`,
  and Planetary Computer's missing item-level `proj:epsg`) caught by actually running
  this against live data.
- Every later phase's model metrics (U-Net/foundation-model IoU, YOLO mAP, the spatial
  epidemiology model's coefficients) get their own row here as they're run for real —
  see [`PLAN.md`](PLAN.md) section 6 for what each phase's "Done when" gate requires
  before its row is added.
