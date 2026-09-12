# Results

One row per phase, per [`PLAN.md`](PLAN.md) section 3's rule: every phase ends with a
reproducible notebook and a metrics row here — updated as each phase actually runs
against real data, not projected in advance.

| Phase | What was proven | Real numbers | Notebook |
| :--- | :--- | :--- | :--- |
| H0 | Cloud STAC search, real Sentinel-2 load in native UTM, NDVI/NDWI/NDBI, both raster<->vector directions, real per-district zonal stats | Smoke tile: Dhaka/Gazipur/Narayanganj, 0.001% cloud scene (2026-02-14). Mean NDVI 0.24 / NDWI −0.26 / NDBI −0.02. 1,931 water polygons (15.1 km²) after sliver filtering. | [`01_fundamentals.ipynb`](notebooks/01_fundamentals.ipynb) |
| H1 | Reusable `get_imagery(aoi, date_range, sensor)` cloud-masked composite (SCL-based), one call for any AOI/season | Dry season (Jan-Feb): 3 scenes, always_cloudy_fraction 0.01 (clean). Monsoon (Jul-Aug): 5 scenes at 60-70% cloud each, always_cloudy_fraction **0.65** — real cloud contamination confirmed (mean NDVI came out lower than dry season, backwards from real phenology). All ~46 scenes across Jun-Sep still left ~0.49 always-cloudy: persistent regional cloud, not scene count, is the limit. Confirms PLAN.md's own stated monsoon-cloud pitfall from real data. | [`02_cloud_data_access.ipynb`](notebooks/02_cloud_data_access.ipynb) |
| H2 | RandomForest land-cover baseline (7 features: 4 bands + NDVI/NDWI/NDBI) trained against real ESA WorldCover 2021 reference labels, real spatial block holdout; SAM label-assist tested for real | Accuracy 0.85 (vegetation F1 0.92, water 0.47, built-up 0.39, bare 0.02) — real confusion likely from the 2021-label vs. 2026-imagery gap in a fast-urbanizing city, not a classifier bug. `class_weight='balanced'`: accuracy 0.70 but much better minority recall (bare 0.01→0.67, water 0.36→0.66) — a real precision/recall tradeoff, not a fix. SAM (`samgeo`): default settings took ~955s for one small chip on CPU; a fast setting (9s) found only 9 scattered segments — real evidence 20m Sentinel-2 is the wrong resolution for SAM, deferred to H6's higher-res imagery. | [`03_landcover_baseline.ipynb`](notebooks/03_landcover_baseline.ipynb) |
| H3 | U-Net (ResNet34, multi-label water+built-up) at 10m resolution, 80 tiles, spatial-block holdout, flip/rotate augmentation; NDVI/NDWI/NDBI computed via TorchGeo (cross-checked exact match vs `src/indices.py`) | Held-out block: water IoU **0.25**, built_up IoU **0.20** — below PLAN.md's target (0.75/0.65). Root cause, checked before training: training region is ~30% built-up, held-out eastern block only ~4-5% — a real spatial distribution shift, not a training bug. In-sample (training-tile) IoU (0.31/0.51) confirms the gap. This is exactly the generalization problem spatial (block) CV exists to surface, that a random pixel split would have hidden. | [`04_unet_segmentation.ipynb`](notebooks/04_unet_segmentation.ipynb) |
| H4 | `build_predictor_table()` — one real command: H3's actual trained model run tiled over the full AOI → real per-district zonal predictor table + real GeoPackage (vectorized, cleaned water/built-up polygons) | 705 water polygons + 1055 built-up polygons after cleanup (min area 400m², simplified); real 3-district predictor table (7 spectral features + both class probabilities). Reproducibility checked, not assumed: identical inputs run twice produce an identical table (`DataFrame.equals`) and matching polygon counts. | [`05_vectorize_zonal_stats.ipynb`](notebooks/05_vectorize_zonal_stats.ipynb) |

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
