# Results

One row per phase, per [`PLAN.md`](PLAN.md) section 3's rule: every phase ends with a
reproducible notebook and a metrics row here — updated as each phase actually runs
against real data, not projected in advance.

| Phase | What was proven | Real numbers | Notebook |
| :--- | :--- | :--- | :--- |
| H0 | Cloud STAC search, real Sentinel-2 load in native UTM, NDVI/NDWI/NDBI, both raster<->vector directions, real per-district zonal stats | Smoke tile: Dhaka/Gazipur/Narayanganj, 0.001% cloud scene (2026-02-14). Mean NDVI 0.24 / NDWI −0.26 / NDBI −0.02. 1,931 water polygons (15.1 km²) after sliver filtering. | [`01_fundamentals.ipynb`](notebooks/01_fundamentals.ipynb) |

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
