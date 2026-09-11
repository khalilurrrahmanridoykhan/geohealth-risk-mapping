# Geospatial AI for Public Health — Plan

Learn **satellite / drone image processing with AI models** (semantic segmentation,
object detection, EO foundation models) and connect the outputs to **public health**
decision-making — dengue / flood / heat risk mapping for Bangladesh.

Same working discipline as the Prohori plans: **one branch → one PR (`Phase X — <name>`)
→ merge → tag `phase-x`**, granular commits, `main` always demoable, every phase ends
with a **"Done when"** gate, post one public note/screenshot per phase.

**Repo:** `github.com/khalilurrrahmanridoykhan/geohealth-risk-mapping`
**Live target:** a Streamlit dashboard (Streamlit Community Cloud) + exported
shapefiles/GeoTIFF.
**Related:** [[project_prohori_fhir]] (DGHS data path), `Research Paper Plan: Bangladesh 2023 Dengue Outbreak.md`,
`Research Paper Plan: Bangladesh Diarrhea Flood Analysis.md`, `Nepal Flash Flood Dashboard — Plan.md`,
`Spatial_Epidemiology_Measles_2026_Plan.md`.

---

## 1. The core idea

Take **Sentinel-2** (10 m optical), **Sentinel-1** (SAR, sees through monsoon cloud),
land-surface temperature and rainfall over an area of interest. Run an **AI model** that
turns pixels into classes/objects — water, built-up, vegetation, flooded area, waste /
tyre dumps, informal settlements. **Vectorize** the result to polygons / shapefile,
compute **zonal statistics** per administrative unit (city ward / union / upazila),
**join** with public-health data (DGHS dengue counts, IEDCR bulletins, clinic
locations, WorldPop population), and fit a **spatial-epidemiology model** producing a
**ward-level risk score** and a **decision dashboard** for targeting vector control,
flood response, or heat-vulnerability outreach.

Field name: **Geospatial AI / Earth Observation (EO) for planetary health**. Method is
grounded in current literature — malaria risk in flood-prone zones (Scientific Reports,
2026), multimodal satellite + public-health frameworks (Nature Sci Data), dengue risk
modeling reviews.

---

## 2. Flagship deliverable

**"Dengue / flood-health risk mapping for Bangladesh using satellite imagery + AI"**

1. Pull Sentinel-2 + Sentinel-1 + LST + CHIRPS rainfall for Dhaka (and one flood-prone
   district, e.g. Sylhet or Kurigram).
2. AI model segments: built-up, vegetation, standing/stagnant water, bare soil;
   stretch: waste/tyre dumps, informal settlements.
3. Vectorize predictions → polygons → shapefile.
4. Zonal stats per city ward: % built-up, distance to water, greenness (NDVI),
   land-surface temperature, population (WorldPop / Google Open Buildings).
5. Statistical model links predictors to **DGHS / IEDCR dengue case counts** per ward,
   population as offset.
6. Output: ward-level risk map + Streamlit dashboard + shapefile/GeoTIFF exports the
   health department's own GIS can open.

**Why this project:** locally relevant (severe annual dengue + flooding), free data,
matches published methods (so results are checkable), existing DGHS relationship from
Prohori, and it reuses the dengue/flood research plans already in this folder.

---

## 3. Rules

- One branch → one PR (`Phase X — <name>`) → merge → tag `phase-x`. Granular commits,
  `--merge` not squash.
- `main` always runs end-to-end on at least a small sample AOI (a "smoke tile").
- **Python** primary; spatial stats in **R** (INLA) where it earns its place.
- **Free tiers only.** Google Colab / Kaggle free GPU to train; Google Earth Engine +
  Microsoft Planetary Computer for data (no bulk downloads).
- **Synthetic / public / aggregated data only** for anything published. Never publish
  household-level settlement or poverty maps. Ethics note in every phase that touches
  vulnerable populations.
- Every phase: a reproducible notebook + a `RESULTS.md` metrics row + one public post.
- Track metrics honestly: **IoU / F1 with spatial (block) cross-validation**, never a
  plain random split.

---

## 4. Skills / model coverage matrix — the point of this plan

| Capability | Covered by |
| :--- | :--- |
| Rasters, vectors, CRS/EPSG, GeoTIFF, spectral indices (NDVI/NDWI/NDBI) | **H0** |
| Shapefile ↔ raster: rasterize labels, vectorize predictions, zonal stats | **H0, H4** |
| Cloud data access: Google Earth Engine, STAC / Planetary Computer | **H1** |
| Classical ML pixel classification: Random Forest baseline | **H2** |
| Deep semantic segmentation: U-Net, DeepLabV3+ (PyTorch, SMP, TorchGeo) | **H3** |
| Tiling, augmentation, class imbalance, Dice/focal loss, **spatial CV** | **H3** |
| SAR flood mapping (Sentinel-1), multi-sensor stacking | **H5** |
| Object detection on drone imagery: YOLO (Ultralytics) | **H6** |
| EO **foundation models**: Prithvi-EO-2.0, TerraMind, Clay via **TerraTorch** | **H7** |
| **Segment Anything (SAM 2)** + `samgeo` for label-assist / zero-shot | **H2+ (tool, used throughout)** |
| Time-series / multi-season classification | **H8** |
| Spatial epidemiology: Poisson/NB GLM, offset, Moran's I, space-time scan | **H9** |
| Bayesian spatial model (R-INLA), SHAP interpretation | **H9, H10** |
| Zonal aggregation + health-data join + risk scoring | **H10** |
| Delivery: Streamlit dashboard, Leaflet/deck.gl map, shapefile/GeoTIFF export | **H11** |
| Near-real-time scheduled pipeline (new image → updated risk) | **H12 (stretch)** |

---

## 5. Tooling introduced per phase (install as needed)

| Phase | New tools |
| :--- | :--- |
| H0 | QGIS · GDAL · `rasterio`, `rioxarray`, `xarray`, `geopandas`, `shapely`, `rasterstats` |
| H1 | `earthengine-api` + `geemap` · `pystac-client`, `stackstac` / `odc-stac` (Planetary Computer) |
| H2 | `scikit-learn` · `samgeo` (SAM 2) · CVAT or Roboflow for labeling |
| H3 | PyTorch · **TorchGeo** · `segmentation-models-pytorch` · `albumentations` · Colab/Kaggle GPU |
| H5 | Sentinel-1 GRD via GEE · `pyroSAR` (optional) |
| H6 | **Ultralytics YOLO (v11)** · a drone/UAV sample set (OpenAerialMap / own) |
| H7 | **TerraTorch** · Hugging Face `transformers` · Prithvi-EO-2.0 / TerraMind / Clay weights |
| H8 | `stackstac` time cubes · `tsai` or simple temporal CNN |
| H9 | R + `INLA` · `pysal` / `esda` (Moran's I) · SaTScan or `libpysal` space-time |
| H10 | `xgboost` + `shap` · `rasterstats` zonal · health data from DGHS / IEDCR bulletins / DHS |
| H11 | **Streamlit** · `leafmap` / Folium · `geopandas.to_file` exports |
| H12 | GitHub Actions cron · a small object store for outputs |

---

## 6. Phases

### Phase H0 — Geospatial fundamentals + the smoke tile
- Learn: raster vs vector, CRS/EPSG, bands, resolution, NDVI/NDWI/NDBI.
- Build: load one Sentinel-2 scene over Dhaka in `rasterio`/`rioxarray`, compute the
  three indices, clip to a ward boundary shapefile, run a zonal-stats example.
- Practice both directions: rasterize a polygon layer into a mask; vectorize a
  thresholded index back to polygons; write a shapefile + a GeoTIFF.
- **Done when:** a notebook goes from a raw scene + an admin shapefile to a table of
  per-ward mean NDVI, plus a written one-pager on CRS pitfalls.

### Phase H1 — Cloud data access (GEE + STAC)
- Learn: Google Earth Engine model, `geemap`; STAC catalogs, Microsoft Planetary
  Computer, `pystac-client` + `stackstac`.
- Build: a reusable `get_imagery(aoi, date_range, sensor)` function — cloud mask,
  cloud-free composite, band stack — working from both GEE and Planetary Computer.
- **Done when:** one call returns an analysis-ready multi-band array for any AOI/season,
  documented, with a monsoon vs dry-season example for Dhaka.

### Phase H2 — Random Forest land-cover baseline + SAM label-assist
- Learn: supervised pixel classification; SAM 2 / `samgeo` for fast polygon labeling.
- Build: label ~300–500 chips (water / built-up / vegetation / bare) with SAM-assist;
  train a `scikit-learn` RandomForest; map over the AOI.
- **Done when:** RF land-cover map with a confusion matrix and per-class F1 on a
  spatially held-out block; labeling workflow written up.

### Phase H3 — Deep semantic segmentation (the core skill)
- Learn: U-Net / DeepLabV3+ via `segmentation-models-pytorch` and **TorchGeo**;
  tiling to 256×256, augmentation, Dice/focal loss, IoU/F1; **block cross-validation**.
- Build: train a U-Net for **water + built-up** segmentation on Sentinel-2; compare to
  the RF baseline.
- **Done when:** water IoU > 0.75 and built-up IoU > 0.65 on a spatially held-out
  region; a short note on why spatial CV changes the numbers.

### Phase H4 — Vectorize + clean + zonal stats pipeline
- Build: prediction raster → polygons (`rasterio.features.shapes`) → simplify, drop
  slivers, fix geometry → shapefile; then zonal statistics of every predictor to each
  ward.
- **Done when:** one command turns model output into a clean ward-level predictor table
  + a GeoPackage, reproducibly.

### Phase H5 — SAR flood mapping (Sentinel-1)
- Learn: SAR backscatter, water = low backscatter; multi-sensor stacking.
- Build: Sentinel-1 water/flood segmentation (threshold + a small U-Net); a
  before/during-flood pair for a real Bangladesh flood event; overlay health-facility
  points (OSM) to list cut-off / at-risk facilities and estimated affected population.
- **Done when:** a flood-extent map for one dated event + a table of affected wards,
  facilities, and population.

### Phase H6 — Drone imagery + YOLO breeding-site detection
- Learn: object detection, `Ultralytics YOLO`, labeling in Roboflow/CVAT.
- Build: detect containers / tyres / stagnant pools / rooftop tanks in
  drone / very-high-res imagery for one neighborhood.
- **Done when:** YOLO model with mAP reported on a held-out set + a density map of
  detected potential breeding sites per block.

### Phase H7 — EO foundation models (current SOTA)
- Learn: **TerraTorch**; fine-tuning **Prithvi-EO-2.0**, **TerraMind** (multimodal:
  optical + SAR + DEM), **Clay** with segmentation decoders.
- Build: fine-tune one foundation model on the Phase H3 labels; compare accuracy and
  label-efficiency (train on 25% / 50% / 100% of labels) against the U-Net.
- **Done when:** a table showing foundation-model IoU vs U-Net IoU at each label
  budget, with a recommendation.

### Phase H8 — Multi-season / time series
- Build: stack dry + monsoon + post-monsoon composites; classify seasonal standing
  water and vegetation dynamics as predictors.
- **Done when:** per-ward seasonal water-persistence and greenness-change layers added
  to the predictor table.

### Phase H9 — Spatial epidemiology core
- Learn: incidence, population offset, confounding; Moran's I; space-time scan
  statistics; Poisson / negative-binomial GLM; intro R-INLA.
- Build: assemble ward-level dengue counts (DGHS / IEDCR bulletins / published data);
  test spatial autocorrelation; fit a Poisson/NB model of counts ~ predictors + offset.
- **Done when:** a fitted model with coefficient interpretation, residual spatial
  autocorrelation checked, limitations written (ecological fallacy explicitly).

### Phase H10 — Risk scoring + health-data join + interpretation
- Build: combine GLM/INLA and an XGBoost + SHAP model; produce a normalized ward-level
  **risk score**; validate against a held-out season; SHAP plots for which
  satellite-derived factors drive risk.
- **Done when:** a ranked ward risk list with uncertainty, back-tested on one season,
  plus a one-page "how to read this / what it is not" for health officials.

### Phase H11 — Dashboard + exports (the live deliverable)
- Build: **Streamlit** app — pick district, see risk map, toggle predictor layers,
  download shapefile / GeoTIFF; `leafmap` interactive map. Deploy to Streamlit
  Community Cloud.
- **Done when:** public URL, a 2-minute demo video, exports open cleanly in QGIS.

### Phase H12 — Scheduled near-real-time pipeline (stretch)
- Build: GitHub Actions cron → fetch latest imagery → run inference → refresh risk
  layers + dashboard monthly (or after a flood alert).
- **Done when:** one unattended successful run visible in the dashboard's "last
  updated" stamp.

### Phase H13 — Write-up (stretch)
- A technical blog series (one post per phase already posted) consolidated into a
  methods note / preprint; offer it to DGHS / icddr,b / a university for real health
  data and field validation.
- **Done when:** a draft methods paper + an outreach email sent.

---

## 7. Suggested schedule

| Milestone | Phases | Rough time | Proof |
| :--- | :--- | :--- | :--- |
| **MVP** (portfolio-ready) | H0–H4, H9(light), H11(static) | ~3 months | Water/built-up U-Net (IoU > 0.75), ward predictor table, one dengue correlation, one risk map, GitHub repo |
| **v1** | H5, H7, H8, H9, H10, H11 | +3 months | SAR flood maps, foundation-model comparison, proper spatiotemporal model, live Streamlit dashboard, shapefile/GeoTIFF exports |
| **Stretch** | H6, H12, H13 | +2–3 months | Drone YOLO breeding-site map, monthly auto-refresh, methods paper + institutional outreach |

---

## 8. Data sources (all free)

| Data | Source | Use |
| :--- | :--- | :--- |
| Sentinel-2 (10 m optical) | Copernicus / GEE / Planetary Computer (STAC) | Land cover, indices |
| Sentinel-1 (SAR) | GEE / Planetary Computer | Flood mapping through cloud |
| Landsat 8/9 | USGS / GEE | Long time series, thermal band |
| MODIS / VIIRS LST | GEE | Land-surface temperature, heat |
| SRTM / Copernicus DEM | GEE | Elevation, flood modeling |
| VIIRS nightlights | GEE | Urbanization proxy |
| CHIRPS rainfall | GEE | Climate predictor |
| Google Open Buildings / Microsoft Building Footprints | Public | Building polygons, population downscaling |
| WorldPop / HRSL | worldpop.org | Gridded population |
| OpenStreetMap (HOT) | Overpass / HOT exports | Roads, clinics, water |
| Maxar Open Data, SpaceNet, xView2 | Public | Labeled disaster imagery for training |
| Health data | DGHS, DHIS2, IEDCR dengue bulletins, DHS surveys, published outbreak datasets | Case counts, facility locations |

Start with **Google Earth Engine** via `geemap` (browser, no downloads). Move to
**Microsoft Planetary Computer** when PyTorch training needs raw arrays.

---

## 9. Tech stack

- **Language:** Python; R for INLA-based spatial stats.
- **GIS:** QGIS, GDAL.
- **Data access:** `geemap`, `pystac-client`, `stackstac`, `odc-stac`.
- **Arrays / vectors:** `rasterio`, `rioxarray`, `xarray`, `geopandas`, `shapely`, `rasterstats`.
- **DL:** PyTorch, **TorchGeo**, `segmentation-models-pytorch`, **TerraTorch**, Ultralytics YOLO, `samgeo`.
- **Stats:** `scikit-learn`, `xgboost`, `shap`, `pysal`/`esda`, R-`INLA`.
- **Viz / app:** `leafmap`, Folium/Leaflet, deck.gl, **Streamlit**.
- **Compute:** Google Colab / Kaggle free GPU to start; rented GPU later.

---

## 10. Pitfalls & ethics

- **Spatial data leakage** — always block / spatial cross-validation, never a random split.
- **Ecological fallacy** — ward-level association ≠ individual risk; state it every time.
- **Label scarcity** — budget most time for labeling; this is why foundation models matter.
- **Monsoon cloud** — lean on Sentinel-1 SAR for anything time-critical.
- **Sensitive mapping** — settlement / poverty maps can enable eviction and stigma.
  Aggregate outputs, never publish household-level maps, respect data sovereignty, seek
  ethics review before any institutional partnership.
- **Don't overclaim** — decision *support* validated against real surveillance, not a
  predictor that replaces epidemiologists.

---

## 11. Learning resources

- Google **"Introduction to Google Earth Engine"** + `geemap` tutorials (Qiusheng Wu, YouTube).
- **TorchGeo** docs and tutorials.
- **TerraTorch** GitHub examples + ITU "AI for Good" Prithvi-EO-2.0 + TerraTorch workshop.
- Book: *Geographic Data Science with Python* (Rey, Arribas-Bel) — free online.
- Method papers to model on:
  - Scientific Reports 2026 — malaria risk in flood-prone zones — https://www.nature.com/articles/s41598-026-54811-7
  - Nature Sci Data — multimodal satellite + public-health framework — https://www.nature.com/articles/s41597-024-03366-1
  - Dengue risk modeling review — https://pmc.ncbi.nlm.nih.gov/articles/PMC4273492/
  - Stanford Human & Planetary Health — foundation models for disease-ecology habitat mapping — https://hph.stanford.edu/news/breaking-barriers-habitat-mapping-disease-ecology-accessible-ai-foundation-models-and
  - IBM Research — TerraMind — https://research.ibm.com/blog/terramind-esa-earth-observation-model
  - TerraTorch — https://github.com/torchgeo/terratorch
  - Awesome Remote Sensing Foundation Models — https://github.com/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models

---

*Created 2026-09-09.*
