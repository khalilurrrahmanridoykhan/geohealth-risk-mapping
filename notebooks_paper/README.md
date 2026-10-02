# AI Paper Track (Phases P0-P6)

Separate from the phased `notebooks/` (H0-H10). See the plan doc:
`~/Documents/AIWORK/plan/Geospatial AI for Public Health — AI Paper Track (Phases P0–P6) — Plan.md`

These notebooks are **GPU-dependent and built for Kaggle**, not this project's local
CPU-only environment. Run on Kaggle (Settings > Accelerator > GPU, Internet > On),
bring back the actual output/errors, and each gets fixed or confirmed from a real run
rather than assumed correct.

| Notebook | Phase | Status |
| :--- | :--- | :--- |
| `P0_prithvi_vs_unet_fair_finetune.ipynb` | P0 | **Done** -- real run on Kaggle (T4, ~66 min), both models completed the full 100-epoch budget. Prithvi-EO-2.0 beats a from-scratch U-Net on held-out IoU for water (0.442 vs 0.325) and built_up (0.284 vs 0.201), ties on the dominant "other" class (0.934 vs 0.929). Single run, no seeds yet (Phase P4). Four real bugs found and fixed from actual tracebacks/source along the way: an empty `kernelspec` blocking headless execution, a stale-numpy-at-kernel-startup crash (worked around by running the pipeline as a subprocess), TerraTorch's `plot_sample` crashing without a `LightningDataModule` (fixed via its documented `plot_on_val=False`), and a case-sensitive metric-key bug that was silently dropping Prithvi's own results from the comparison table. |
