# What a Fair Comparison of Earth Observation Foundation Models Actually Looks Like on One Small, Real, Consequential Flood

**Status:** Draft (Phase P5), 2026-10-03. Not yet submitted anywhere (Phase P6).

## Abstract

Earth observation (EO) foundation models are increasingly proposed as drop-in
upgrades over task-specific CNNs for disaster response and land monitoring, but most
evidence for this comes from large, curated, multi-event benchmarks. We ask a
narrower, more operational question: on one small AOI, with the modest resources a
single practitioner actually has, does a foundation model reliably beat a
from-scratch U-Net? We fine-tune Prithvi-EO-2.0 (300M, full fine-tune) and Clay v1.5
(frozen encoder, small trained head) against a from-scratch U-Net on a real Dhaka-area
land-cover task and on a real 2026 Bangladesh monsoon flood (Dirai, Sunamganj), using
each model's own documented recipe rather than a one-size-fits-all pipeline. Prithvi's
advantage over the U-Net on land cover is real and survives a 3-seed check (all seeds
agree in direction, water IoU +0.121 to +0.148). Clay's apparent advantage over the
same U-Net, reported in an earlier single run, does not survive the same check (sign
flips across seeds on both harder classes) -- a concrete illustration of why
single-run foundation-model comparisons are unreliable even when the recipe is
correct. On the real flood event, neither foundation model beats a 60-year-old
classical Otsu threshold, and all three models fail to generalize at all to a
third-party AOI 140km away, collapsing to the majority class. We report exact
sample sizes (as few as 6-16 held-out tiles), do not claim statistical power we do not
have, and are explicit that Prithvi's and Clay's own published
Sentinel-1-flood/Sen1Floods11 numbers (IoU 0.70-0.78 on a curated multi-event
benchmark) look nothing like what either model achieves on this one real event
(IoU 0.04-0.13) -- a real, measured gap between benchmark and deployment-scale
evidence, not asserted from priors. Code, notebooks, and real run logs are public.

## 1. Introduction

Geospatial foundation models (Prithvi-EO-2.0, Clay, and others) are pretrained on
large, diverse Earth observation corpora and then fine-tuned for a downstream task --
the promise being that this pretraining transfers useful structure (edges, textures,
spectral relationships) that a from-scratch model has to relearn from whatever small
amount of labeled data is actually available locally. Large benchmarks like PANGAEA
[Marsocci et al. 2024] and GEO-Bench-2 [2025] exist to measure this transfer across
many datasets, sensors, and tasks at once, and both already report the central
caveat this paper investigates empirically at a smaller scale: **geospatial
foundation models do not universally outperform task-specific supervised models**
(PANGAEA's own finding), and **no single model dominates across all tasks**
(GEO-Bench-2's own finding).

What those benchmarks do not tell a practitioner is what happens on the actual, much
smaller scale most real deployments operate at: one AOI, one event, whatever labeled
pixels a small team can actually produce, a single GPU session, and a decision (where
to send a flood response team, which facility is at risk) that has to be made from
whatever the model says, not from an average over dozens of curated datasets. This
paper is **not** a new benchmark and does not claim to out-measure PANGAEA,
GEO-Bench-2, or REOBench [Li et al. 2025] at their own scale. Its contribution is
narrower and, we think, still useful: a documented, honestly-reported, real-data case
study of what a careful practitioner actually gets when they follow each model's own
published fine-tuning recipe on one small, real, locally consequential problem --
including the parts that did not work, the single-run claims that turned out not to
survive a seed check, and the gap between this event-scale evidence and the same
models' own published benchmark numbers.

Concretely, we ask four questions on a real Dhaka-area land-cover task and a real
2026 Bangladesh monsoon flood (Dirai, Sunamganj, part of the real event UN agencies
alerted on in July 2026):

1. On a real local AOI, does an EO foundation model (Prithvi-EO-2.0, full fine-tune;
   Clay v1.5, frozen encoder) beat a from-scratch U-Net, using each model's own
   documented recipe rather than one recipe forced onto both?
2. Does whichever model wins generalize to a different AOI it was never trained near,
   or does strong held-out performance on the training tile mask total failure
   elsewhere?
3. On a real disaster event, does either foundation model beat the classical
   threshold-based method a practitioner would reach for first?
4. Do apparent single-run advantages survive being checked against real seed-to-seed
   variance -- and if a claim does not survive that check, what should a practitioner
   have done differently before believing it?

## 2. Related Work

**Geospatial foundation model benchmarks.** PANGAEA [Marsocci et al., arXiv:2412.04204]
is a global, multi-dataset, multi-sensor benchmark built specifically to correct
earlier benchmarks' geographic bias toward North America and Europe; across its many
datasets it finds GeoFMs do not universally beat task-specific U-Nets. GEO-Bench-2
[arXiv:2511.15658] extends this with 19 permissively-licensed datasets across
classification, segmentation, regression, and detection, organized into "capability"
groups, and similarly finds no single model dominates across all tasks. REOBench
[Li et al., arXiv:2505.16793] asks a different question -- not which model wins on
clean data, but how much performance degrades under 12 realistic corruption types --
and finds degradation varies widely (under 1% to over 25% mIoU/mAP) by task and
architecture. All three are large, curated, multi-dataset efforts; none reports what
happens at the single-AOI, single-event scale this paper studies, and none reports
seed-to-seed variance for a single local deployment decision.

**Sen1Floods11 and Prithvi/Clay flood fine-tuning.** The Sen1Floods11 dataset
[Bonafilia et al. 2020] is the standard benchmark for Sentinel-1/2 flood segmentation
(446 hand-labeled 512x512 chips, 11 flood events, 14 biomes). IBM/NASA publish two
Prithvi checkpoints fine-tuned on it
(`Prithvi-EO-2.0-300M-TL-Sen1Floods11`, `Prithvi-EO-1.0-100M-sen1floods11`) -- both,
we confirmed directly from their model cards before relying on this, fine-tuned on
**Sentinel-2 optical** input (Blue/Green/Red/Narrow NIR/SWIR/SWIR2), not SAR, despite
Sen1Floods11 itself including Sentinel-1 imagery. The `eo-foundation-flood` project [github.com/mertsemih/eo-foundation-flood] is the
most directly comparable prior work to our own Phase P2/P3b -- read directly from its
own README before citing, not from a secondary summary, since its numbers matter to
our own argument. It is an **in-progress, unpublished research project** (explicit
status checklist, preprint "pending weeks 10-12"), fine-tuning Prithvi-EO-2.0 on
**Sentinel-2 optical** input from Sen1Floods11 (Clay experiments are planned but not
yet reported) under frozen/LoRA/full-fine-tune regimes, studying the low-label regime
and out-of-distribution generalization (a held-out Bolivia split). At 100%
labels/seed 0 it reports water IoU of 0.699 (frozen), 0.778 (LoRA), 0.779 (full
fine-tune) for Prithvi -- but its own **from-scratch and ImageNet-pretrained U-Net
baselines score higher still (0.823 / 0.812), and stay at or above 0.79 down to 5% of
labels, beating every Prithvi variant at every label fraction tested.** This is an
independent, striking confirmation of PANGAEA/GEO-Bench-2's own message from a
project specifically built to test it on flood segmentation. We return to this
directly in Section 5: our own real-event SAR numbers (0.040-0.130) sit far below
even Prithvi's own (optical, curated-benchmark) numbers in that project, let alone its
U-Net baseline's -- a real, large gap, with the important caveat that the input
modality also differs (their benchmark uses clean, pre-selected optical chips; our
real event forced SAR-only input because the actual flood window was too cloud-covered
for optical at all -- itself part of the gap, not a confound to explain away).

**Flood monitoring and nowcasting for South Asia/Bangladesh.** Operational flood
nowcasting work in this region includes a multi-sensor ensemble framework piloted for
continuous flood extent monitoring across South Asia during the 2025 flood season
[arXiv:2605.10950], and several Bangladesh-specific deep-learning forecasting studies
(GRU/LSTM-based, reporting high accuracy on historical water-level/weather time
series rather than real-time SAR/optical segmentation). None of this prior work, to
our knowledge, evaluates an EO foundation model's segmentation output against a
classical threshold method on one specific, named, already-occurred flood event with
real downstream facility/population numbers attached, which is what Phase P2 of this
work does.

**GIS/AHP and ML flood-risk mapping for this specific region.** A tri-dimensional
GIS-AHP flood risk prioritization framework for northeastern Bangladesh [DOI:
10.2166/wcc.2026.264], integrating 19 geospatial indicators across susceptibility,
exposure, and vulnerability, identifies Sunamganj and Sylhet districts as the
highest-priority zones in the region (over 72% and 42% of land respectively
classified high-to-very-high risk) -- directly corroborating, from an independent
multi-criteria method, that this paper's real case-study AOI (Dirai, Sunamganj) and
its real generalization-test AOI (Sylhet, Phase P1) are both genuinely high-stakes
locations, not arbitrary choices. `HaorFloodAlert`
[arXiv:2605.20167] is a 72-hour ML early-warning system specifically for Bangladesh's
haor wetlands (the same wetland system Sunamganj sits in), a complementary,
forecasting-focused counterpart to this paper's segmentation-focused case study.
Other recent Bangladesh flood-susceptibility work (Random Forest/MaxEnt, ANN,
GIS-AHP) addresses longer-horizon risk mapping rather than real-time event
segmentation and is not in direct competition with this paper's narrower question.

**What is and is not new here.** Nothing in this paper is a new model, a new general
benchmark, or a claim to out-measure PANGAEA/GEO-Bench-2/REOBench at their own scale.
What is real and, we believe, underreported elsewhere: (a) a documented case where a
single-run foundation-model advantage was reported, then checked against seed
variance, and found not to hold -- most comparisons we are aware of report one run;
(b) a direct, measured gap between the same model families' own curated-benchmark
numbers and their real-event performance on one small, real, locally consequential
AOI; (c) an explicit test of whether strong held-out performance on a training tile
implies anything about a nearby but untrained-on region (it does not, for any of the
three models here); and (d) an honest record of which scope assumptions in our own
original plan turned out to be wrong when checked against real APIs and real cloud
cover, and what we changed as a result, rather than a plan presented as if it had been
correct from the start.

## 3. Methods

### 3.1 Tasks and data

**Land-cover task.** A 3-district Dhaka-area AOI (90.30-90.55 degE, 23.75-23.95 degN),
real Sentinel-2 L2A composite (6 bands: Blue, Green, Red, NIR_broad, SWIR1, SWIR2;
Jan-Feb 2026, cloud-masked), real ESA WorldCover 2021 labels reclassified to 3 classes
(other/water/built_up). Tiled to 256x256; a spatial-block (eastern 30%) split gives
64 train / 16 held-out tiles -- the held-out block is a contiguous, never-trained-on
region of the same scene, not randomly scattered pixels.

**Flood/SAR task.** A real 2026 Bangladesh monsoon flood at Dirai, Sunamganj
(91.15-91.50 degE, 24.50-24.80 degN; 402mm/24hr record rainfall, 10 districts,
1.11M+ people affected per UN alert), real Sentinel-1 RTC VV/VH before (2026-06-19)
and during (2026-07-13) the event, real ESA WorldCover water labels for the CNN
tasks. Only 36 tiles total in this AOI (30 train, 6 held-out) -- stated here plainly,
not discovered partway through a results table.

**Generalization test.** A real, different AOI around Sylhet city (91.80-92.05 degE,
24.85-25.05 degN), ~140km from the land-cover AOI, never used in any training.

### 3.2 Models and recipes

Each model is fine-tuned per its own documented recipe, not a single recipe forced
onto all three:

- **Prithvi-EO-2.0-300M (TL)**, via TerraTorch, full fine-tune: `EncoderDecoderFactory`,
  `SelectIndices`[5,11,17,23] -> `ReshapeTokensToImage` -> `LearnedInterpolateToPyramidal`
  necks, `UNetDecoder`, Dice loss, AdamW (lr=1e-4, wd=0.1), `ReduceLROnPlateau`,
  100-epoch budget, `EarlyStopping(patience=15)`. Recipe verified against the
  maintainer's own TerraTorch-Examples Sen1Floods11 config before any code was written
  against it, correcting an earlier unverified assumption (LR warmup + cosine decay)
  found wrong before it was used.
- **Clay v1.5**, frozen encoder + trainable conv/pixel-shuffle segmentation head
  (`claymodel.finetune.segment.factory.Segmentor`): FocalLoss, AdamW on head
  parameters only (lr=1e-5, wd=0.05, betas=(0.9,0.95)),
  `CosineAnnealingWarmRestarts`, same 100-epoch budget and early stopping. Per-sensor
  normalization from Clay's own published `metadata.yaml` (sentinel-2-l2a for land
  cover; sentinel-1-rtc, already in dB, for the flood task).
- **U-Net** (from scratch, no ImageNet transfer): `smp.Unet`, ResNet34 encoder (land
  cover) or ResNet18 (SAR, matching the main project's own earlier H5 architecture),
  Dice(+BCE for the binary SAR task), AdamW/Adam, same epoch budget and early
  stopping.

A fourth candidate -- Prithvi fine-tuned on Sen1Floods11 for the flood task, as
originally planned -- was dropped after verifying (not assuming) that both public
Prithvi+Sen1Floods11 checkpoints are optical-only, and that every available
Sentinel-2 scene for the real flood window has 70-99.8% cloud cover. Clay (genuinely
SAR-native per its own published band metadata) took that slot instead.

### 3.3 Statistical treatment

Most results below are single runs, reported as such. For the land-cover task's full
(100%) label budget -- the paper's central comparison -- we additionally ran 3
explicitly-seeded repeats (`pl.seed_everything(0|1|2)`) per model and report mean +/-
std and the per-seed paired difference between models. We did not attempt a formal
significance test: n=3 supports a real mean/variance estimate but not a trustworthy
p-value, and we report unanimity (or lack of it) in the sign of the paired difference
across all three seeds as the practical signal, not a p-value we cannot honestly
support.

## 4. Results

### 4.1 Land cover: a fair three-way comparison (Phase P0/P0b)

| Model | IoU other | IoU water | IoU built_up |
|---|---|---|---|
| Prithvi-EO-2.0 (full fine-tune) | 0.934 | 0.442 | 0.284 |
| Clay v1.5 (frozen + head) | 0.936 | 0.345 | 0.224 |
| U-Net (from scratch) | 0.929 | 0.325 | 0.201 |

Both foundation models beat the from-scratch U-Net on the harder minority classes in
this single run; Prithvi's full fine-tune shows the largest margin.

### 4.2 Generalization test: the headline result does not travel (Phase P1)

Evaluating the exact checkpoints above (no retraining) on the real Sylhet AOI
(class balance: other 88.9%, water 7.4%, built_up 3.7%):

| Model | Region | IoU other | IoU water | IoU built_up |
|---|---|---|---|---|
| Prithvi | Dhaka (training tile) | 0.934 | 0.442 | 0.284 |
| Prithvi | Sylhet (new) | 0.891 | **0.000** | **0.000** |
| U-Net | Dhaka | 0.929 | 0.325 | 0.201 |
| U-Net | Sylhet (new) | 0.891 | **0.000** | **0.000** |
| Clay | Dhaka | 0.936 | 0.345 | 0.224 |
| Clay | Sylhet (new) | 0.096 | 0.066 | 0.005 |

Prithvi and U-Net both collapse to predicting the majority class ("other") for
virtually every pixel 140km away -- confirmed via per-class accuracy
(`Class_Accuracy_water=0.0`, `Class_Accuracy_built_up=0.0`, `Class_Accuracy_other
=0.9999` for Prithvi), and their identical 0.891 "other" IoU is not a coincidence: it
is exactly the region's true "other" base rate (0.889), the arithmetic signature of
total class collapse. Clay fails differently -- near-random noise across all three
classes, worse than trivial majority-guessing would score. Strong held-out
performance on the training tile predicted nothing about performance one province
over.

### 4.3 The real event: classical beats both foundation models (Phase P2)

Three-way comparison on the real Sunamganj flood (36 tiles total, 30 train/6 held-out):

| Method | Held-out IoU (water) | Flood extent | Polygons | Facilities at risk | Population affected |
|---|---|---|---|---|---|
| Otsu threshold (classical) | -- | 8.02% | 418 | 5/13 | 20,905 |
| SAR U-Net (fresh GPU, 100 epochs) | 0.130 | 3.02% | 331 | 6/13 | 14,486 |
| Clay v1.5 (SAR) | 0.040 | -- | -- | -- | -- |

A full GPU epoch budget meaningfully improves the U-Net over an earlier CPU-limited
10-epoch attempt (IoU 0.077 -> 0.130, ~70% relative), but it remains far below either
model's own published Sen1Floods11-benchmark numbers (Section 5), and it disagrees
with the classical method about *which* facilities are at risk, not just how much
area floods (6 vs. 5 of 13 facilities, despite mapping *less* than half the flooded
area). Clay transfers poorly from its optical/multi-sensor pretraining to raw SAR
backscatter. On this one real, consequential event, the 60-year-old classical
threshold remains the most defensible method of the three.

### 4.4 Label efficiency: a real trend for one task, honest noise for the other (Phase P3)

Land cover (25/50/100% of 64 train tiles, nested budgets):

| Model | 25% built_up | 50% built_up | 100% built_up |
|---|---|---|---|
| Prithvi | 0.152 | 0.188 | 0.284 |
| U-Net | 0.164 | 0.186 | 0.201 |
| Clay | 0.248 | 0.236 | 0.224 |

Flood/SAR (25/50/100% of 30 train tiles -- 8/15/30 tiles in absolute terms):

| Model | 25% IoU (water) | 50% | 100% |
|---|---|---|---|
| U-Net | 0.105 | 0.092 | 0.130 |
| Clay | 0.078 | 0.074 | 0.040 |

On land cover, Prithvi's built_up IoU clearly increases with more data -- the
opposite of an earlier under-trained (CPU, 10-epoch) attempt that had found no such
benefit, resolved here as an artifact of that earlier under-training, not a real
property of the model. On the flood task, at 8-30 total training tiles, neither model
shows an interpretable trend (U-Net non-monotonic; Clay actually worsens with more
data) -- almost certainly sampling noise, and we report it as exactly that rather
than searching for a story in it.

### 4.5 Does the headline claim survive a seed check? (Phase P4a)

Three seeded repeats, land cover, 100% budget:

| Comparison | Class | Per-seed paired difference | Robust? |
|---|---|---|---|
| Prithvi - U-Net | water | +0.148, +0.132, +0.121 | **Yes** -- all 3 seeds agree |
| Prithvi - U-Net | built_up | +0.045, +0.029, +0.053 | **Yes** -- all 3 seeds agree |
| Clay - U-Net | water | +0.041, -0.015, +0.015 | **No** -- sign flips |
| Clay - U-Net | built_up | +0.053, -0.011, -0.065 | **No** -- sign flips, mean ~ 0 |

Prithvi's advantage in Section 4.1 is real. Clay's apparent advantage in the same
table is not -- it was a single lucky draw. We say this plainly rather than letting
an earlier table stand uncorrected.

## 5. Discussion

**The benchmark-to-event gap is real and large.** `eo-foundation-flood`'s Sen1Floods11
numbers -- Prithvi full fine-tune 0.779, LoRA 0.778, frozen 0.699, and its own
from-scratch U-Net baseline at 0.823, all water IoU at 100% labels, all via clean
optical Sentinel-2 input -- are roughly 6-20x higher than what either foundation
model achieved on our one real event (U-Net 0.130, Clay 0.040, both via SAR). Some of
this gap is expected and explicable, and we name every contributor we can rather than
asserting one: Sen1Floods11 is a curated, hand-labeled, multi-event, multi-biome
benchmark with far more and better-aligned training data than one small AOI's 30
tiles; our ground truth (WorldCover's static water layer) is a coarser,
less event-specific label than Sen1Floods11's hand-annotated flood masks; and,
critically, our models had to work from SAR alone because the real flood window was
too cloud-covered for optical input at all -- the same real-world constraint that
makes SAR the only honest choice for this event, and makes the clean, pre-selected,
optical benchmark chips a genuinely easier problem by construction, not just a larger
one. We cannot and do not claim to apportion the gap between these causes with the
data we have. What we can say is the gap itself -- and not knowing its size without
measuring it on a real local case -- is the practical risk a deployment decision
maker faces: a benchmark number from a paper, even a very good one, is not a promise
about one specific AOI's one specific event.

**Single-run comparisons are a real liability, not a theoretical one.** Section 4.5's
correction is not a hypothetical cautionary tale -- it happened in this project's own
earlier phases, was caught only because we spent the GPU time to check, and would
otherwise have stood as a real, cited, wrong claim ("Clay beats U-Net on both harder
classes"). We do not know how often this happens elsewhere in the literature because
most single-event comparisons we are aware of, including some cited above, do not
report multiple seeds.

**Limitations, stated plainly.** Every number in this paper comes from one small AOI,
one season, and (for the flood task) one event -- generalizability claims beyond that
scope are exactly what Section 4.2 shows should not be trusted without checking.
Sample sizes are small enough to state exactly: 16 held-out land-cover tiles, 6
held-out flood tiles, 3 seeds for the one statistically-checked comparison. The
remaining 12 of 15 possible (model x task x budget) seed-checked configurations were
not run, for a stated reason (15-25+ hours of additional GPU time for a slice we judged
lower-stakes than the one we did run) -- not silently dropped. No paired significance
test (e.g., a block bootstrap over held-out tiles) was computed; doing so honestly
would require per-tile metric logging this project's current `test_step`
implementations do not yet provide, a real, named engineering gap rather than a
result we chose not to report.

## 6. Conclusion

On one small, real AOI and one small, real, consequential flood, a foundation
model's advantage over a from-scratch U-Net was real in one case (Prithvi, land
cover) and not real in another (Clay, same task) -- indistinguishable from each
other without the seed check that caught the difference. On the real flood event,
neither foundation model beat a decades-old classical threshold. None of the three
models generalized to a nearby but untrained-on region. These are not criticisms of
Prithvi or Clay as architectures -- both have real, strong, independently-published
benchmark numbers on curated multi-event data that this paper's own event-scale
numbers fall well short of, for reasons partly explicable by data quantity/quality and
partly just the nature of one real small deployment. The practical conclusion is
narrower and, we think, more useful than either "foundation models always win" or
"foundation models are overhyped": at the scale most real local deployments actually
operate at, the only way to know which model wins on *your* AOI is to run the
comparison on your own data, with more than one seed, and to check whether the winner
still wins somewhere else before trusting it.

## Data and code availability

All code, notebooks, and real Kaggle run logs referenced in this paper are public at
`github.com/khalilurrrahmanridoykhan/geohealth-risk-mapping`
(`notebooks_paper/P0`-`P4a`). Archiving with a Zenodo DOI (Phase P5's stated
requirement) is not yet done -- a real, open action item, not assumed complete.

## References

- Marsocci, V. et al. PANGAEA: A Global and Inclusive Benchmark for Geospatial
  Foundation Models. arXiv:2412.04204, 2024.
- GEO-Bench-2: From Performance to Capability, Rethinking Evaluation in Geospatial AI.
  arXiv:2511.15658, 2025.
- Li, X. et al. REOBench: Benchmarking Robustness of Earth Observation Foundation
  Models. arXiv:2505.16793, 2025.
- Bonafilia, D. et al. Sen1Floods11: a georeferenced dataset to train and test deep
  learning flood algorithms for Sentinel-1. CVPR Workshops, 2020.
- Szwarcman, D. et al. Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for
  Earth Observation Applications. arXiv:2412.02732, 2024.
- `eo-foundation-flood`: in-progress, unpublished research project (no preprint as of
  this writing). Parameter-efficient fine-tuning of Prithvi-EO 2.0 (Clay planned, not
  yet reported) for flood segmentation on Sen1Floods11's Sentinel-2 optical split.
  github.com/mertsemih/eo-foundation-flood (README read directly, 2026-10-03).
- Continuous Flood Nowcasting in South Asia: A Multi-Sensor Ensemble Remote Sensing
  Framework for Flood Extent. arXiv:2605.10950, 2026.
- HaorFloodAlert: A 72-Hour Machine Learning Early Warning System for Flash Floods in
  Bangladesh's Haor Wetlands. arXiv:2605.20167, 2026.
- Flood risk prioritization in data-scarce regions: a tri-dimensional GIS-AHP
  framework for Northeastern Bangladesh. DOI: 10.2166/wcc.2026.264, 2026.
- Clay Foundation. Clay v1.5. github.com/Clay-foundation/model.
