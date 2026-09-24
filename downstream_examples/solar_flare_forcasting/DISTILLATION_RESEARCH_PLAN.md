**Research plan: compact solar-flare forecasting through distillation from Surya**

Prepared September 17, 2026. This is a proposed study, grounded in the local implementation and primary research sources. No training, benchmark, or full-dataset download was performed for this plan. Performance, timing, and memory targets below are hypotheses or planning estimates until measured.

**1. Research question and scope**

Can a compact model, trained using the existing flare-specific Surya checkpoint, preserve forecasting skill and calibrated probabilities on later solar-cycle data while substantially reducing parameters, input bandwidth, and inference latency?

Start with task-specific distillation. The teacher is the complete model loaded from `assets/solar_flare_weights.pth`, including its required architecture and LoRA configuration. The student is a separate, independently deployable classifier. LoRA reduces the number of trainable parameters but still requires the large backbone at inference; it does not by itself produce the desired small model.

Initial engineering targets, to agree before final experiments: at most 25M student parameters, at least 5x measured model-inference speedup over the teacher, and no more than 0.02 absolute degradation in TSS and average precision. These are provisional objectives, not promised results or established equivalence margins. Report confidence intervals and the full quality-versus-cost curve even if the targets are missed.

Three hypotheses give the experiments a clear purpose:

- H1: soft-target distillation improves a student over the same architecture trained only on labels.
  - Labels only: The training label says “flare occurred” (1) or “no flare” (0). The student learns to predict those labels.
  - Soft-target distillation: The teacher provides a probability, such as “80% chance of a flare.” The student learns to match the teacher’s probabilities, usually while also learning from the true labels.
  - **So the research question is: Does learning from the teacher improve the student’s performance on unseen data compared with training that same student only on labels? That’s a hypothesis to test, not a guaranteed outcome.**
- H2: intermediate features provide additional useful information beyond the teacher's single binary logit.
  - **The hypothesis is: Does matching intermediate features improve the student beyond matching only the teacher’s output? To test it, compare the same student trained with labels + output distillation against labels + output distillation + feature matching. More information does not automatically mean better performance.**
- H3: a full-resolution teacher helps a lower-resolution student retain forecasting skill under the Solar Cycle 24-to-25 distribution shift.
  - **This is a hypothesis that a teacher trained on detailed solar images can help a student using smaller, less detailed images keep its prediction accuracy when tested on a different solar cycle.**

A model evaluated only on flares should be described as a compact flare forecaster distilled from a foundation model. Claiming a compact general heliophysics foundation model would require broader representation training and evaluation on several downstream tasks. Surya's original scope and pretraining are described in the [Surya paper](https://arxiv.org/abs/2508.14112).

**2. Prior work and what to take from it**

| Work | Relevant method | Implication for this project |
|---|---|---|
| [Hinton et al., 2015](https://arxiv.org/abs/1503.02531) | Ground-truth supervision plus temperature-softened teacher outputs. | Establish a simple labels-plus-logits baseline first. Teacher supervision also needs representative input data. |
| [Feng et al., ApJS 2023](https://doi.org/10.3847/1538-4365/ace96a), *Toward Model Compression for a Deep Learning–Based Solar Flare Forecast on Satellites* | ResNet18 teacher to a four-convolution-layer student, followed by pruning and quantization. Table 4 reports 11.177M teacher parameters and 1.898M after KD alone. | Solar-flare distillation already has precedent. Distinguish architecture compression, quantized file size, and measured speed; results on that dataset do not transfer numerically to SuryaBench. |
| [DeiT, ICML 2021](https://proceedings.mlr.press/v139/touvron21a.html) | A transformer student learns through a distillation token, commonly from a CNN teacher. | Teacher and student can use different architectures. A CNN student is a valid first choice for a transformer teacher. |
| [DINOv2](https://arxiv.org/html/2304.07193v2) | A large vision model teaches smaller models using global and patch-level objectives. | Motivate a feature-distillation ablation and, later, an optional representation-transfer stage on unlabeled training-period images. |
| [MobileSAM](https://arxiv.org/html/2306.14289v2) | Distills a heavy foundation-model encoder into a small encoder by matching embeddings; caches teacher embeddings before training. | Run the expensive teacher once per selected view, then reuse its outputs across student experiments. Its reported training time is not a prediction for Surya. |
| [EfficientViT-SAM](https://arxiv.org/abs/2402.05008) | Encoder distillation followed by end-to-end task training. | Feature pretraining followed by supervised refinement is an extension if joint distillation is insufficient. Hardware measurements must accompany parameter counts. |

A recent related direction is [Cross-Modal Decoupled Knowledge Distillation for Imbalanced Solar Flare](https://openreview.net/pdf?id=wVGowBaAfu), which transfers multimodal, engineered-feature knowledge into a reduced-input student. The [MONTI 2026 organizer page](https://sites.google.com/view/monti2026/home) lists it as an accepted non-archival workshop paper. It is relevant to channel-reduction extensions, but is not evidence of Surya image-model distillation.

The proposed contribution is therefore a controlled study of representation transfer, resolution reduction, and forecasting reliability across time. A targeted search did not establish that this exact Surya study exists; that is not proof of novelty. Refresh the related-work search before committing to a paper claim.

**3. Define the scientific task before training**

Use the existing binary target: whether an M1.0-or-stronger flare occurs within the next 24 hours. The local dataset reads `label_max` at the reference timestamp. The [official label card](https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-flare-forecasting) documents the threshold and horizon. The [SuryaBench paper](https://www.nature.com/articles/s41597-026-06552-5) specifies event-start-based window membership and explains temporal separation.

Inputs are two observations, at t-60 minutes and t, with the same 13 AIA/HMI channels as the teacher. Keep channel order, time order, units, registration, masks, and the teacher's released normalization fixed. Changing the YAML `time_delta_target_minutes: +60` does not create one-hour flare labels; that setting belongs to the inherited image-forecast loader. A different flare horizon requires new labels and appropriate teacher supervision.

Document event-window endpoint conventions, missing-channel handling, quality flags, and whether already ongoing flares are present at forecast issuance. Inspect a few GOES events manually against their labels, including M1.0 boundary cases. Report that operational data availability can differ from retrospective product timestamps.

*GOES (Geostationary Operational Environmental Satellites) carry an X-Ray Sensor measuring disk-integrated solar soft X-ray flux in the 1–8 Å band, at ~1-minute cadence, continuously since the 1970s.
A flare event is a catalogued entry in that flux record: start time, peak time, end time, peak flux, and often an associated active region (NOAA AR number).
The class letter is a log scale of peak 1–8 Å flux in W/m²: A = 1e-8, B = 1e-7, C = 1e-6, M = 1e-5, X = 1e-4, with the number as a linear multiplier. So M1.0 = 1e-5 W/m², C5.3 = 5.3e-6.*

**4. Data prerequisites and split policy**

The local folders contain only 15 NetCDF inference files and 23 January 2014 validation-example files. They are suitable for checking shapes and teacher loading, not estimating distillation effectiveness. They are also not a clean training subset of the benchmark. Synthetic tensors can test optimization plumbing; benchmark validation samples must remain excluded from final student optimization.

The locally downloaded label CSVs contain the following counts before joining to available imagery:

| Split | Label rows | Positive rows | Positive fraction |
|---|---:|---:|---:|
| Train | 74,760 | 9,051 | 12.11% |
| Validation | 3,672 | 400 | 10.89% |
| Test | 43,848 | 12,903 | 29.43% |

These are correlated forecast windows, not independent flare events. Usable counts will shrink after image availability and quality checks. The current image index starts later than the label collection, so do not equate label count with training-set size.

Preserve the released training dates in 2010–2019, January 15–31 validation windows, and 2020–2024 test period. Exclude `leaky_validation.csv`; preserve the buffer dates. The official label card defines these splits. Prevent input or label windows crossing split boundaries, and use temporal/event grouping rather than random frame splitting. If creating alternative splits, use approximately 14-day separation in addition to window-boundary checks, consistent with the SuryaBench temporal protocol.

Audit the 2019-to-2020 boundary explicitly: late-December training labels can include January test events. Distinguish results using the published split from a stricter purged protocol. Any stricter claim must cover teacher exposure as well as student exposure; purging student examples cannot undo a released teacher's training history.

Reserve the later period for the final evaluation. Do not train on its labels, distill on its unlabeled inputs, or choose hyperparameters using its scores. The released base Surya paper describes pretraining on 2011–2019, but the [flare checkpoint card](https://huggingface.co/nasa-ibm-ai4science/solar_flares_surya) is empty. Confirm the task checkpoint's actual training and selection history before making strict out-of-time claims. If provenance cannot be established, disclose the limitation or train a controlled teacher from the base model using the prescribed training dates.

The full ML-ready image collection is available via the [official core-SDO card](https://huggingface.co/datasets/nasa-ibm-ai4science/core-sdo) and [NASA SuryaBench AWS registry](https://registry.opendata.aws/surya-bench/). It is hundreds of terabytes at full cadence. Retrieve only timestamps needed by the chosen task manifest; inspect actual object names rather than assuming the local year/month layout matches remote keys.

The registry and dataset card describe different coverage endpoints, so verify actual object availability before promising a complete 2020–2024 image test set. If only a subset is accessible, declare its dates and missingness and evaluate every model on the same cohort.

Build a manifest containing sample ID, reference UTC time, input paths, channel order, target, split, quality flags, and event/group information when available. Verify actual files and readability, not only `present=1` in a CSV. Record missingness by date and class. Fail explicitly on missing samples during research runs rather than silently substituting another example.

Start the pilot with approximately 2,000–5,000 eligible training windows spread across years and activity levels. Expand toward all available official training windows after validating throughput and teacher quality. If the pilot oversamples positive events, document it; preserve natural prevalence for validation and test.

**5. Repository issues to address before experiments**

These are source-code observations, not conclusions about the published model's accuracy:

- `surya/datasets/helio.py` requires and reads a future t+1h image even for flare classification. The classifier uses only the two input images. Add a classification-only dataset to eliminate that extra read and availability restriction. This future image is currently unnecessary I/O, not a demonstrated future-image input leak. Record any resulting change in eligible sample counts.
- `metrics.py` compares predictions directly with 0.5, but `finetune.py` supplies raw logits. Apply sigmoid before a probability threshold, or convert the chosen probability threshold to logit space. A 0.5 probability threshold corresponds to logit zero.
- The example limits training to 2,000 batches and validation to 200. Its sampler setup and missing epoch advancement need review. Use complete, deterministic validation with no dropped or duplicated examples, and deliberate epoch sampling for training. Final metrics must cover the entire declared evaluation manifest.
- `infer.py` is a small random-sample demonstration. Its `data_type` argument does not actually select distinct split paths. Build a dedicated evaluation runner that takes an explicit manifest.
- The teacher uses `infer.py`'s base-model plus LoRA construction and strict task-checkpoint loading. Verify missing/unexpected keys, finite logits, output variation, and known sample predictions. Freeze all teacher parameters and use evaluation mode for target generation.
- Existing CNN students use `weights=None`. Disable LoRA for students and train their parameters normally. Use separate teacher/student configurations so reducing student resolution never silently changes the teacher.
- A smaller SpectFormer cannot simply load all full-size weights. Spectral filters depend on spatial grid size, and the existing factory filters incompatible shapes. Explicitly report initialization and loaded parameter coverage; otherwise an apparently pretrained small model may be mostly random.
- If retraining the teacher, audit which parameters receive gradients. In this code, attention projections are named `qkv`/`proj`, whereas configured LoRA attention targets use other names; MLP `fc1`/`fc2` targets do match. Also explicitly check that the classifier head is trainable. Do not modify a released checkpoint's architecture without verifying compatibility.

Add meaningful checks for the new research path: manifest separation, label/window alignment, cached target/sample identity, teacher frozen/student gradients, binary loss consistency, and full evaluation coverage.

**6. Student architectures**

| Candidate | Starting input | Approximate parameters | Purpose |
|---|---|---:|---|
| Existing ResNet18 classifier | Two 13-channel frames at 512×512 | 11.25M | Primary student; simple, already supported, about 32x fewer parameters than the current teacher. |
| Existing MobileNetV2 classifier | Same | 2.23M | More aggressive compression; tests the accuracy/size boundary. |
| Small SpectFormer, optional | 512×512; width 256–384; 4–6 blocks; compatible head count | Target below 25M; measure after construction | Tests whether preserving architectural structure helps feature transfer. |

The two existing CNNs concatenate time and channel dimensions, giving 26 input channels and one output logit. Verify exact counts in the final implementation, including heads. A two-branch temporal encoder or extra high-resolution active-region branch is a later ablation, not needed for the first comparison.

Keep all 13 channels initially. Compare 512 and 1024 resolution after selecting the main loss; test 256 only if further compression is required. Use aligned full-disk downsampling with a documented pooling and normalization order. Since nonlinear normalization and pooling do not commute, keep preprocessing identical across student baselines and version it in the manifest/cache.

Teacher inference remains at its trained 4096 resolution. At patch size 16, that is 65,536 spatial tokens, compared with 1,024 at resolution 512. Reduced token count is not a guaranteed speedup factor: Surya uses spectral and long-short attention, and actual runtime includes I/O and kernel overhead.

**7. Distillation objectives and training**

Let z_t and z_s be teacher and student binary logits, y the observed label, and T the distillation temperature. Start with

```text
p_t = sigmoid(z_t / T)
L = (1-alpha) * BCEWithLogits(z_s, y)
    + alpha * T^2 * BCEWithLogits(z_s / T, p_t)
    + beta * L_feature
```

The soft-label BCE has the same student gradients as Bernoulli KL up to a teacher-only constant. Do not apply softmax to a one-element logit; it would always return one. The hard-label loss uses unscaled student logits. Compute losses in FP32 for stability even when the model forward uses mixed precision.

Use beta=0 first. Then match the teacher's 1280-dimensional class-token representation before its penultimate linear layer, using a trainable projection from the student's pooled feature. A normalized cosine or mean-squared feature loss avoids dependence on arbitrary feature scale. The projection can be discarded at deployment unless it is part of the prediction path.

Initial search: T in {1, 2, 4}; alpha in {0.25, 0.5, 0.75}; a small beta search after inspecting feature-loss scale. Start at T=2, alpha=0.5. Select a few settings on the pilot before expanding, rather than running the full Cartesian product. Keep supervised baseline tuning effort comparable.

For optimization, start with AdamW, effective batch 16–32 as memory allows, a short warmup and cosine decay, 30 epochs with validation-based checkpoint selection. Pilot learning rates such as 1e-4 and 3e-4; extend the schedule only if learning curves justify it. These are proposed initial settings, not established optima. Use three random seeds for final comparisons.

Begin with unweighted BCE on the natural training distribution. Investigate a positive-class weight computed only from training labels as an explicit ablation. If weighted supervision or sampling improves rare-event recall, also examine its effect on calibration. Do not automatically apply class weighting to the teacher probability target. Keep temperature used for KD distinct from any post-training calibration temperature.

Start with deterministic views and no aggressive augmentation. Arbitrary crops may omit the responsible active region, and flips or rotations of vector magnetic channels require physically consistent component transformations. If adding augmentations, either cache matching teacher views or explicitly define and test cross-view distillation. Reduced-channel students with a full-channel teacher are an intentional privileged-information experiment, not an interchangeable cached view.

**8. Cache the teacher once**

Run the frozen teacher over eligible training inputs, storing raw FP32 logits and pooled FP16 features. Cache validation outputs separately for evaluation and model selection; keep final-test outputs outside all training and selection paths. Save sample identifiers, checkpoint hash, preprocessing hash, precision, channel/time ordering, and feature-layer name. Raw logits permit changing T without rerunning the teacher.

At 100,000 examples, a single FP32 logit plus a 1280-value FP16 feature costs about 256 MB before metadata. In contrast, a full 65,536×1280 FP16 token map is 160 MiB per example, about 15.3 TiB for 100,000 examples. Start with pooled features. An optional 8×8 spatial feature grid is around 15.3 GiB for 100,000 examples, per saved layer.

First validate the cached path against online teacher outputs on a small fixed subset. The caching precision and any BF16 teacher execution must preserve useful probabilities relative to the reference FP32 evaluation. Keep FFT operations in the precision required by the current spectral implementation.

Convert each source frame once into the student representation and store unique frames where practical, allowing neighboring forecast samples to share them. Chunked arrays or shards reduce repeated NetCDF decompression. The original source data can be staged in manageable blocks while permanent manifests and student caches are retained.

**9. Experiments that answer the research question**

| Experiment | Student | Objective | What it isolates |
|---|---|---|---|
| T | Released flare teacher | Evaluation only | Reference quality, calibration, memory, and speed. |
| B0 | Training-prevalence forecast; optional past-only flare-persistence model | Simple baseline | Whether the learned models add useful skill. |
| B1 | ResNet18, 512 | Labels only | Strong directly supervised small-model baseline. |
| D1 | Same ResNet18 | Labels + teacher logits | Effect of ordinary KD. |
| D2 | Same ResNet18 | Labels + logits + pooled features | Additional representation transfer. |
| B2 / D3 | MobileNetV2, 512 | Labels only / selected KD loss | Robustness of the improvement to student capacity. |
| R | Selected student, 1024 | Labels only / selected KD loss | Separate resolution effects from KD effects. |
| U, optional | Selected student | Feature pretraining on additional training-period imagery, then task learning | Value of unlabeled representation transfer. |

Match data, architecture, resolution, initialization policy, optimization budget, and tuning opportunity for each paired comparison. If U sees extra inputs, include an appropriate additional-data control and account for the extra teacher compute. Do not attribute all gains to a loss change.

Possible second-stage contribution: compare pooled features with an aligned spatial feature grid, or add a full-disk plus local-region student to preserve fine magnetic structure. Evaluate region selection using information available at forecast issuance. Avoid adding several losses, channels, resolutions, and architectures simultaneously before establishing which component helps.

**10. Evaluation and statistical evidence**

Predeclare TSS at a validation-selected threshold and average precision as primary metrics. Report HSS, ROC-AUC, recall, precision, false-alarm rate, confusion counts, and Brier score with reliability diagrams. Define whether PR-AUC means average precision or trapezoidal integration and keep that definition fixed. Use a training-derived climatology reference for Brier skill when reported.

For each model, choose its threshold and any calibration only on validation data, then freeze them for test. Report an operationally useful recall-versus-false-alarm tradeoff, not only the threshold maximizing TSS. Never select the threshold using the test labels.

Evaluate each test year separately as well as pooled. In the local labels, the positive-window fraction changes from roughly 0.55% in 2020 to 69.69% in 2024; these are windows, not proportions of unique events. This shift makes pooled accuracy and even pooled average precision insufficient descriptions of performance.

Use paired time-block bootstrap confidence intervals for student-minus-teacher and KD-minus-supervised differences, because adjacent hourly forecasts and 24-hour labels overlap. Approximately 14–27-day blocks are a starting choice to validate for temporal dependence. Group by flare/active-region episodes where feasible. Report seed variation separately; three seeds are not three independent datasets. If a yearly split has very few independent positive episodes, report the limitation and wide interval rather than relying on a point estimate.

Further methodological reading: [Ahmadzadeh et al., How to Train Your Flare Prediction Model](https://arxiv.org/abs/2103.07542) on temporal coherence and sampling, and [Leka et al., operational forecasting comparison](https://arxiv.org/abs/1907.02905) on assessing forecast skill and reliability.

Measure parameter count, serialized student size, MACs/FLOPs with a stated counting convention, peak allocated/reserved GPU memory, batch-one median/p95 latency, throughput, and end-to-end latency including preprocessing and reads. Use the same hardware, precision policy, warmup, and synchronization for comparisons. Checkpoint file size includes buffers and serialization overhead and is not a direct parameter count. If later using quantization, report its gains separately and recalibrate/evaluate after conversion.

**11. Server and storage plan for the available GH200**

The supplied `nvidia-smi` shows one GH200 named “120GB,” with 97,871 MiB exposed GPU memory: about 95.6 GiB. Budget from the reported usable amount. The inspected execution host is aarch64 and reports approximately 238 GiB system RAM and a 1.8 TiB local `/tmp` filesystem. Those are host observations, not guaranteed per-job allocations or storage quotas. The earlier training warning showed only one CPU available to that job, so request CPUs explicitly for new runs.

| Resource | Initial plan |
|---|---|
| GPU | One GH200 for teacher caching and student training in separate phases. Start teacher batch size 1; select student batch by measurement. |
| CPU | Request 8–16 allocated cores for the pilot, potentially 16–32 for parallel preprocessing after profiling. |
| Host RAM | Request 64–128 GiB initially; increase if measured decompression and prefetch peaks require it. |
| Fast local storage | Use a few hundred GiB for a pilot and up to about 1 TiB for active staging/cache, subject to allocation policy and free space. |
| Persistent storage | Plan roughly 1–3 TiB for a broad 512-resolution student study, depending on unique-frame reuse, retained resolutions, and compression; much more if retaining full-resolution source data. Verify quota separately. |
| Software | Pin the functioning aarch64-compatible PyTorch/CUDA, torchvision, PEFT, xarray/h5netcdf, normalization, and data-library environment. Record exact versions. |

One pair of 13-channel frames requires the following uncompressed FP16 storage; these are tensor calculations, not GPU training peaks:

| Resolution | MiB per two-frame sample | TiB per 100,000 pairs |
|---|---:|---:|
| 256×256 | 3.25 | 0.31 |
| 512×512 | 13 | 1.24 |
| 1024×1024 | 52 | 4.96 |
| 4096×4096 | 832 | 79.35 |

The teacher's FP32 two-frame input alone is about 1.625 GiB. Activations, spectral FFT workspace, intermediate allocations, and preprocessing copies increase the actual peak substantially. Local source files are around 640 MiB each; 100,000 unique source frames would be about 61 TiB. Thus data movement and source retention can dominate the project despite a capable GPU. There is no need to retain the whole core archive for a staged task-specific study.

This single GPU is a reasonable platform for the proposed workflow; actual fit and throughput must be measured. More GPUs can shorten independent cache-generation shards and student sweeps. Ordinary DDP replicates the model and does not pool GPU memory to solve a per-device out-of-memory failure.

Begin with a 10–20 GPU-hour profiling/pilot allocation rather than committing to a large sweep. Measure teacher seconds/example including I/O and student examples/second from real data. Estimate:

```text
teacher GPU-hours = number_of_cached_examples * seconds_per_example / 3600
student GPU-hours = epochs * training_examples / examples_per_second / 3600
total = teacher caching + all student runs + validation + preprocessing + margin
```

Illustration only: 60,000 teacher examples at 5 seconds each cost 83.3 GPU-hours. A 30-epoch student at 80 examples/second costs 6.25 GPU-hours on that set. Eighteen such runs cost 112.5 GPU-hours, before validation and other overhead. None of those throughput values has been measured on this GH200. Replace them with pilot measurements; CPU preprocessing and network transfer may extend wall-clock time without consuming equivalent GPU-hours.

**12. Milestones and proposed implementation layout**

| Milestone | Approximate effort after data access | Exit condition |
|---|---|---|
| Task/data/teacher audit | 1 week | Exact labels and splits, checkpoint provenance recorded, teacher loads and predicts, data manifest validated. |
| Classification loader and profile | 1 week | No future-image dependency, correct metrics, full evaluation coverage, real memory/I/O timing. |
| Pilot cache and baseline | 1 week | Teacher/cache agreement, supervised student learns, viable cost estimate. |
| KD and feature experiments | 2–3 weeks | Paired baselines and ablations, three-seed final runs, chosen resolution. |
| Locked test and analysis | 1–2 weeks | Per-year metrics, confidence intervals, calibration, latency, reproducible report. |

Roughly 6–8 weeks is a planning range once data access works, not a deadline guarantee. Data transfer, queue limits, and teacher provenance can change it.

Keep the new implementation separate from the example training path initially:

```text
distillation/
  build_manifest.py       # split, timestamp, quality and file audit
  prepare_student_data.py # deterministic downsampling/normalization/cache
  dataset.py              # input-only classifier dataset
  teacher.py              # strict checkpoint loading and selected features
  cache_teacher.py        # resumable, versioned logits/features
  students.py             # existing CNN wrappers and optional small transformer
  losses.py               # binary soft-target and feature losses
  train_student.py        # controlled training and complete checkpoints
  evaluate.py             # full manifests, calibration, grouped uncertainty
  benchmark.py            # synchronized timing and memory measurement
  configs/               # independent teacher, student and experiment settings
```

Save model/optimizer/scheduler/scaler state, random seeds, environment and source revision, data/cache hashes, threshold/calibrator, and preprocessing configuration. The final artifact must load and infer without the teacher. Report teacher-cache cost separately from deployment savings.

The first concrete experiment is ResNet18 at 512×512 on two 13-channel frames, comparing labels-only supervision with cached-teacher KD, then adding class-token feature matching. Expand the architecture or scientific objective only after this comparison is reliable.
