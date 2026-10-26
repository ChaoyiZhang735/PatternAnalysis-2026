# ADNI MRI Classification: Baseline CNN vs. ConvNeXt-Tiny

**Course:** COMP3710 Pattern Recognition (2026)  
**Project:** AD versus cognitively normal (CN/NC) classification  
**Selected difficulty:** Hard — ConvNeXt-Tiny  
**Implementation:** PyTorch / TorchVision  
**Status:** Both models implemented, trained and evaluated on a held-out patient-independent test set. **Tutor feasibility check-off remains to be confirmed.**

> **Research prototype, not a medical device.** Predicted scores are not validated clinical probabilities. No result in this project justifies autonomous diagnosis or bypassing clinical review.

## 1. Project overview and engineering question

Alzheimer's disease (AD) can produce structural changes in brain MRI, but anatomical variability and imaging differences make automated recognition challenging. This project classifies course-provided, preprocessed two-dimensional ADNI MRI slices into **cognitively normal (NC = 0)** or **AD (1)**. We implement (1) a lightweight CNN trained from scratch and (2) a **ConvNeXt-Tiny** model initialized from ImageNet-1K weights and fine-tuned for two classes. We compare them using identical patient-disjoint data partitions, preprocessing and evaluation metrics.

**Engineering dilemma:** Does a substantially larger modern convolutional network achieve better recognition and more dependable confidence estimates than a small CNN, and do its benefits justify the greater computational cost? Beyond accuracy, we examine false negatives, calibration, selective human review and specific failure cases.

![Experimental workflow](figures/workflow.png)

**Main result:** ConvNeXt achieved **83.74% slice-level** and **83.33% patient-level** accuracy on the independent test set, compared with **80.29%** and **78.43%** for the baseline. However, its slice-level expected calibration error (ECE) was higher (**0.1066 vs. 0.0350**), and **403 of its 725 erroneous slice predictions** had confidence at least 0.90. Accuracy gains alone therefore do not establish dependable triage.

## 2. Feasibility review (compulsory check-off)

**Consultation status:** Planned for **23 October 2026**; **not yet recorded as completed**. The official midpoint deadline must be checked against the course profile. This written section is intended for the tutor's required one-page feasibility discussion; an updated one-page handout is provided separately.

### 2.1 User need, scope and testable acceptance criteria

The intended user is a medical-image reviewer investigating whether an *experimental* classification assistant could prioritize uncertain AD/CN MRI cases for expert attention. The prototype is not intended for clinical deployment. The measurable criteria are:

1. **Data integrity:** all selected images have valid metadata and decodable pixels, with **zero patient overlap** among training, validation and test partitions.
2. **Recognition:** aim for at least **0.80 ConvNeXt held-out test accuracy**, and report class-wise precision/recall, macro-F1, AUROC and AD false negatives for both models.
3. **Confidence-aware routing:** report ECE and Brier score, choose an illustrative review threshold from validation results, and quantify coverage, accepted-case accuracy and remaining errors on the independent test set.
4. **Auditable failures:** inspect 3–5 real false negatives/positives, relate slice predictions to their patient-level outcomes, and separate visual observations from unverified medical explanations.
5. **Operational feasibility:** work within an **8 GB RTX 4060 Laptop GPU**, measuring parameters, allocated CUDA memory, training time and forward-only inference latency.

### 2.2 Model choice and course concepts

The dataset contains 2D grayscale ADNI JPEG slices. A small four-stage CNN provides an interpretable baseline. ConvNeXt-Tiny uses a modern convolutional backbone (including depthwise convolutions and residual-style stages); its ImageNet classification head is replaced with a two-class head. Both use supervised cross-entropy loss, AdamW optimization, validation-driven checkpoint selection and the same input preprocessing. Patient-level separation addresses correlated slices and repeated scans. The model choice follows Liu et al. [2] and the course's Hard-difficulty classification option [1]. **Whether using the TorchVision implementation satisfies the course's required depth of implementation must be confirmed with the tutor.**

### 2.3 Preliminary and completed feasibility evidence

The full image audit successfully decoded **30,520** grayscale JPEG slices (**256 × 240**) and mapped them to **680 patients**. The source-folder train/test split shared **216 patients**; rebuilding splits by patient (seed 42) removed overlap. The PyTorch data-loader smoke test produced `(8, 3, 224, 224)` batches on the CUDA-capable laptop. Both networks completed forward/backward smoke tests and full training. On the final independent test set, ConvNeXt met the 0.80 slice-accuracy target (**0.8374**), with the limitations discussed in Sections 6–10. These are observed results, not predictions.

### 2.4 Risks, budget and fallback

Key risks are scanner/slice-position differences, small independent patient sample sizes, overfitting, severe overconfidence, use of a changed split rather than the original course split, and domain mismatch between ImageNet pretraining and MRI. We control leakage at patient level, select checkpoints/thresholds from validation results only and avoid making clinical claims. Training was feasible with mixed precision at batch size 8; the measured ConvNeXt peak **PyTorch allocated** training memory was **903.2 MiB** (not total device VRAM usage). If retraining is needed, possible fallbacks are reducing batch size, freezing backbone layers, stronger regularization or a lower learning rate. Any such changes would be separately logged and must not be selected retrospectively using the already-evaluated test set.

**Questions for tutor:** Are the revised patient-disjoint split and direct TorchVision ConvNeXt-Tiny fine-tuning acceptable for this assessment? Has the mandatory check-off been formally recorded, and is the consultation within the course deadline?

## 3. Dataset, labels and leakage prevention

### 3.1 Source and labels

The course-provided ADNI data were available at `/home/groups/comp3710/ADNI`. This project uses the preprocessed JPEG subset in `AD_NC/train/{AD,NC}` and `AD_NC/test/{AD,NC}`. The accompanying `meta_data_with_label.json` identifies scans and patients. The original JSON label **0** corresponds to NC, original label **2** corresponds to AD, and label **1** is not included in this binary task. During training, the labels are remapped to **NC = 0**, **AD = 1**.

Each JPEG filename encodes a scan identifier and slice index. The patient identifier is parsed from the matching scan's metadata path. Because a patient can contribute multiple scans and many correlated slices, splitting individual images would leak patient-specific information into evaluation.

### 3.2 Data audit and corrected partitions

The initial audit found **216 patients in both source `train` and source `test`**, despite no repeated scan IDs across these folders. We therefore **did not use the source-folder train/test assignment as the experimental split**. Instead, `dataset.py` creates a reproducible patient-stratified split using random seed **42**, targeting approximately **70% / 15% / 15% of patients**, while keeping each patient's scans and slices together. Source images are left in place; CSV manifests record their *new* assignment. Thus, a filename beginning with `train/` can legitimately appear in the **new** test manifest.

| New split | NC patients | AD patients | Total patients | NC slices | AD slices | Total slices |
|---|---:|---:|---:|---:|---:|---:|
| Train | 321 | 155 | **476** | 11,140 | 10,220 | **21,360** |
| Validation | 69 | 33 | **102** | 2,180 | 2,520 | **4,700** |
| Test | 69 | 33 | **102** | 2,340 | 2,120 | **4,460** |
| **Total** | **459** | **221** | **680** | **15,660** | **14,860** | **30,520** |

Checks performed: all 30,520 source JPEGs decoded, each image assigned exactly once, image dimensions/modes checked, folder labels matched metadata, patient diagnoses checked for conflicts, and **zero shared patient IDs, scan IDs or image paths** confirmed across every new pair of splits. These are checks on the downloaded images, not cryptographic proof of an identical server copy. The 38 AD/NC metadata scans without downloaded JPEGs were excluded; the working dataset is defined by actual images present.

**Split limitation:** replacing the original split may affect direct comparability with course examples or published work. The justification and acceptability should be discussed with the tutor.

### 3.3 Input preprocessing

For **both** networks, each input is converted to grayscale, resized while preserving aspect ratio, black-padded to **224 × 224**, replicated into three channels, converted to a tensor and normalized with ImageNet channel means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`. These three channels do **not** represent three MRI modalities; they encode the same source grayscale image. The comparison uses the same deterministic preprocessing. No random data augmentation was enabled in the recorded full-training runs, and no test-time augmentation was used.

## 4. Models and training method

### 4.1 Baseline CNN

The baseline is a four-block network with channel progression **3 → 16 → 32 → 64 → 128**. Each `ConvBlock` contains two `3 × 3` convolutions, batch normalization, ReLU and `2 × 2` max pooling. An adaptive `4 × 4` average pool preserves a coarse spatial feature grid, followed by a `64`-unit hidden linear layer, dropout (`p = 0.3`) and two raw output logits. **425,746 parameters**; initialized randomly and trained from scratch.

### 4.2 ConvNeXt-Tiny

`modules.py` constructs TorchVision `convnext_tiny` and replaces its final classifier layer with a two-output `torch.nn.Linear`. The recorded model uses **ImageNet-1K V1 pretrained weights** and fine-tunes the network for AD/NC. **27,821,666 parameters**. The architecture is **reused from TorchVision** rather than implemented independently layer-by-layer; this is disclosed for clarity and attribution [2, 3].

The baseline and ConvNeXt both output logits `[batch, 2]`. The loss is `CrossEntropyLoss` on unnormalized logits; softmax is applied only when probabilities are needed for evaluation.

### 4.3 Experimental settings and checkpoint policy

| Setting | Baseline CNN | ConvNeXt-Tiny |
|---|---|---|
| Initialization | Random | ImageNet-1K V1 pretrained |
| Optimizer | AdamW | AdamW |
| Learning rate | `1e-3` | `1e-4` |
| Weight decay | `1e-4` | `1e-4` |
| Batch size | 8 | 8 |
| Maximum epochs | 12 | 12 |
| Early stopping patience | 5 | 4 |
| Epochs completed | 11 | 6 |
| Best checkpoint | Epoch **6** | Epoch **2** |
| Selection rule | Lowest validation cross-entropy | Lowest validation cross-entropy |
| Seed | 42 | 42 |
| Training | Full train manifest | Full train manifest |
| Final evaluation | Same held-out test manifest | Same held-out test manifest |

Both training runs used CUDA automatic mixed precision. We tracked loss, slice accuracy, macro-F1 and patient accuracy on validation. **No test samples were used for fitting, early stopping, checkpoint choice or the confidence threshold.** The classifier comparison is not a controlled architectural ablation: ConvNeXt receives transfer learning, whereas the baseline does not.

![Training and validation loss](figures/training_loss.png)

The baseline's best validation loss was **0.4415** (epoch 6); ConvNeXt's was **0.6564** (epoch 2). Later training produced higher training accuracy without sustained validation-loss improvement, consistent with overfitting and/or overly sharp wrong predictions. The abrupt baseline validation-loss spike at epoch 5 is an observation, not a diagnosed cause.

The measured local environment used Windows PowerShell, **Python 3.12.10**, **PyTorch 2.8.0+cu126**, **TorchVision 0.23.0+cu126**, **Pillow 12.3.0**, and **Matplotlib 3.11.2**. CUDA was available using an **NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)**. The experiments reported in this README were run in this environment.
## 5. Implementation and reproducibility

### 5.1 Environmen
The measured local environment used Windows PowerShell, **Python 3.12.10**, **PyTorch 2.8.0+cu126**, **TorchVision 0.23.0+cu126**, **Pillow 12.3.0**, and **Matplotlib 3.11.2**. CUDA was available using an **NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)**. The experiments reported in this README were run in this environment.

### 5.2 Local setup and data manifest generation (PowerShell)

Run in the project directory `recognition/ConvNeXt_s4983318`. The example paths follow the local setup used in the reported experiments.

```powershell
cd "$HOME\COMP3710\PatternAnalysis-2026\recognition\ConvNeXt_s4983318"
$py = "$HOME\COMP3710\.venv\Scripts\python.exe"
$data = "$HOME\Downloads\COMP3710\ADNI\AD_NC"
$metadata = "$HOME\Downloads\COMP3710\meta_data_with_label.json"
$manifests = "$HOME\Downloads\COMP3710\prepared_seed42"
$runs = "$HOME\Downloads\COMP3710\runs"

& $py -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
& $py -m pip freeze | Select-String "^(torch|torchvision|Pillow|matplotlib)=="
```

Only for **a fresh preparation** (the existing `prepared_seed42` directory already contains the validated splits):

```powershell
& $py .\dataset.py --data-root $data --metadata $metadata `
  --output-dir "$HOME\Downloads\COMP3710\prepared_seed42_reproduction" `
  --seed 42 --expected-images 30520
```

Use the previously prepared `$manifests` for reproducing the exact existing experiments; do not overwrite its CSVs. The following smoke tests can be rerun without retraining a full model:

```powershell
& $py .\test_dataset.py --data-root $data --manifest-dir $manifests --batch-size 8
& $py .\test_models.py --data-root $data --manifest-dir $manifests --batch-size 8 --device cuda
& $py .\test_convnext.py --data-root $data --manifest-dir $manifests --batch-size 2 --device cuda
```

### 5.3 Full training commands

Use new, empty output folders when reproducing; `train.py` refuses to overwrite populated run directories. This example uses **new `*_repro` folder names** to avoid overwriting the recorded checkpoints.

```powershell
& $py .\train.py --model baseline `
  --data-root $data --manifest-dir $manifests `
  --output-dir "$runs\baseline_repro_seed42" `
  --epochs 12 --batch-size 8 --lr 0.001 --patience 5 --device cuda

& $py .\train.py --model convnext --pretrained `
  --data-root $data --manifest-dir $manifests `
  --output-dir "$runs\convnext_repro_seed42_bs8" `
  --epochs 12 --batch-size 8 --lr 0.0001 --patience 4 --device cuda
```

The checkpoints actually used for the reported results were:

- `$runs\baseline_full_seed42\best_baseline.pt` (epoch 6)
- `$runs\convnext_full_seed42_bs8\best_convnext.pt` (epoch 2)

### 5.4 Reproduce the recorded final evaluations

The final evaluation script requires an explicit `--confirm-test` to reduce accidental test-set access. The review threshold **0.90 was considered using validation outputs and held fixed** for test reporting.

```powershell
& $py .\predict.py --model baseline `
  --checkpoint "$runs\baseline_full_seed42\best_baseline.pt" `
  --data-root $data --manifest-dir $manifests `
  --output-dir "$runs\eval_baseline_test_repro" `
  --split test --confirm-test --batch-size 8 --device cuda

& $py .\predict.py --model convnext `
  --checkpoint "$runs\convnext_full_seed42_bs8\best_convnext.pt" `
  --data-root $data --manifest-dir $manifests `
  --output-dir "$runs\eval_convnext_test_repro" `
  --split test --confirm-test --batch-size 8 --device cuda
```

Outputs include `evaluation.json`, `slice_predictions.csv`, `patient_predictions.csv`, confidence/review plots, `failure_cases.csv` and an illustrative `failure_examples.png`. Generated patient/slice prediction tables and original MRI images must **not** be committed to public GitHub. The analytic plots included in this README use aggregated statistics only.

## 6. Quantitative evaluation: independent test data

The final test partition contains **4,460 slices from 102 patients**. The checkpoints were chosen using validation loss before the test was evaluated. Slice-level results treat each slice as a prediction; patient-level results take the **mean predicted AD probability across that patient's available slices** and classify as AD when the mean is at least 0.5. Correlated slices are not statistically independent observations.

![Held-out test results](figures/test_performance.png)

### 6.1 Slice-level test results

| Metric | Baseline CNN | ConvNeXt-Tiny |
|---|---:|---:|
| Accuracy | 0.8029 | **0.8374** |
| Macro-F1 | 0.8029 | **0.8374** |
| AUROC | 0.8934 | **0.9240** |
| NC precision | 0.8406 | **0.8996** |
| NC recall | 0.7705 | **0.7769** |
| AD precision | 0.7680 | **0.7860** |
| AD recall | 0.8387 | **0.9042** |
| AD false negatives (AD → NC) | 342 | **203** |
| NC false positives (NC → AD) | 537 | **522** |

Confusion matrices use **true rows × predicted columns**, ordered `[NC, AD]`:

- **Baseline:** `[[1803, 537], [342, 1778]]`
- **ConvNeXt:** `[[1818, 522], [203, 1917]]`

ConvNeXt improves slice accuracy by **3.45 percentage points** and AD recall by about **6.56 points**, while meeting the course's **0.80 test accuracy** target. These are descriptive single-split results, not estimates of statistical significance.

### 6.2 Patient-level test results

| Metric | Baseline CNN | ConvNeXt-Tiny |
|---|---:|---:|
| Accuracy | 0.7843 (80/102) | **0.8333 (85/102)** |
| Macro-F1 | 0.7667 | **0.8227** |
| AUROC | 0.9043 | **0.9372** |
| AD recall | 0.7879 (26/33) | **0.9091 (30/33)** |
| Missed AD patients | 7 | **3** |
| NC patients incorrectly flagged AD | 15 | **14** |

- **Baseline patient confusion:** `[[54, 15], [7, 26]]`
- **ConvNeXt patient confusion:** `[[55, 14], [3, 30]]`

Patient-level aggregation corrects some slice-level errors, but **17/102** patients remain incorrectly classified by ConvNeXt. Because the independent test contains only 102 patients, performance estimates are uncertain, and external data would be required before clinical claims.

## 7. Confidence, calibration and an illustrative review rule

Top-label confidence is `max(P(NC), P(AD))`. **ECE** compares average reported confidence with accuracy across confidence bins (smaller is better). The **Brier score** measures squared error in the predicted AD probability (also smaller is better). These evaluate different aspects of prediction quality and need not rank models identically.

![Calibration comparison](figures/calibration_comparison.png)

| Test metric | Baseline CNN | ConvNeXt-Tiny |
|---|---:|---:|
| Slice ECE ↓ | **0.0350** | 0.1066 |
| Slice Brier score ↓ | 0.1419 | **0.1312** |
| Wrong slices with confidence ≥0.90 | **138** | 403 |
| Total wrong slices | 879 | **725** |
| Patient ECE ↓ | 0.0751 | **0.0505** |

Despite making **154 fewer slice-level errors**, ConvNeXt made **403 errors at ≥0.90 confidence** (55.6% of its erroneous slices), compared with **138** for the baseline (15.7% of its erroneous slices). This is important evidence of **overconfident failures**. However, ConvNeXt's better test Brier score cautions against claiming that it is globally worse calibrated under every measure.

### 7.1 Locked patient-level review policy

Using **validation-set** results, we selected the same **illustrative threshold of 0.90** for both models:

1. Aggregate a patient's slice-level probabilities: `p_AD = mean(P(AD) over slices)`.
2. Predict `AD` if `p_AD ≥ 0.5`, else `NC`.
3. Compute `confidence = max(p_AD, 1 - p_AD)`.
4. If `confidence < 0.90`, **refer to human review**; otherwise, count the classification as provisionally *accepted* for this simulation.

| Independent test at fixed threshold 0.90 | Baseline CNN | ConvNeXt-Tiny |
|---|---:|---:|
| Accepted patient classifications | 25 / 102 | **57 / 102** |
| Review referrals | 77 / 102 | **45 / 102** |
| Accepted-case accuracy | 96.0% (24/25) | **96.49% (55/57)** |
| Incorrect accepted classifications | 1 | 2 |
| Accepted AD → NC errors | 0 | 0 |
| Accepted NC → AD errors | 1 | 2 |

ConvNeXt would send fewer cases to review at a similar observed accepted-case accuracy; however, **two confidently wrong NC-to-AD decisions remain**. The absence of accepted AD false negatives in this small test **does not guarantee safety**. The threshold is not medically calibrated, and the term “accepted” is only part of a research simulation: actual clinical decisions require professional review and external validation.

## 8. Failure case autopsy: five actual test-set slice errors

Five errors were chosen deterministically by the evaluation script: two high-confidence **AD → NC** (false negatives), two high-confidence **NC → AD** (false positives), and one low-confidence/borderline incorrect slice. They correspond to **five distinct patients**. The table checks whether each slice error persists after patient-level aggregation. Individual patient identifiers and original MRI pictures are intentionally excluded from this public README pending data-governance review.

| Case | True → predicted slice | Slice confidence | Patient-level result | Patient `P(AD)` | Patient confidence | Review at 0.90 |
|---|---|---:|---|---:|---:|---|
| 1 | AD → NC | 0.99985 | **Wrong: AD → NC** | 0.3712 | 0.6288 | Referred |
| 2 | AD → NC | 0.99969 | **Correct: AD → AD** | 0.7259 | 0.7259 | Referred |
| 3 | NC → AD | 0.99997 | **Wrong: NC → AD** | 0.9886 | 0.9886 | **Wrongly accepted** |
| 4 | NC → AD | 0.99995 | **Wrong: NC → AD** | 0.8284 | 0.8284 | Referred |
| 5 | NC → AD | 0.50146 | **Correct: NC → NC** | 0.0978 | 0.9022 | Correctly accepted |

**Cases 1–2 (high-confidence false negatives).** Both single slices were labelled AD but confidently predicted NC. Their visible brain cross sections have different spatial coverage and contrast, illustrating substantial within-task anatomical and acquisition variation. Patient averaging corrected **Case 2** but not **Case 1**. A plausible *hypothesis* is slice-position sensitivity; the images alone cannot establish causation or the underlying neuropathology.

**Cases 3–4 (high-confidence false positives).** Both NC slices were incorrectly predicted AD with near-unit confidence. The selected slices vary in visible tissue structure and intensity. Potential texture, anatomical-sampling or scanner-related shortcuts merit investigation, but are **not proven explanations**. In **Case 3**, the patient was also wrongly predicted AD with confidence **0.9886**, so the 0.90 referral rule would miss this false alarm. **Case 4** would be referred because its patient confidence was 0.8284.

**Case 5 (borderline slice error).** The selected NC slice received **P(AD) = 0.50146**; the individual slice fell just across the classification boundary. The patient's average across slices was **P(AD) = 0.0978**, correctly predicting NC with confidence 0.9022. This provides a concrete example of how aggregation can correct a borderline per-slice error.

**What can and cannot be concluded:** differences in anatomical coverage, brightness and local shape are visible; whether the network used those differences to decide AD/NC is unknown. A controlled follow-up could study per-scan aggregation, stratification by slice position or scan type, and Grad-CAM/feature attribution. Such work must not be presented as a demonstrated causal explanation or a radiologist's diagnosis.

## 9. Computational cost and deployment trade-off

All measurements below came from the same CUDA-capable laptop using batch size **8** for the benchmark. **Forward-only latency** uses an already-transferred GPU batch and excludes disk reads, host-to-device transfer, postprocessing and visualization.

| Resource | Baseline CNN | ConvNeXt-Tiny | Interpretation |
|---|---:|---:|---|
| Parameters | 425,746 | 27,821,666 | ConvNeXt has **65.3×** more parameters |
| Training epochs completed | 11 | 6 | Both stopped early |
| Recorded training time | 615.2 s | ~830.5 s* | ConvNeXt has more expensive epochs |
| Peak PyTorch allocated CUDA memory, training | 152.9 MiB | 903.2 MiB | **Allocated by PyTorch**, not full GPU usage |
| Forward-only median latency per slice | **0.354 ms** | 0.967 ms | ConvNeXt **~2.7×** slower |
| Forward-only peak PyTorch allocated CUDA memory | 42.0 MiB | 184.0 MiB | Separate from training memory |

*ConvNeXt training duration is the **sum of six epoch times from the recorded console output**, rather than a reported final wall-clock total; overhead is excluded. The baseline duration is from its saved `summary.json`. These measurements characterize one device, batch size and software environment, not a deployment benchmark.

ConvNeXt offers better recognition and higher patient-level review coverage, but its size, latency and high-confidence failures impose real costs. Training-time memory figures are not directly comparable to forward-only inference figures. No CPU throughput, multi-site robustness, or deployment power draw was measured.

## 10. Discussion, limitations and engineering recommendation

**Recommendation:** Advance ConvNeXt-Tiny as the stronger **research baseline for supervised ADNI AD/NC classification**, not as an autonomous clinical screening tool. Relative to the smaller CNN, it improves held-out slice accuracy by **3.45 percentage points**, patient accuracy by **4.90 points**, and patient AUROC from **0.9043 to 0.9372**. Under the validation-selected 0.90 threshold, it provisionally covers **57/102** test patients rather than **25/102** for the baseline. These benefits are counterbalanced by **~65×** parameters, **~2.7×** forward latency and conspicuous overconfident slice errors.

**Limits to generalization:** (i) only 102 independent test patients; (ii) 2D slices and average pooling over all slices may overweight patients with more scans; (iii) the new patient-level split is different from the course's source-folder split; (iv) ConvNeXt uses transfer learning whereas the baseline is trained from scratch; (v) reported calibration has not been corrected by temperature scaling or externally validated; (vi) error cases do not establish anatomical causes; (vii) the models were evaluated on one source dataset and one random split; and (viii) existing test labels are not a substitute for clinical adjudication.

**Next engineering steps, not completed in this report:** confirm implementation/split acceptability with the tutor; compare models under equal pretraining conditions, fit temperature scaling using validation data only, test robustness across sites/scans and patient subgroups, investigate per-scan versus per-patient aggregation, estimate patient-level confidence intervals, and commission clinical review before considering real triage. These are proposals, not claims of completed experiments.

## 11. Repository organization and submission checklist

```text
recognition/ConvNeXt_s4983318/
├── README.md                # Technical report and reproducibility
├── modules.py               # CNN components and TorchVision ConvNeXt builder
├── dataset.py               # Audit, patient-level split, Dataset and DataLoader
├── train.py                 # Training, early stopping, logging and weights
├── predict.py               # Evaluation, confidence and error analysis
├── test_dataset.py          # DataLoader smoke test
├── test_models.py           # Baseline forward/backward smoke test
├── test_convnext.py         # ConvNeXt forward/backward smoke test
└── figures/
    ├── workflow.png
    ├── training_loss.png
    ├── test_performance.png
    └── calibration_comparison.png
```

The following administrative steps must be completed before the final course submission:

- [ ] **Feasibility check-off:** present the updated one-page review to the tutor and confirm recorded completion **within the course-approved deadline**.
- [ ] Ask whether the **patient-disjoint re-split** and **TorchVision ConvNeXt** approach satisfy course requirements; document any tutor feedback honestly.
- [ ] Insert exact locally installed **Pillow and Matplotlib** versions (run `pip freeze` shown in Section 5).
- [ ] Verify the Markdown, diagrams and command examples in the GitHub preview; check that the plots match actual logs.
- [ ] Commit code, README and **aggregate-statistics-only figures** in incremental, meaningful Git commits. **Never commit ADNI images, patient IDs, manifests, predictions or model weight files.**
- [ ] Open the required pull request from the personal fork's `topic-recognition` to the course repository's `topic-recognition` **before the assessment deadline**.
- [ ] Export the final `README.md` to PDF for Turnitin and submit it.
- [ ] Review, respond to and incorporate tutor/maintainer PR feedback within the announced response window.

## 12. Artificial Intelligence Usage Disclosure

**AI tools used:** OpenAI ChatGPT.

**Tasks assisted:** experimental planning and task breakdown; PowerShell/Python environment debugging; patient-level leakage audit and data-splitting script scaffolding; PyTorch `Dataset`/`DataLoader` implementation; BaselineCNN and TorchVision ConvNeXt integration; training/evaluation script scaffolding; plotting and draft technical writing; generating candidate hypotheses and error-analysis structure.

**Human verification and responsibility:** The user ran all reported analyses locally on the course ADNI data; confirmed the 30,520-image decode audit and zero-overlap split; tested real DataLoader batches; verified CUDA forward/backward passes for each model; completed full training and saved checkpoint selection; and ran both models on the independent held-out test set. Reported metrics were cross-checked against saved `history.csv`, `summary.json` and model-specific `evaluation.json` outputs. The five failure examples were inspected together with their CSV entries and patient-level aggregation outcomes. The original source files were not edited during splitting. Unverified anatomical explanations are described explicitly as hypotheses. All code, numerical claims and the final submission must still be reviewed and owned by the student; AI-generated text or scripts are **not** evidence of independent correctness without these checks.

## References

1. The University of Queensland, COMP3710 / COMP7710 *Pattern Recognition — Working Prototype Project in the Era of Artificial Intelligence*, assessment specification version 2.4 (2026), Sections 2.1, 3–6 and 8.
2. Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell and S. Xie, “A ConvNet for the 2020s,” *Proceedings of CVPR*, pp. 11976–11986, 2022. [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html).
3. PyTorch/TorchVision, *ConvNeXt model implementation and pretrained weights*, `torchvision.models.convnext_tiny` / `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`. [Documentation](https://pytorch.org/vision/stable/models/convnext.html).
4. Alzheimer's Disease Neuroimaging Initiative (ADNI), dataset information. [ADNI](https://adni.loni.usc.edu/).
