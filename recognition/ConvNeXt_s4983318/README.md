# ADNI Classification with ConvNeXt-Tiny

## Project plan
- Course: COMP3710
- Selected difficulty: Hard
- Task: Classify Alzheimer's disease (AD) versus cognitively normal (CN).
- Proposed model: ConvNeXt-Tiny.
- Baseline: A small convolutional neural network.
- Research question: Does ConvNeXt improve classification and confidence reliability compared with the baseline?

## Current progress
- Local environment: Python 3.12.10, PyTorch 2.8.0+cu126.
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM.
- CUDA tensor computation verified.
- Course ADNI dataset downloaded and validated.

## Planned evaluation
Split data by patient to prevent leakage.
Compare classification metrics, confidence calibration, and computational cost.
Inspect 3–5 representative failure cases.

## Artificial Intelligence Usage Disclosure
ChatGPT assisted with project planning, environment setup, and data audit/splitting script scaffolding.
The Python, Git, and GPU checks were executed locally, and CUDA tensor computation succeeded.
Model implementation and experiments have not yet been completed.


## Data Audit and Leakage Prevention

- Dataset: ADNI AD/CN classification
- Images: 30,520 grayscale JPEG slices (256 x 240)
- Patients: 680
- Original train/test split: 216 overlapping patients
- New split: patient-level stratified split with seed 42
- Training: 476 patients, 21,360 images
- Validation: 102 patients, 4,700 images
- Test: 102 patients, 4,460 images
- Patient overlap between new splits: zero
- All images successfully decoded and assigned once

The original split was not used because of patient leakage.
Both baseline CNN and ConvNeXt will use the same new split.

The AI-assisted data preparation script was tested locally
against the complete downloaded image dataset.

