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
- Course ADNI dataset download in progress.

## Planned evaluation
Split data by patient to prevent leakage.
Compare classification metrics, confidence calibration, and computational cost.
Inspect 3–5 representative failure cases.

## Artificial Intelligence Usage Disclosure
ChatGPT assisted with project planning and environment setup.
The Python, Git, and GPU checks were executed locally, and CUDA tensor computation succeeded.
Model implementation and experiments have not yet been completed.
