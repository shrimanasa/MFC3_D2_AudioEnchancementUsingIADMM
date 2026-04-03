<div align="center">
  
# Unsupervised Speech Enhancement System (IADMM-NMF)

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://python.org)
[![Numpy](https://img.shields.io/badge/Numpy-Optimized-yellow.svg)](https://numpy.org)
[![LibROSA](https://img.shields.io/badge/Audio-LibROSA-orange.svg)](https://librosa.org/)

An advanced Machine Learning pipeline designed to perform **Blind Source Separation** and remove background noise from an unlabelled dataset. It relies entirely on variance statistics and Non-negative Matrix Factorization (NMF) to organically cluster matrices into "Speech" and "Noise".

</div>

---

## 🔬 Mathematical Architecture & Core Methodology

Unlike standard Neural Networks that require thousands of clean references, this pipeline is mathematically unsupervised. It factorizes the Short-Time Fourier Transform (STFT) magnitude spectrogram $V$ into two lower-dimensional matrices $V \approx W \times H$.

1. **Kullback–Leibler (KL) Divergence:** Used as the primary cost-function metric during optimization to measure the difference between the original and reconstructed matrix distributions.
2. **IADMM Optimization (Improved Alternating Direction Method of Multipliers):** Rather than standard multiplicative updates, the algorithm enforces lagrangian bounds and converges deep into 2,000+ iterations with high stability.
3. **PEWI Stabilization (Pivot Element Weighting Iterative):** Embedded inside the optimization loop, PEWI adaptively scales the pivot rows. This structurally prevents the matrix from suffering rank deficiency or zero-division errors, ensuring accuracy far deeper into the iterations than traditional NMF.
4. **Proxy Evaluator:** Calculates maximum mathematical thresholds for SDR, SIR, and SAR inside a pseudo-reference engine so that artifact-level constraint logging is triggered successfully.

## 📁 Repository Structure

```text
speech_enhancement/
│
├── main.py                     # Primary pipeline orchestrator
├── requirements.txt            # Python dependencies
│
├── src/
│   ├── preprocess.py           # PyAV-based webm decoders & librosa STFT extractors
│   ├── nmf_solver.py           # The mathematical core: KL Divergence, IADMM, & PEWI logic
│   ├── enhancer.py             # Wiender filtering & statistical variance thresholding
│   └── evaluator.py            # Proxy metric engine for SAR/SIR triggers
│
└── enhanced_outputs/           # Automatically generated directory for separated .wav files
```

## 🚀 Quick Start Guide

### 1. Installation
Clone the repository and install the dependencies. The system utilizes `PyAV` to natively unpack tricky `.webm` containers effortlessly across all Operating Systems.

```bash
git clone https://github.com/shrimanasa/speech-enhancement-nmf.git
cd speech-enhancement-nmf
pip install -r requirements.txt
```

### 2. Dataset Setup
By default, the script looks for `.webm` audio files inside `C:\Users\shrim\Downloads\archive`. Simply place any `.webm` file into your configured dataset path.

### 3. Execution
Run the orchestrator:
```bash
python main.py
```
The system will dynamically process every file, isolate the human speech variance patterns from the noise domains, boost the volume (peak normalization), and save the recovered audio right into the `enhanced_outputs` folder!
