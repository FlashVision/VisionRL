![Module Logo](logo.png)

# Module 02: Image Processing Basics

## Overview

This module covers the fundamental operations that transform images: spatial filtering, edge detection, morphological analysis, geometric transformations, and denoising. Each operation is derived from its mathematical foundations — convolution theory, gradient calculus, set theory, projective geometry, and statistical estimation. These operations form the **action space** of RL agents: every action an image-processing agent can take corresponds to one of the operations developed here.

---

## Prerequisites

- **Module 01: Image Foundations** — image formation, pixel/channel structure, color spaces, matrix algebra, histograms
- **Mathematics**: Calculus (partial derivatives, integrals), linear algebra (matrix operations, eigenvalues), probability (PDFs, estimation), set theory basics
- **Programming**: Python 3.8+, NumPy, familiarity with array operations

---

## Learning Path

| # | Sub-Topic | Core Concepts | Estimated Time |
|---|-----------|---------------|----------------|
| 2.1 | [Filters & Convolutions](./01_Filters_And_Convolutions/) | Convolution theorem, frequency domain, separable filters, computational complexity | 3–4 hours |
| 2.2 | [Edge Detection](./02_Edge_Detection/) | Gradient operators, Canny algorithm, Laplacian of Gaussian, zero-crossings | 3–4 hours |
| 2.3 | [Morphological Operations](./03_Morphological_Operations/) | Set theory, erosion/dilation duality, opening/closing, granulometry | 3–4 hours |
| 2.4 | [Image Transformations](./04_Image_Transformations/) | Homogeneous coordinates, affine group, interpolation, projective geometry | 3–4 hours |
| 2.5 | [Noise & Denoising](./05_Noise_And_Denoising/) | Noise models, Wiener filter, bilateral filter, NLM, PSNR/SSIM derivations | 4–5 hours |

---

## Module Structure

```
Module_02_Image_Processing_Basics/
├── README.md                                ← You are here
├── 01_Filters_And_Convolutions/
│   ├── README.md                            ← Theory & math deep-dive
│   └── filters_and_convolutions.ipynb       ← Hands-on notebook
├── 02_Edge_Detection/
│   ├── README.md
│   └── edge_detection.ipynb
├── 03_Morphological_Operations/
│   ├── README.md
│   └── morphological_operations.ipynb
├── 04_Image_Transformations/
│   ├── README.md
│   └── image_transformations.ipynb
└── 05_Noise_And_Denoising/
    ├── README.md
    └── noise_and_denoising.ipynb
```

---

## Key Mathematical Themes

### 1. Convolution as the Universal Operation

The discrete 2D convolution $(I * K)[m,n] = \sum_i \sum_j I[m-i, n-j] \cdot K[i,j]$ is the foundational operation of image processing. Blurring, sharpening, edge detection, and denoising are all special cases with different kernels $K$.

### 2. The Frequency Domain Perspective

The Convolution Theorem ($\mathcal{F}\{I * K\} = \mathcal{F}\{I\} \cdot \mathcal{F}\{K\}$) reveals that spatial convolution equals pointwise multiplication in the frequency domain. This duality is the basis for efficient filtering and frequency analysis.

### 3. Mathematical Morphology

Set-theoretic operations (erosion, dilation) provide a non-linear framework complementary to linear filtering. These operations analyze shape and structure, not intensity gradients.

### 4. Geometric Groups

Image transformations (translation, rotation, scaling, affine, projective) form algebraic groups, enabling composition, inversion, and parametric analysis.

### 5. Statistical Estimation

Denoising is an estimation problem: given a noisy observation $y = x + n$, estimate the clean signal $x$. Different noise models and priors lead to different optimal estimators (Wiener, bilateral, NLM).

---

## Connection to Reinforcement Learning

Every concept in this module maps to an RL action or reward component:

| Image Processing Operation | RL Component |
|----------------------------|--------------|
| Convolution kernel $K$ | **Action** — the agent selects which filter to apply |
| Kernel parameters (size, $\sigma$) | **Continuous action space** — the agent chooses parameter values |
| Edge detection output | **State feature** — edge maps inform the agent about image structure |
| Morphological opening/closing | **Discrete action** — shape-based cleanup operations |
| Geometric transformation params | **Action** — crop, rotate, scale as agent decisions |
| Denoising strength | **Action parameter** — how aggressively to filter |
| PSNR, SSIM | **Reward signal** $r_t$ — quantifies the quality of the agent's action |

The goal of Modules 07–10 is to train RL agents that learn **which** operations to apply, in **what order**, with **what parameters** — optimizing image quality metrics as cumulative reward.

---

## Datasets Used

All sub-topics use auto-downloading datasets:

- **CIFAR-10** (Topic 2.1): `torchvision.datasets.CIFAR10(download=True)` — 60,000 tiny color images
- **scikit-image built-in images**: `astronaut()`, `camera()`, `coins()` — loaded instantly
- **Synthetic images**: Gaussian noise, geometric patterns, binary shapes — generated programmatically
- **BSD68 subset**: For denoising benchmarks (auto-downloaded or synthesized from scikit-image)

---

## How to Study This Module

1. **Read each README** for the mathematical theory and derivations
2. **Work through notebooks** cell by cell
3. **Connect operations to RL**: For each operation, ask "How would an RL agent decide to use this?"
4. **Experiment with parameters**: Vary kernel sizes, thresholds, noise levels
5. **Compare approaches**: e.g., Gaussian blur vs. bilateral filter — when does each excel?

---

## Next Steps

After completing Module 02, proceed to **[Module 03: RL Mathematical Foundations](../Module_03_RL_Mathematical_Foundations/)** to learn the formal framework (MDPs, Bellman equations, value functions) that will let you train agents to intelligently apply the image processing operations from this module.
