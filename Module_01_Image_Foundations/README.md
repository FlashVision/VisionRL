![Module Logo](logo.png)

# Module 01: Image Foundations

## Overview

This module builds the mathematical and conceptual foundations of digital image processing from first principles. Starting with the physics of image formation and ending with statistical analysis of pixel distributions, you will develop a rigorous understanding of how images are represented, stored, and analyzed as mathematical objects. Every concept is grounded in the theory that later enables Reinforcement Learning agents to perceive, interpret, and manipulate visual data.

---

## Prerequisites

- **Mathematics**: Linear algebra (matrices, eigenvalues, SVD), calculus (partial derivatives, integrals), probability theory (PDFs, CDFs, expected values)
- **Programming**: Python 3.8+, familiarity with NumPy array operations
- **Libraries**: NumPy, Matplotlib, scikit-image, OpenCV (all installable via pip)

---

## Learning Path

| # | Sub-Topic | Core Concepts | Estimated Time |
|---|-----------|---------------|----------------|
| 1.1 | [What Is An Image](./01_What_Is_An_Image/) | Image formation equation, sampling theorem, quantization | 2–3 hours |
| 1.2 | [Pixels and Channels](./02_Pixels_And_Channels/) | Pixel addressing, channel decomposition, memory layout, bit depth | 2–3 hours |
| 1.3 | [Color Spaces](./03_Color_Spaces/) | RGB, HSV, LAB, YCbCr — full conversion derivations | 3–4 hours |
| 1.4 | [Image as Matrix](./04_Image_As_Matrix/) | Linear algebra of images — SVD, eigendecomposition, rank approximation | 3–4 hours |
| 1.5 | [Image Histograms](./05_Image_Histograms/) | Probability distributions, CDF, histogram equalization proof | 2–3 hours |

---

## Module Structure

```
Module_01_Image_Foundations/
├── README.md                          ← You are here
├── 01_What_Is_An_Image/
│   ├── README.md                      ← Theory & math deep-dive
│   └── what_is_an_image.ipynb         ← Hands-on notebook
├── 02_Pixels_And_Channels/
│   ├── README.md
│   └── pixels_and_channels.ipynb
├── 03_Color_Spaces/
│   ├── README.md
│   └── color_spaces.ipynb
├── 04_Image_As_Matrix/
│   ├── README.md
│   └── image_as_matrix.ipynb
└── 05_Image_Histograms/
    ├── README.md
    └── image_histograms.ipynb
```

---

## Key Mathematical Themes

Throughout this module, five mathematical pillars recur:

1. **Continuous → Discrete**: Sampling theory (Nyquist–Shannon) governs how we faithfully capture continuous signals as discrete arrays.

2. **Linear Algebra**: Images are matrices; matrix decompositions (SVD, eigendecomposition) reveal structure, enable compression, and define the state space for RL agents.

3. **Coordinate Transformations**: Color space conversions are linear or nonlinear maps between coordinate systems, each optimized for different perceptual or computational goals.

4. **Probability & Statistics**: Histograms are empirical probability distributions; equalization is a monotonic CDF transform that reshapes them.

5. **Signal Processing**: Images are 2D signals; understanding their frequency content and sampling constraints is essential before applying filters (Module 02).

---

## Connection to Reinforcement Learning

Every concept in this module maps directly to RL components:

| Image Concept | RL Component |
|---------------|--------------|
| Pixel tensor $\mathbf{I} \in \mathbb{R}^{H \times W \times C}$ | **State** $s_t$ — what the agent observes |
| Histogram statistics (mean, variance, entropy) | **State features** — compact representations |
| Color space conversion | **State preprocessing** — choosing the right observation space |
| SVD rank-$k$ approximation | **Dimensionality reduction** — tractable state spaces |
| Image quality metrics (contrast, entropy) | **Reward signals** $r_t$ — what the agent optimizes |

An RL agent that processes images must understand what it is looking at. This module ensures that understanding is mathematically rigorous.

---

## Datasets Used

All sub-topics use **auto-downloading datasets** — no manual uploads required:

- **scikit-image built-in images**: `astronaut()`, `camera()`, `coins()`, `coffee()`, `chelsea()` — loaded with a single function call
- **MNIST digits** (Topic 1.4): Auto-downloaded via `torchvision.datasets.MNIST(download=True)`
- **Synthetic images**: Generated programmatically (gradients, patterns, noise) for controlled experiments

---

## How to Study This Module

1. **Read the README** for each sub-topic to understand the theory and math
2. **Work through the notebook** cell by cell, running every code block
3. **Verify the math**: Check that code outputs match the equations
4. **Experiment**: Modify parameters, try different images, observe the effects
5. **Summarize**: Write down the key equations and their RL connections before moving on

---

## Next Steps

After completing Module 01, proceed to **[Module 02: Image Processing Basics](../Module_02_Image_Processing_Basics/)**, where you will apply filters, detect edges, perform morphological operations, apply geometric transformations, and denoise images — all building toward the action space of RL agents.
