![Module Logo](logo.png)

# Module 07: Reinforcement Learning for Image Enhancement

## Overview

This module bridges the gap between foundational RL theory and real-world computer vision applications. We apply Deep Reinforcement Learning to solve classical **image enhancement** problems — brightness/contrast adjustment, color correction, denoising, super-resolution, and HDR tone mapping. Instead of hand-tuning parameters or training end-to-end supervised models, an RL agent *learns* to sequentially apply enhancement operations to maximize perceptual image quality metrics.

This is the **first application module** in the VisionRL curriculum. Everything you learned in Modules 01–06 (MDPs, value functions, policy gradients, actor-critic methods, exploration strategies) is now deployed on pixel-level tasks.

## Why RL for Image Enhancement?

Traditional image enhancement pipelines rely on fixed transformations (histogram equalization, gamma correction, Wiener filtering). These methods:

1. **Cannot adapt** to diverse degradation types within a single framework
2. **Require manual tuning** of hyperparameters per image
3. **Lack sequential reasoning** — they apply one-shot corrections

RL provides a fundamentally different paradigm:

- The agent **observes** the current image state
- **Selects** an enhancement action (adjust brightness, reduce noise, sharpen, etc.)
- **Receives reward** based on perceptual quality improvement (PSNR, SSIM, or learned metrics)
- **Iteratively refines** the image over multiple steps

## Mathematical Framework

### Image Enhancement as an MDP

We formalize image enhancement as a Markov Decision Process:

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma \rangle$$

| Component | Definition |
|-----------|-----------|
| **State** $s_t$ | Current image tensor $I_t \in \mathbb{R}^{H \times W \times C}$ (optionally with quality features) |
| **Action** $a_t$ | Enhancement operation with parameters (e.g., increase brightness by $+10$) |
| **Transition** $\mathcal{T}$ | Deterministic: $I_{t+1} = f(I_t, a_t)$ where $f$ is the enhancement function |
| **Reward** $r_t$ | Change in image quality: $r_t = Q(I_{t+1}, I^*) - Q(I_t, I^*)$ |
| **Discount** $\gamma$ | Typically $0.9$–$0.99$; encourages efficient enhancement |

### Core Quality Metrics

**Peak Signal-to-Noise Ratio (PSNR):**

$$\text{PSNR}(I, I^*) = 10 \cdot \log_{10}\!\left(\frac{\text{MAX}^2}{\text{MSE}(I, I^*)}\right)$$

where $\text{MSE} = \frac{1}{HWC}\sum_{i,j,c}(I_{i,j,c} - I^*_{i,j,c})^2$ and $\text{MAX} = 255$ for 8-bit images.

**Structural Similarity Index (SSIM):**

$$\text{SSIM}(x, y) = l(x,y) \cdot c(x,y) \cdot s(x,y)$$

with luminance $l(x,y) = \frac{2\mu_x\mu_y + C_1}{\mu_x^2 + \mu_y^2 + C_1}$, contrast $c(x,y) = \frac{2\sigma_x\sigma_y + C_2}{\sigma_x^2 + \sigma_y^2 + C_2}$, and structure $s(x,y) = \frac{\sigma_{xy} + C_3}{\sigma_x\sigma_y + C_3}$.

## Module Structure

| Sub-module | Topic | Key Concepts |
|-----------|-------|-------------|
| **01** | Brightness & Contrast Agent | DQN, linear transforms, PSNR/SSIM reward |
| **02** | Color Correction Agent | Chromatic adaptation, CIE ΔE2000, LAB space |
| **03** | Denoising Agent | Wiener filter, bilateral filter, NLM, noise models |
| **04** | Super-Resolution Agent | Degradation model, sub-pixel convolution, pixel shuffle |
| **05** | HDR Tone Mapping Agent | Reinhard operator, TMQI, gamma correction |

## Prerequisites

- **Module 01–03**: MDP fundamentals, value-based methods (DQN)
- **Module 04–05**: Policy gradient and actor-critic methods
- **Module 06**: Exploration strategies
- **Python**: NumPy, OpenCV basics, PyTorch tensors
- **Math**: Linear algebra, probability, basic calculus

## Datasets Used

All datasets in this module are **tiny and auto-download** — no manual data upload required:

- **CIFAR-10** (60K 32×32 images, ~163 MB) — `torchvision.datasets.CIFAR10`
- **Set5 / Set14** (5–14 images) — classic super-resolution benchmarks
- **BSD68** (68 images) — standard denoising benchmark
- **scikit-image sample images** — built-in, zero download

## Getting Started

```python
import torch
import torchvision
import torchvision.transforms as transforms

# All datasets auto-download
cifar10 = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True,
    transform=transforms.ToTensor()
)
```

## Next Steps

After completing this module, proceed to **Module 08: RL for Image Segmentation**, where agents learn to divide images into meaningful regions using pixel-level decision making.
