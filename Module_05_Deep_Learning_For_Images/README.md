![Module Logo](logo.png)

# Module 05: Deep Learning for Images

## Overview

This module bridges classical image processing with modern deep learning, establishing the neural network foundations required for Deep Reinforcement Learning in later modules. We progress from single neurons through fully-connected networks to Convolutional Neural Networks (CNNs), covering the complete mathematical machinery of backpropagation, feature extraction, transfer learning, and end-to-end image classification.

**Why this matters for RL:** In Deep RL (Module 06+), the agent's policy network must process raw pixel inputs and output actions. CNNs serve as the **state encoder** — transforming high-dimensional image observations into compact feature vectors that RL algorithms can work with. Every concept here directly feeds into DQN, Policy Gradient, and Actor-Critic architectures.

---

## Module Structure

| # | Topic | Key Math | Folder |
|---|-------|----------|--------|
| 5.1 | Neural Network Basics | Backprop chain rule, Universal Approximation | `01_Neural_Network_Basics/` |
| 5.2 | CNN from Scratch | Convolution math, equivariance, receptive fields | `02_CNN_From_Scratch/` |
| 5.3 | Feature Extraction | Grad-CAM, PCA, cosine similarity | `03_Feature_Extraction/` |
| 5.4 | Transfer Learning | Domain adaptation bounds, fine-tuning theory | `04_Transfer_Learning/` |
| 5.5 | Image Classification | Cross-entropy, BatchNorm, dropout as Bayes | `05_Image_Classification/` |

---

## The Deep Learning Pipeline for Vision

```
Raw Image             Feature Extraction           Task Output
[H × W × C] ──CNN──▶ [Feature Vector] ──FC──▶ [Class Probabilities]
   ↑                        ↑                         ↑
   │                        │                         │
Pixel space           Learned representation     Decision space
(Module 01-02)        (Module 05)               (Module 06: RL actions)
```

## Mathematical Thread

The core mathematical thread running through this module:

$$f_\theta: \mathbb{R}^{H \times W \times C} \rightarrow \mathbb{R}^K$$

We learn parameters $\theta$ such that $f_\theta$ maps images to useful representations by minimizing:

$$\theta^* = \arg\min_\theta \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}(f_\theta(x_i), y_i) + \lambda \Omega(\theta)$$

where $\mathcal{L}$ is the task loss and $\Omega(\theta)$ is regularization.

---

## Prerequisites

- **Module 01-02**: Image fundamentals (pixels, channels, convolution as filtering)
- **Module 03-04**: RL theory (you'll see why neural networks matter for RL)
- **Python**: NumPy, basic linear algebra
- **Math**: Calculus (derivatives, chain rule), linear algebra (matrix multiplication)

## What You'll Build

By the end of this module, you will have implemented:
1. A neural network from scratch (NumPy only) trained on MNIST
2. A CNN with conv/pool/FC layers trained on CIFAR-10
3. Feature visualization and Grad-CAM heatmaps
4. Transfer learning between datasets (CIFAR-10 → STL-10)
5. A full image classification pipeline with BatchNorm, dropout, and learning rate scheduling

## Connection to Deep RL (Module 06)

Every architecture built here becomes a building block in Module 06:

| Module 05 Concept | Module 06 Application |
|---|---|
| CNN feature extractor | DQN state encoder |
| Backpropagation | Policy gradient updates |
| Transfer learning | Pre-trained encoders for RL |
| BatchNorm + Dropout | Stabilizing RL training |
| Classification head | Q-value / policy heads |

---

## Datasets Used

All datasets **auto-download** via `torchvision.datasets` — no manual setup needed:

- **MNIST** (28×28 grayscale digits, 70K images)
- **CIFAR-10** (32×32 color images, 60K images, 10 classes)
- **STL-10** (96×96 color images, transfer learning target)
- **Pre-trained ImageNet weights** (via `torchvision.models`)
