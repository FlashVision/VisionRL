![Module Logo](logo.png)

# Module 08: Reinforcement Learning for Image Segmentation

## Overview

This module applies Deep Reinforcement Learning to **image segmentation** — the task of dividing an image into meaningful, non-overlapping regions. Unlike supervised segmentation networks that predict all pixel labels in a single forward pass, RL agents learn to *sequentially* build segmentations through interactive decisions: classifying pixels, growing regions, tracing boundaries, and refining predictions. This sequential approach enables more interpretable, controllable, and adaptive segmentation strategies.

## Why RL for Segmentation?

Traditional segmentation methods fall into two paradigms:

1. **Classical methods** (thresholding, region growing, graph cuts): Rely on hand-designed criteria, struggle with complex scenes
2. **Deep learning** (FCN, U-Net, DeepLab): Powerful but require massive labeled datasets, produce non-interpretable outputs

RL offers a **third paradigm** where segmentation emerges from learned decision-making:

- The agent **actively explores** the image, making pixel/region-level decisions
- **Reward signals** guide the agent toward accurate segmentation (IoU, Dice)
- **Sequential processing** allows the agent to focus on difficult regions
- **Interactive segmentation** naturally maps to RL (human clicks as rewards)

## Mathematical Framework

### Segmentation as an MDP

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma \rangle$$

| Component | Definition |
|-----------|-----------|
| **State** $s_t$ | Image $I$ + current partial segmentation mask $M_t$ |
| **Action** $a_t$ | Segment a pixel, grow a region, place a boundary point, etc. |
| **Transition** | $M_{t+1} = \text{update}(M_t, a_t)$ — deterministic mask update |
| **Reward** | $r_t = \text{IoU}(M_{t+1}, M^*) - \text{IoU}(M_t, M^*)$ — improvement in overlap |
| **Discount** $\gamma$ | $0.95$–$0.99$ to encourage efficient segmentation |

### Core Segmentation Metrics

**Intersection over Union (IoU / Jaccard Index):**

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}$$

**Dice Coefficient (F1 Score):**

$$\text{Dice}(A, B) = \frac{2|A \cap B|}{|A| + |B|}$$

**Relationship:** $\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}$

**Mean IoU (mIoU)** for $C$ classes:

$$\text{mIoU} = \frac{1}{C}\sum_{c=1}^{C} \text{IoU}_c$$

## Module Structure

| Sub-module | Topic | Key Concepts |
|-----------|-------|-------------|
| **01** | Pixel Classification Agent | Per-pixel MDP, IoU/Dice reward, patch-based state |
| **02** | Region Growing Agent | Homogeneity criteria, merge decisions, graph segmentation |
| **03** | Boundary Detection Agent | Active contours, level sets, edge tracing |
| **04** | Semantic Segmentation RL | FCN, skip connections, mIoU, per-pixel rewards |
| **05** | Interactive Segmentation | Click-based RL, GrabCut, information gain |

## Prerequisites

- **Module 01–06**: Complete RL foundations
- **Module 07**: RL for image enhancement (state/action/reward design for vision)
- **Python**: NumPy, OpenCV, PyTorch
- **Math**: Set theory, probability, graph theory basics

## Datasets Used

All datasets auto-download with no manual intervention:

- **Oxford-IIIT Pet** — `torchvision.datasets.OxfordIIITPet` (37 categories, segmentation masks)
- **PASCAL VOC 2012** — `torchvision.datasets.VOCSegmentation` (20 classes, standard benchmark)
- **scikit-image** built-in images (`coins`, `cells`, etc.)
- **Synthetic shapes** (generated programmatically — no download at all)

## Getting Started

```python
import torchvision

# Oxford Pets with segmentation masks (auto-download)
pets = torchvision.datasets.OxfordIIITPet(
    root='./data', split='trainval', target_types='segmentation',
    download=True
)

# VOC 2012 segmentation (auto-download)
voc = torchvision.datasets.VOCSegmentation(
    root='./data', year='2012', image_set='train',
    download=True
)
```

## Next Steps

After completing this module, proceed to **Module 09: RL for Object Detection**, where agents learn to locate and identify objects using attention-based search, bounding box refinement, and multi-object strategies.
