![Module Logo](logo.png)

# Module 09: Reinforcement Learning for Object Detection

## Overview

This module applies Deep Reinforcement Learning to **object detection** — the task of locating and identifying objects within images by predicting bounding boxes and class labels. Unlike traditional detectors (YOLO, Faster R-CNN) that process the entire image in one pass, RL agents learn to **actively search** for objects through sequential attention shifts, bounding box refinements, and multi-object discovery strategies. This module builds on segmentation skills from Module 08 and introduces spatial reasoning, localization metrics, and multi-object coordination.

## Why RL for Object Detection?

Standard object detectors are powerful but have limitations:

1. **Fixed computation**: Process every region equally, regardless of content
2. **No active search**: Cannot allocate more attention to difficult or cluttered areas
3. **Post-hoc reasoning**: NMS and score thresholding are disconnected from the detection process
4. **No sequential refinement**: One-shot predictions with no iterative correction

RL enables a fundamentally different approach:

- **Active visual search**: The agent decides where to look next, mimicking human gaze patterns
- **Iterative refinement**: Bounding boxes are progressively adjusted rather than predicted in one shot
- **Budget-aware detection**: The agent can trade off computation time vs. detection accuracy
- **Multi-object coordination**: Inhibition-of-return prevents redundant detections

## Mathematical Framework

### Object Detection as an MDP

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma \rangle$$

| Component | Definition |
|-----------|-----------|
| **State** $s_t$ | Image $I$ + current view region + detection history |
| **Action** $a_t$ | Shift attention, adjust bounding box, classify, or terminate |
| **Transition** | Deterministic: new view/box based on action |
| **Reward** | IoU with ground truth, +1 per correct detection, penalties for duplicates |
| **Discount** $\gamma$ | $0.9$–$0.99$; encourages efficient detection |

### Core Detection Metrics

**Intersection over Union (IoU)** for bounding boxes $A = (x_1^A, y_1^A, x_2^A, y_2^A)$ and $B = (x_1^B, y_1^B, x_2^B, y_2^B)$:

$$\text{IoU}(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)}$$

$$\text{Area}(A \cap B) = \max(0, x_2^{\min} - x_1^{\max}) \cdot \max(0, y_2^{\min} - y_1^{\max})$$

**Average Precision (AP)** at IoU threshold $\tau$:

$$\text{AP}_\tau = \int_0^1 p(r)\,dr \approx \sum_{k} (r_{k+1} - r_k) \cdot p_{\text{interp}}(r_{k+1})$$

**Mean Average Precision (mAP):**

$$\text{mAP} = \frac{1}{C}\sum_{c=1}^{C}\text{AP}_c$$

## Module Structure

| Sub-module | Topic | Key Concepts |
|-----------|-------|-------------|
| **01** | Attention-Based Search | Soft attention, spatial transformers, foveal vision |
| **02** | Bounding Box Refinement | IoU/GIoU, box parameterization, NMS, action-based adjustment |
| **03** | Active Object Localization | Hierarchical search tree, information gain, budget-aware detection |
| **04** | Multi-Object Detection | Inhibition of return, Hungarian matching, AP/mAP |
| **05** | Visual Question Answering | Multimodal fusion, question-driven attention, VQA as RL |

## Prerequisites

- **Module 01–06**: Complete RL foundations
- **Module 07**: RL for image enhancement (vision + RL integration)
- **Module 08**: RL for segmentation (pixel/region-level decisions)
- **Math**: Linear algebra, probability, set theory, optimization basics

## Datasets Used

All datasets auto-download or are generated programmatically:

- **MNIST digits on canvas** (synthetic — generated on-the-fly)
- **PASCAL VOC** — `torchvision.datasets.VOCDetection` (auto-download)
- **Synthetic colored shapes** (generated programmatically)
- No user uploads required for any sub-module

## Getting Started

```python
import torchvision
import torch
import numpy as np

# PASCAL VOC detection dataset (auto-download)
voc = torchvision.datasets.VOCDetection(
    root='./data', year='2012', image_set='train',
    download=True
)

# Synthetic MNIST detection scene
mnist = torchvision.datasets.MNIST(root='./data', download=True,
                                    transform=torchvision.transforms.ToTensor())

def create_detection_scene(mnist_dataset, canvas_size=128, num_digits=3):
    """Place MNIST digits on a canvas with bounding box annotations."""
    canvas = torch.zeros(1, canvas_size, canvas_size)
    boxes = []
    labels = []
    for _ in range(num_digits):
        idx = np.random.randint(len(mnist_dataset))
        digit, label = mnist_dataset[idx]
        x = np.random.randint(0, canvas_size - 28)
        y = np.random.randint(0, canvas_size - 28)
        canvas[0, y:y+28, x:x+28] = torch.max(
            canvas[0, y:y+28, x:x+28], digit[0]
        )
        boxes.append([x, y, x+28, y+28])
        labels.append(label)
    return canvas, boxes, labels
```

## Next Steps

After completing this module, proceed to **Module 10: RL for Image Generation**, where RL agents learn to guide generative models (GANs, diffusion models) to produce high-quality images.
