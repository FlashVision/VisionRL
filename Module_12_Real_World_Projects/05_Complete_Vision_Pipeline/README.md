![Module Logo](../logo.png)

# 12.5 — Complete Vision Pipeline (Capstone)

## Overview

This capstone project integrates **everything from the entire course** into a single, end-to-end RL-driven vision system. The pipeline has three sequential stages — Enhancement, Segmentation, and Detection — each controlled by an RL agent. A **hierarchical MDP** coordinates the stages, with a meta-controller allocating computational budget and a total reward that balances quality across all three stages. This is the culmination of 12 modules, demonstrating that RL can orchestrate a complete vision pipeline from raw input to final output.

## Mathematical Foundation

### 1. Full Pipeline as Hierarchical MDP

The pipeline is formalized as a **three-stage hierarchical MDP**:

$$\text{Pipeline} = \text{Stage}_1(\text{Enhance}) \to \text{Stage}_2(\text{Segment}) \to \text{Stage}_3(\text{Detect})$$

Each stage $k$ is itself an MDP with its own state, action, and reward:

| Stage | State $s_t^{(k)}$ | Action $a_t^{(k)}$ | Local Reward $R^{(k)}$ |
|-------|-------------------|---------------------|------------------------|
| Enhancement | Current image quality metrics | Filter/adjustment parameters | $\Delta$PSNR + $\Delta$SSIM |
| Segmentation | Enhanced image + partial mask | Pixel/region labeling | $\Delta$IoU |
| Detection | Segmented image + partial boxes | Box adjustment $(x, y, w, h)$ | $\Delta$mAP |

### 2. Total Pipeline Reward

The total reward combines all stages with learned weights:

$$R_{\text{total}} = w_1 R_{\text{enhance}} + w_2 R_{\text{segment}} + w_3 R_{\text{detect}}$$

where the weights can be:

- **Fixed**: Predetermined based on task importance
- **Learned**: A meta-controller adjusts $(w_1, w_2, w_3)$ based on current performance:

$$w_k^{(t)} = \text{softmax}\left(f_{\text{meta}}(s_{\text{global}}^{(t)})\right)_k$$

**Constraint**: $\sum_k w_k = 1$, $w_k \geq 0$.

### 3. Enhancement Stage: PSNR and SSIM

**Peak Signal-to-Noise Ratio (PSNR)**:

$$\text{PSNR}(\hat{I}, I) = 10 \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}(\hat{I}, I)}\right) = 10 \log_{10}\left(\frac{\text{MAX}^2}{\frac{1}{N}\sum_i (\hat{I}_i - I_i)^2}\right)$$

**Structural Similarity Index (SSIM)**:

$$\text{SSIM}(\hat{I}, I) = \frac{(2\mu_{\hat{I}}\mu_I + C_1)(2\sigma_{\hat{I}I} + C_2)}{(\mu_{\hat{I}}^2 + \mu_I^2 + C_1)(\sigma_{\hat{I}}^2 + \sigma_I^2 + C_2)}$$

where $C_1 = (K_1 L)^2$, $C_2 = (K_2 L)^2$ are stabilization constants ($K_1 = 0.01$, $K_2 = 0.03$, $L = 255$).

**SSIM decomposition** into luminance, contrast, and structure:

$$\text{SSIM} = l(\hat{I}, I)^\alpha \cdot c(\hat{I}, I)^\beta \cdot s(\hat{I}, I)^\gamma$$

$$l = \frac{2\mu_{\hat{I}}\mu_I + C_1}{\mu_{\hat{I}}^2 + \mu_I^2 + C_1}, \quad c = \frac{2\sigma_{\hat{I}}\sigma_I + C_2}{\sigma_{\hat{I}}^2 + \sigma_I^2 + C_2}, \quad s = \frac{\sigma_{\hat{I}I} + C_3}{\sigma_{\hat{I}}\sigma_I + C_3}$$

### 4. Segmentation Stage: IoU / Jaccard Index

**Intersection over Union (IoU)** for class $c$:

$$\text{IoU}_c = \frac{|P_c \cap G_c|}{|P_c \cup G_c|} = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{FN}_c}$$

**Mean IoU (mIoU)** across $C$ classes:

$$\text{mIoU} = \frac{1}{C}\sum_{c=1}^{C} \text{IoU}_c$$

**Dice coefficient** (equivalent to $F_1$ for binary masks):

$$\text{Dice}_c = \frac{2|P_c \cap G_c|}{|P_c| + |G_c|} = \frac{2 \cdot \text{IoU}_c}{1 + \text{IoU}_c}$$

### 5. Detection Stage: mAP

**Average Precision (AP)** for class $c$ at IoU threshold $\tau$:

$$\text{AP}_c^{(\tau)} = \int_0^1 p_c(r) \; dr$$

where $p_c(r)$ is the precision-recall curve. Approximated by the **11-point interpolation**:

$$\text{AP}_c \approx \frac{1}{11} \sum_{r \in \{0, 0.1, \ldots, 1.0\}} \max_{r' \geq r} p_c(r')$$

**Mean Average Precision (mAP)**:

$$\text{mAP} = \frac{1}{C}\sum_{c=1}^{C} \text{AP}_c$$

**mAP@[0.5:0.95]** (COCO-style): Average over IoU thresholds from 0.5 to 0.95 in steps of 0.05:

$$\text{mAP}_{\text{COCO}} = \frac{1}{10}\sum_{\tau \in \{0.5, 0.55, \ldots, 0.95\}} \text{mAP}^{(\tau)}$$

### 6. End-to-End Optimization

The pipeline is optimized end-to-end. The total objective:

$$\max_{\theta_1, \theta_2, \theta_3, \psi} \; \mathbb{E}\left[\sum_{t} \gamma^t R_{\text{total}}^{(t)}\right]$$

where $\theta_1, \theta_2, \theta_3$ are the stage-level agents and $\psi$ is the meta-controller.

**Gradient flow**: The meta-controller receives gradients from all stages:

$$\nabla_\psi J = \sum_{k=1}^{3} \frac{\partial R_{\text{total}}}{\partial w_k} \cdot \nabla_\psi w_k = \sum_{k=1}^{3} R^{(k)} \cdot \nabla_\psi w_k$$

**Stage interaction**: Enhancement quality affects segmentation accuracy (better image → better segmentation), and segmentation quality affects detection (better masks → better bounding boxes):

$$R_{\text{segment}} = f(\hat{I}_{\text{enhanced}}), \quad R_{\text{detect}} = g(\hat{M}_{\text{segmented}})$$

This creates **cascading dependencies** where early-stage errors propagate — the meta-controller learns to invest more in stages with higher downstream impact.

### 7. Computational Budget Allocation

The meta-controller also allocates **compute budget** $B$ across stages:

$$B = B_1 + B_2 + B_3, \quad B_k = \text{max RL steps for stage } k$$

$$\text{Budget allocation}: \quad (B_1, B_2, B_3) = \text{softmax}(f_\psi(s_{\text{global}})) \cdot B_{\text{total}}$$

This balances thoroughness vs. speed — if the image is already well-exposed, the agent allocates fewer steps to enhancement and more to segmentation.

## Step-by-Step Breakdown

1. **Load CIFAR-10**: Auto-download base dataset.
2. **Create synthetic ground truth**: For each image, generate:
   - Degraded version (for enhancement)
   - Simple segmentation masks (based on color clustering)
   - Bounding boxes (based on connected components of segmentation)
3. **Build Stage 1 - Enhancement Agent**: PPO agent adjusting brightness/contrast/noise.
4. **Build Stage 2 - Segmentation Agent**: DQN agent performing region-based pixel labeling.
5. **Build Stage 3 - Detection Agent**: PPO agent refining bounding boxes.
6. **Build Meta-Controller**: Small network that sets stage weights and budget allocation.
7. **Train end-to-end**: Each episode runs all three stages; total reward trains all agents + meta-controller.
8. **Evaluate**: Report per-stage metrics (PSNR, mIoU, mAP) and total pipeline quality; compare against independent (non-coordinated) stage agents.

## Dataset Used

| Property | Value |
|----------|-------|
| **Name** | CIFAR-10 |
| **Source** | `torchvision.datasets.CIFAR10` |
| **Auto-download** | `CIFAR10(root='./data', download=True)` |
| **Size** | 60,000 images (32×32 color), 10 classes |
| **Synthetic labels** | Segmentation masks via k-means on pixel colors; bounding boxes via connected components |
| **Why** | Universally available; diverse enough for a multi-stage pipeline; class labels enable detection evaluation |

```python
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch
import numpy as np
from sklearn.cluster import KMeans

dataset = CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

def generate_pipeline_labels(image, n_segments=3):
    """Generate synthetic segmentation + detection labels."""
    img_np = image.permute(1, 2, 0).numpy().reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=n_segments, random_state=0, n_init=3)
    labels = kmeans.fit_predict(img_np)
    seg_mask = labels.reshape(image.shape[1], image.shape[2])
    
    bboxes = []
    for seg_id in range(n_segments):
        ys, xs = np.where(seg_mask == seg_id)
        if len(ys) > 10:
            bboxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
    
    return seg_mask, bboxes

def degrade_image(image, noise_std=0.15):
    """Apply degradation for enhancement stage."""
    degraded = image * 0.5  # darken
    degraded = degraded + torch.randn_like(degraded) * noise_std  # noise
    return degraded.clamp(0, 1)

img, label = dataset[0]
seg_mask, bboxes = generate_pipeline_labels(img)
degraded = degrade_image(img)
```

## Key Equations Summary

| Equation | Description |
|----------|-------------|
| $R_{\text{total}} = w_1 R_{\text{enhance}} + w_2 R_{\text{segment}} + w_3 R_{\text{detect}}$ | Weighted total reward |
| $\text{PSNR} = 10\log_{10}(\text{MAX}^2 / \text{MSE})$ | Enhancement quality |
| $\text{IoU}_c = \|P_c \cap G_c\| / \|P_c \cup G_c\|$ | Segmentation quality |
| $\text{mAP} = \frac{1}{C}\sum_c \text{AP}_c$ | Detection quality |
| $\text{SSIM} = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}$ | Structural similarity |
| $(B_1,B_2,B_3) = \text{softmax}(f_\psi(s)) \cdot B_{\text{total}}$ | Budget allocation |

## Connection to RL

The complete vision pipeline is the ultimate RL-for-vision demonstration:

- **Hierarchical decision-making**: The meta-controller makes high-level decisions (weight allocation, budget), while stage agents make low-level decisions (pixel adjustments, region labeling, box refinement).
- **Multi-objective optimization**: Balancing enhancement, segmentation, and detection quality is inherently multi-objective — the RL framework handles this via weighted rewards.
- **Cascading dependencies**: Errors propagate through stages — the RL agents learn to invest in early stages to prevent downstream failures.
- **Resource allocation**: Budget allocation across stages is itself a sequential decision problem.
- **End-to-end learning**: Unlike traditional pipelines with independently optimized stages, the RL pipeline is optimized jointly for final output quality.
- **Course integration**: This project uses concepts from every prior module — image processing (M1-2), RL theory (M3-4), deep learning (M5-6), and all application modules (M7-11).

## Prerequisites & Next Steps

**Prerequisites:**
- All prior modules (1–11)
- Modules 7–9 specifically (enhancement, segmentation, detection)
- Module 11.2: Hierarchical RL (for the multi-stage architecture)

**Congratulations!** Completing this capstone means you have mastered the full spectrum of RL for computer vision — from pixel-level operations to end-to-end intelligent vision systems.
