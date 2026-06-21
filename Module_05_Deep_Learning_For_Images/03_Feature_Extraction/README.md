![Module Logo](../logo.png)

# Feature Extraction

## Overview

Deep CNNs learn hierarchical feature representations where early layers detect low-level features (edges, textures) and deeper layers compose them into high-level concepts (objects, scenes). This document develops the theory of feature hierarchy emergence, the information bottleneck principle for understanding deep representations, Fisher information and feature selection, and the complete derivation of Grad-CAM (Gradient-weighted Class Activation Mapping).

## Prerequisites

- CNN architecture (Module 05.2)
- Information theory basics (entropy, mutual information)
- Calculus (gradients, chain rule)

---

## 1. Mathematical Foundations

### 1.1 Core Definition — Feature Representation

A feature extractor is a learned mapping $\phi: \mathcal{X} \to \mathcal{Z}$ from the input space to a feature space. For a CNN with $L$ layers, each layer $l$ defines a feature map:

$$\phi_l: \mathbb{R}^{C_0 \times H_0 \times W_0} \to \mathbb{R}^{C_l \times H_l \times W_l}$$

The feature at layer $l$ for input $\mathbf{x}$ is $\mathbf{z}_l = \phi_l(\mathbf{x})$.

### 1.2 Feature Hierarchy Emergence Theory

**Observation:** Trained CNNs exhibit a characteristic hierarchy:
- **Layer 1:** Gabor-like filters (edges, oriented gratings)
- **Layer 2–3:** Textures, patterns, corners
- **Layer 4–5:** Object parts (eyes, wheels)
- **Deeper layers:** Whole objects, scenes

**Theoretical justification via compositionality:**

**Step 1:** Natural images have a compositional structure: objects are composed of parts, parts of sub-parts, down to edges and textures.

**Step 2:** A deep network with $L$ layers can represent functions of the form:

$$f = g_L \circ g_{L-1} \circ \cdots \circ g_1$$

**Step 3:** Each layer composes features from the previous layer. The number of distinct patterns representable by an $L$-layer network with $n$ neurons per layer scales as:

$$N_{\text{patterns}} \sim n^{2^L}$$

(doubly exponential in depth), whereas a single-layer network can only represent $O(n)$ patterns. This exponential advantage of depth explains why deeper networks are more efficient.

### 1.3 Information Bottleneck Principle

**Definition:** The information bottleneck (IB) minimizes:

$$\mathcal{L}_{\text{IB}} = I(X; Z) - \beta \, I(Z; Y)$$

where $I(\cdot;\cdot)$ denotes mutual information, $X$ is the input, $Y$ is the target, and $Z$ is the learned representation.

**Derivation of the IB Lagrangian:**

**Step 1:** We want a representation $Z$ that preserves as much information about $Y$ as possible (high $I(Z;Y)$) while compressing the input (low $I(X;Z)$).

**Step 2:** By the data processing inequality, $I(Z;Y) \leq I(X;Y)$ (no representation can create information about $Y$ that wasn't in $X$).

**Step 3:** The IB trade-off is parameterized by $\beta > 0$. The optimal encoder $p(z|x)$ satisfies:

$$p^*(z|x) = \frac{p(z)}{Z(\beta, x)} \exp\!\left(\beta \, D_{\text{KL}}[p(y|x) \| p(y|z)]\right)$$

**Step 4:** As $\beta \to \infty$: maximize $I(Z;Y)$ without compression constraint (equivalent to supervised learning).

**Step 5:** As $\beta \to 0$: maximize compression (trivial representation $Z = \text{const}$).

**Connection to deep learning:** Each layer of a trained network approximately solves the IB problem at a different $\beta$: early layers have high $I(X;Z)$ (preserve input details), later layers have low $I(X;Z)$ but high $I(Z;Y)$ (task-relevant abstraction).

### 1.4 Fisher Information and Feature Selection

**Definition:** The Fisher Information Matrix (FIM) for a parametric model $p(y|\mathbf{z};\theta)$ is:

$$\mathbf{F}(\theta) = \mathbb{E}\left[\nabla_\theta \log p(y|\mathbf{z};\theta) \cdot \nabla_\theta \log p(y|\mathbf{z};\theta)^T\right]$$

**Connection to feature quality:**

**Step 1:** The Cramér–Rao bound states that the covariance of any unbiased estimator $\hat{\theta}$ satisfies:

$$\text{Cov}(\hat{\theta}) \succeq \mathbf{F}(\theta)^{-1}$$

**Step 2:** Features $\mathbf{z}$ that produce a large Fisher information matrix enable more precise estimation of $\theta$ — these are "informative" features.

**Step 3:** The Fisher information of feature $z_j$ about the class label is:

$$F_j = \mathbb{E}\left[\left(\frac{\partial \log p(y|z_j)}{\partial z_j}\right)^2\right]$$

Features with high $F_j$ are most discriminative and should be retained; features with low $F_j$ can be pruned.

### 1.5 Grad-CAM Derivation

**Goal:** Given a trained CNN and a target class $c$, identify which spatial regions of the input image most influence the prediction.

**Step 1:** Let $A^k \in \mathbb{R}^{H \times W}$ be the activation map of the $k$-th channel in the last convolutional layer. Let $y^c$ be the score (pre-softmax) for class $c$.

**Step 2:** Compute the gradient of $y^c$ with respect to each activation map:

$$\frac{\partial y^c}{\partial A^k_{ij}}$$

**Step 3:** Global average pool the gradients to get importance weights:

$$\alpha_k^c = \frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W}\frac{\partial y^c}{\partial A^k_{ij}}$$

**Derivation of why GAP works:** This approximates the contribution of feature map $k$ to class $c$. The class score can be approximated (first-order Taylor) as:

$$y^c \approx \sum_k \left(\frac{1}{HW}\sum_{ij}\frac{\partial y^c}{\partial A^k_{ij}}\right)\left(\sum_{ij}A^k_{ij}\right) = \sum_k \alpha_k^c \cdot \text{GAP}(A^k)$$

**Step 4:** Compute the Grad-CAM heatmap:

$$L^c_{\text{Grad-CAM}} = \text{ReLU}\!\left(\sum_k \alpha_k^c A^k\right)$$

**Step 5:** The ReLU ensures we only consider features with positive influence on class $c$ (negative influences correspond to features of other classes).

**Result:**

$$\boxed{L^c_{\text{Grad-CAM}} = \text{ReLU}\!\left(\sum_k \alpha_k^c A^k\right), \quad \alpha_k^c = \frac{1}{HW}\sum_{ij}\frac{\partial y^c}{\partial A^k_{ij}}}$$
$\blacksquare$

---

## 2. Algorithm / Method

### 2.1 Pseudocode: Grad-CAM

```
Algorithm: Grad_CAM
Input: Image x, trained CNN, target class c
Output: Heatmap L_c of spatial importance

1. FORWARD: Compute activations A^k at target layer
2. SCORE: Compute class score y^c
3. BACKWARD: Compute gradients ∂y^c/∂A^k
4. WEIGHTS: α_k = (1/HW) Σ_{ij} ∂y^c/∂A^k_{ij}
5. COMBINE: L_c = ReLU(Σ_k α_k · A^k)
6. UPSAMPLE: Resize L_c to input resolution
7. Return L_c
```

### 2.2 Complexity Analysis

- **Grad-CAM:** $O(C \cdot H \cdot W)$ — one forward + backward pass through the last conv layer
- **Space:** $O(C \cdot H \cdot W)$ for storing feature maps and gradients

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

- **State:** $s_t = \phi(\mathbf{I}_t)$ — extracted feature representation of the image
- **Action:** Policy output based on features
- **Reward:** Task-specific
- **Transition:** Feature extraction is part of the state encoding

### 3.2 Why RL?

Feature quality directly determines RL performance. Grad-CAM can visualize what the RL agent "looks at" — critical for debugging image-based policies. The information bottleneck principle explains why RL agents should learn compressed, task-relevant state representations.

---

## 4. Dataset

- **Name:** ImageNet (pre-trained features) or CIFAR-10
- **Size:** 1.2M images (ImageNet) or 60K (CIFAR-10)
- **Auto-download:**

```python
from torchvision import models, transforms
model = models.resnet18(pretrained=True)
model.eval()
```

---

## 5. Key Equations Summary

| Equation | Meaning |
|----------|---------|
| $\phi_l: \mathcal{X} \to \mathcal{Z}_l$ | Layer-wise feature mapping |
| $\mathcal{L}_{\text{IB}} = I(X;Z) - \beta I(Z;Y)$ | Information bottleneck |
| $\mathbf{F} = \mathbb{E}[\nabla\log p \cdot \nabla\log p^T]$ | Fisher information matrix |
| $L^c = \text{ReLU}(\sum_k \alpha_k^c A^k)$ | Grad-CAM heatmap |
| $\alpha_k^c = \text{GAP}(\partial y^c/\partial A^k)$ | Channel importance weight |

---

## 6. References

- Selvaraju, R. R. et al. "Grad-CAM: Visual Explanations from Deep Networks," *ICCV*, 2017.
- Tishby, N. & Zaslavsky, N. "Deep Learning and the Information Bottleneck Principle," *ITW*, 2015.
- Zeiler, M. D. & Fergus, R. "Visualizing and Understanding Convolutional Networks," *ECCV*, 2014.
- Montavon, G. et al. "Methods for Interpreting and Understanding Deep Neural Networks," *Digital Signal Processing*, 2018.
