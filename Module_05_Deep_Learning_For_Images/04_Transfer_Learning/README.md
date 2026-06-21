![Module Logo](../logo.png)

# Transfer Learning

## Overview

Transfer learning leverages knowledge from a source domain to improve learning in a target domain, enabling effective deep learning with limited data. This document develops the Ben-David domain adaptation bound, provides a rigorous analysis of fine-tuning versus feature extraction strategies, analyzes catastrophic forgetting through the lens of Fisher information, and derives optimal transfer strategies.

## Prerequisites

- Feature extraction (Module 05.3)
- Neural network training (Module 05.1)
- Statistical learning theory basics (VC dimension, PAC learning)

---

## 1. Mathematical Foundations

### 1.1 Core Definition — Transfer Learning Framework

**Source domain:** $\mathcal{D}_S = \{(\mathbf{x}_i^S, y_i^S)\}_{i=1}^{n_S}$ drawn from $P_S(\mathbf{x}, y)$, typically large.

**Target domain:** $\mathcal{D}_T = \{(\mathbf{x}_j^T, y_j^T)\}_{j=1}^{n_T}$ drawn from $P_T(\mathbf{x}, y)$, typically small ($n_T \ll n_S$).

**Goal:** Learn a hypothesis $h: \mathcal{X} \to \mathcal{Y}$ that performs well on the target domain, using knowledge from the source domain.

### 1.2 Ben-David Domain Adaptation Bound

**Theorem (Ben-David et al., 2010):** For a hypothesis class $\mathcal{H}$ with VC dimension $d$, the target risk of hypothesis $h$ is bounded by:

$$\epsilon_T(h) \leq \epsilon_S(h) + \frac{1}{2}d_{\mathcal{H}\Delta\mathcal{H}}(P_S, P_T) + \lambda^*$$

where:

- $\epsilon_S(h)$ is the source risk
- $d_{\mathcal{H}\Delta\mathcal{H}}(P_S, P_T)$ is the $\mathcal{H}$-divergence between domains
- $\lambda^* = \min_{h \in \mathcal{H}}[\epsilon_S(h) + \epsilon_T(h)]$ is the ideal joint risk

**Derivation of the $\mathcal{H}$-divergence:**

**Step 1:** The $\mathcal{H}$-divergence measures how well a classifier from $\mathcal{H}$ can distinguish source from target:

$$d_{\mathcal{H}}(P_S, P_T) = 2\sup_{h \in \mathcal{H}}|P_S(h(\mathbf{x}) = 1) - P_T(h(\mathbf{x}) = 1)|$$

**Step 2:** The symmetric $\mathcal{H}\Delta\mathcal{H}$-divergence uses the set $\mathcal{H}\Delta\mathcal{H} = \{h \oplus h' : h, h' \in \mathcal{H}\}$:

$$d_{\mathcal{H}\Delta\mathcal{H}}(P_S, P_T) = 2\sup_{h \in \mathcal{H}\Delta\mathcal{H}}|P_S(h(\mathbf{x}) = 1) - P_T(h(\mathbf{x}) = 1)|$$

**Step 3:** This divergence can be estimated empirically by training a domain classifier:

$$\hat{d}_{\mathcal{H}\Delta\mathcal{H}} = 2\left(1 - 2\min_{h \in \mathcal{H}}\left[\frac{1}{n_S}\sum_{i:x_i \in S}\mathbb{1}[h(x_i) = 1] + \frac{1}{n_T}\sum_{j:x_j \in T}\mathbb{1}[h(x_j) = 0]\right]\right)$$

**Proof of the bound:**

**Step 4:** For any $h$ and ideal hypothesis $h^*$:

$$\epsilon_T(h) \leq \epsilon_T(h^*) + \epsilon_T(h, h^*)$$

where $\epsilon_T(h, h^*) = P_T(h(x) \neq h^*(x))$ is the disagreement.

**Step 5:** Relate target disagreement to source disagreement using the $\mathcal{H}$-divergence:

$$|\epsilon_T(h, h^*) - \epsilon_S(h, h^*)| \leq \frac{1}{2}d_{\mathcal{H}\Delta\mathcal{H}}(P_S, P_T)$$

**Step 6:** Since $\epsilon_S(h, h^*) \leq \epsilon_S(h) + \epsilon_S(h^*)$ (triangle inequality for classification error):

$$\epsilon_T(h) \leq \epsilon_S(h) + \frac{1}{2}d_{\mathcal{H}\Delta\mathcal{H}} + \epsilon_S(h^*) + \epsilon_T(h^*) = \epsilon_S(h) + \frac{1}{2}d_{\mathcal{H}\Delta\mathcal{H}} + \lambda^*$$
$\blacksquare$

**Intuition:** Transfer is effective when: (1) the model performs well on the source ($\epsilon_S$ small), (2) the domains are similar ($d_{\mathcal{H}\Delta\mathcal{H}}$ small), and (3) there exists a hypothesis good for both domains ($\lambda^*$ small).

### 1.3 Fine-Tuning vs. Feature Extraction Analysis

**Feature extraction:** Freeze the pre-trained backbone $\phi_\theta$; train only the classifier head $g_\psi$:

$$\hat{y} = g_\psi(\phi_\theta(\mathbf{x}))$$

**Effective sample complexity:** $O(d_{\text{head}} / \varepsilon^2)$ where $d_{\text{head}}$ is the dimension of the classifier head (small).

**Fine-tuning:** Update all parameters $(\theta, \psi)$:

**Effective sample complexity:** $O((d_{\text{backbone}} + d_{\text{head}}) / \varepsilon^2)$ — much larger, requiring more data.

**When to fine-tune (formal criterion):** Fine-tune when the domain divergence $d_{\mathcal{H}\Delta\mathcal{H}}$ is large but the ideal risk $\lambda^*$ is small. Feature extraction suffices when domains are similar.

### 1.4 Catastrophic Forgetting Analysis

**Problem:** Fine-tuning on the target domain degrades performance on the source domain.

**Fisher Information Perspective:**

**Step 1:** The importance of parameter $\theta_i$ for the source task is measured by the diagonal of the Fisher Information Matrix:

$$F_i = \mathbb{E}_{P_S}\left[\left(\frac{\partial \log p(y|\mathbf{x};\theta)}{\partial \theta_i}\right)^2\right]$$

**Step 2:** Parameters with high $F_i$ are critical for source performance. Modifying them during fine-tuning causes catastrophic forgetting.

**Step 3 — Elastic Weight Consolidation (EWC):** Add a regularization term:

$$\mathcal{L}_{\text{EWC}} = \mathcal{L}_T(\theta) + \frac{\lambda}{2}\sum_i F_i (\theta_i - \theta_i^*)^2$$

where $\theta^*$ are the pre-trained weights.

**Step 4:** This penalizes changes to important parameters (high $F_i$) while allowing changes to unimportant ones. The quadratic penalty approximates the KL divergence between the source posterior and the current parameters:

$$D_{\text{KL}}(p(y|\mathbf{x};\theta) \| p(y|\mathbf{x};\theta^*)) \approx \frac{1}{2}\sum_i F_i(\theta_i - \theta_i^*)^2$$

### 1.5 Layer-wise Transfer Analysis

**Step 1:** Define the task similarity at layer $l$ as:

$$\text{sim}_l = 1 - \frac{\|W_l^{\text{source}} - W_l^{\text{target}}\|_F}{\|W_l^{\text{source}}\|_F}$$

**Step 2:** Empirical finding (Yosinski et al., 2014): Early layers have $\text{sim}_l \approx 1$ (universal features), while later layers have $\text{sim}_l < 1$ (task-specific features).

**Step 3:** Optimal strategy: freeze layers 1 through $l^*$ (universal features), fine-tune layers $l^*+1$ through $L$ (task-specific features). The optimal split point $l^*$ depends on task similarity.

---

## 2. Algorithm / Method

### 2.1 Pseudocode: Transfer Learning Pipeline

```
Algorithm: Transfer_Learning
Input: Pre-trained model (φ_θ, g_ψ), target data D_T, strategy ∈ {freeze, finetune}
Output: Adapted model

1. LOAD pre-trained weights θ*, ψ*
2. REPLACE classifier head: g_ψ' for target classes
3. If strategy == "freeze":
     Fix θ = θ* (no gradient)
     Train only ψ' on D_T
4. If strategy == "finetune":
     Set learning rate: lr_backbone = lr_base / 10
     Set learning rate: lr_head = lr_base
     Train (θ, ψ') on D_T with differential learning rates
5. EVALUATE on target test set
6. Return adapted model
```

### 2.2 Complexity Analysis

- **Feature extraction:** $O(n_T \cdot d_{\text{head}})$ per epoch (only head is updated)
- **Fine-tuning:** $O(n_T \cdot d_{\text{total}})$ per epoch
- **Space:** $O(d_{\text{total}})$ for parameters in both cases

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

- **State:** $s_t = \phi_\theta(\mathbf{I}_t)$ — pre-trained features as state encoding
- **Action:** RL policy output
- **Reward:** Task reward in the target domain
- **Transition:** Target environment dynamics

### 3.2 Why RL?

Transfer learning enables RL agents to leverage pre-trained visual features (e.g., ImageNet-trained CNNs) for state encoding, dramatically reducing the number of environment interactions needed. The Ben-David bound quantifies when transfer will help vs. hurt.

---

## 4. Dataset

- **Name:** ImageNet (source) → CIFAR-10/custom (target)
- **Size:** 1.2M → 50K (or fewer for few-shot)
- **Auto-download:**

```python
from torchvision import models, datasets
model = models.resnet18(pretrained=True)
target_data = datasets.CIFAR10('./data', download=True)
```

---

## 5. Key Equations Summary

| Equation | Meaning |
|----------|---------|
| $\epsilon_T(h) \leq \epsilon_S(h) + \frac{1}{2}d_{\mathcal{H}\Delta\mathcal{H}} + \lambda^*$ | Ben-David bound |
| $d_{\mathcal{H}} = 2\sup_h\|P_S(h=1) - P_T(h=1)\|$ | Domain divergence |
| $\mathcal{L}_{\text{EWC}} = \mathcal{L}_T + \frac{\lambda}{2}\sum_i F_i(\theta_i - \theta_i^*)^2$ | Elastic weight consolidation |
| $F_i = \mathbb{E}[(\partial\log p/\partial\theta_i)^2]$ | Fisher information |

---

## 6. References

- Ben-David, S. et al. "A Theory of Learning from Different Domains," *Machine Learning*, 79:151–175, 2010.
- Yosinski, J. et al. "How Transferable Are Features in Deep Neural Networks?," *NeurIPS*, 2014.
- Kirkpatrick, J. et al. "Overcoming Catastrophic Forgetting in Neural Networks," *PNAS*, 114(13):3521–3526, 2017.
- Pan, S. J. & Yang, Q. "A Survey on Transfer Learning," *IEEE TKDE*, 22(10):1345–1359, 2010.
