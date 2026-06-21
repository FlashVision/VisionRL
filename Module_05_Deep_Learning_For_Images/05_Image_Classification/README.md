![Module Logo](../logo.png)

# Image Classification

## Overview

Image classification assigns a categorical label to an input image using a trained neural network. This document derives the softmax function from the exponential family of distributions, derives the cross-entropy loss from maximum likelihood estimation, analyzes top-$k$ accuracy, and develops calibration theory with a complete Platt scaling derivation.

## Prerequisites

- Neural networks and CNNs (Modules 05.1–05.2)
- Probability theory (MLE, Module 03.1)
- Information theory (KL divergence)

---

## 1. Mathematical Foundations

### 1.1 Core Definition — Classification Problem

Given an input $\mathbf{x} \in \mathbb{R}^d$ and $K$ classes, learn a function $f: \mathbb{R}^d \to \Delta^{K-1}$ mapping inputs to probability distributions over classes, where $\Delta^{K-1} = \{\mathbf{p} \in \mathbb{R}^K : p_k \geq 0, \sum_k p_k = 1\}$ is the $(K-1)$-simplex.

### 1.2 Softmax Derivation from the Exponential Family

**Step 1:** The categorical distribution belongs to the exponential family. For a random variable $Y \in \{1, \ldots, K\}$:

$$P(Y = k) = \frac{\exp(\eta_k)}{\sum_{j=1}^K \exp(\eta_j)}$$

where $\eta_k$ are the natural (canonical) parameters.

**Step 2:** In a neural network, the logits $\mathbf{z} = f_\theta(\mathbf{x}) \in \mathbb{R}^K$ are the natural parameters. The softmax function converts logits to probabilities:

$$\text{softmax}(z_k) = \frac{\exp(z_k)}{\sum_{j=1}^K \exp(z_j)}$$

**Step 3 — Derivation via maximum entropy.** Among all distributions on $\{1, \ldots, K\}$ with prescribed expected sufficient statistics $\mathbb{E}[\phi(Y)] = \boldsymbol{\mu}$, the maximum entropy distribution has the form:

$$p^*(k) = \frac{1}{Z(\boldsymbol{\eta})} \exp(\boldsymbol{\eta}^T \phi(k))$$

For the categorical case with $\phi(k) = \mathbf{e}_k$ (one-hot encoding), this gives:

$$p^*(k) = \frac{\exp(\eta_k)}{\sum_j \exp(\eta_j)} = \text{softmax}(\eta_k)$$

**Step 4 — Properties of softmax:**
- Invariant to constant shift: $\text{softmax}(\mathbf{z} + c\mathbf{1}) = \text{softmax}(\mathbf{z})$
- Monotone: if $z_i > z_j$, then $\text{softmax}(z_i) > \text{softmax}(z_j)$
- Jacobian: $\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$

**Result:**

$$\boxed{P(Y = k \mid \mathbf{x}; \theta) = \text{softmax}(f_\theta(\mathbf{x}))_k = \frac{\exp(z_k)}{\sum_j \exp(z_j)}}$$
$\blacksquare$

### 1.3 Cross-Entropy Loss Derivation from MLE

**Step 1:** Given $N$ training examples $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$ with $y_i \in \{1, \ldots, K\}$, the likelihood under the softmax model is:

$$L(\theta) = \prod_{i=1}^{N} P(Y = y_i \mid \mathbf{x}_i; \theta) = \prod_{i=1}^N \frac{\exp(z_{y_i}^{(i)})}{\sum_j \exp(z_j^{(i)})}$$

**Step 2:** The log-likelihood:

$$\ell(\theta) = \sum_{i=1}^N \left[z_{y_i}^{(i)} - \log\sum_j \exp(z_j^{(i)})\right]$$

**Step 3:** The negative log-likelihood (the loss to minimize) is:

$$\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^N \log P(Y = y_i \mid \mathbf{x}_i; \theta)$$

**Step 4:** Using the one-hot encoding $\mathbf{q}_i = \mathbf{e}_{y_i}$:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \sum_{k=1}^K q_{ik} \log p_{ik} = \frac{1}{N}\sum_{i=1}^N H(\mathbf{q}_i, \mathbf{p}_i)$$

where $H(\mathbf{q}, \mathbf{p}) = -\sum_k q_k \log p_k$ is the cross-entropy.

**Step 5:** Note that cross-entropy decomposes as:

$$H(\mathbf{q}, \mathbf{p}) = H(\mathbf{q}) + D_{\text{KL}}(\mathbf{q} \| \mathbf{p})$$

Since $H(\mathbf{q}) = 0$ for one-hot distributions, minimizing cross-entropy is equivalent to minimizing KL divergence from the empirical distribution.

**Step 6 — Gradient computation:**

$$\frac{\partial \mathcal{L}}{\partial z_k} = p_k - q_k = P(Y = k \mid \mathbf{x}; \theta) - \mathbb{1}[y = k]$$

This elegant gradient is the prediction error: the gradient pushes predicted probabilities toward the true one-hot label.

**Result:**

$$\boxed{\mathcal{L}_{\text{CE}} = -\sum_k q_k \log p_k, \quad \nabla_{z_k}\mathcal{L} = p_k - q_k}$$
$\blacksquare$

### 1.4 Top-$k$ Accuracy Analysis

**Definition:** Top-$k$ accuracy is the fraction of samples where the true label is among the $k$ highest-scoring predictions:

$$\text{Acc@}k = \frac{1}{N}\sum_{i=1}^N \mathbb{1}[y_i \in \text{top-}k(\mathbf{p}_i)]$$

**Relationship to softmax probabilities:**

$$P(y \in \text{top-}k) = \sum_{\pi} P(\text{rank}(z_y) \leq k \mid \text{permutation } \pi)$$

For a well-calibrated model: $\text{Acc@}k \approx \sum_{j=1}^k p_{(j)}$ where $p_{(j)}$ is the $j$-th largest predicted probability.

**Bound:** $\text{Acc@}k \geq 1 - (1 - \text{Acc@}1)^k$ if errors are independent (lower bound, pessimistic assumption).

### 1.5 Calibration Theory — Platt Scaling Derivation

**Problem:** A classifier is well-calibrated if $P(Y = k \mid P_\theta(Y = k) = p) = p$. Modern neural networks are often overconfident.

**Expected Calibration Error (ECE):**

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N}|\text{acc}(B_m) - \text{conf}(B_m)|$$

where $B_m$ are bins of predicted confidence.

**Platt Scaling — Derivation:**

**Step 1:** Model the calibrated probability as a logistic function of the uncalibrated logit $z$:

$$P_{\text{cal}}(Y = 1 \mid z) = \sigma(az + b) = \frac{1}{1 + \exp(-(az + b))}$$

**Step 2:** The parameters $(a, b)$ are learned by minimizing the NLL on a validation set:

$$\mathcal{L}(a, b) = -\sum_{i=1}^{N_{\text{val}}} [y_i \log \sigma(az_i + b) + (1-y_i)\log(1 - \sigma(az_i + b))]$$

**Step 3:** Take derivatives:

$$\frac{\partial \mathcal{L}}{\partial a} = -\sum_i (y_i - \sigma(az_i + b)) z_i = 0$$

$$\frac{\partial \mathcal{L}}{\partial b} = -\sum_i (y_i - \sigma(az_i + b)) = 0$$

**Step 4:** These are solved by gradient descent or Newton's method (the Hessian is readily computable).

**Step 5 — Multi-class extension (Temperature Scaling):** A special case of Platt scaling where $a = 1/T$ and $b = 0$:

$$P_{\text{cal}}(Y = k \mid \mathbf{x}) = \text{softmax}(\mathbf{z}/T)$$

The temperature $T > 1$ softens the distribution (reducing overconfidence).

**Derivation of optimal $T$:** Minimize NLL:

$$T^* = \arg\min_T -\sum_i \log \text{softmax}(\mathbf{z}_i / T)_{y_i}$$

Taking the derivative and setting to zero:

$$\sum_i \left[\frac{z_{y_i}^{(i)}}{T^2} - \frac{\sum_k z_k^{(i)} e^{z_k^{(i)}/T}}{T^2 \sum_k e^{z_k^{(i)}/T}}\right] = 0$$

This is a one-dimensional optimization (line search suffices).
$\blacksquare$

---

## 2. Algorithm / Method

### 2.1 Pseudocode: Image Classification Pipeline

```
Algorithm: Image_Classification
Input: Training data D, CNN architecture, epochs E
Output: Trained classifier

1. BUILD model: CNN backbone + softmax classifier
2. For epoch = 1 to E:
     For each batch (X, Y):
       Z = model(X)            # logits
       P = softmax(Z)          # probabilities
       L = cross_entropy(P, Y) # loss
       ∇ = backward(L)         # backpropagation
       θ = θ - lr · ∇          # gradient descent
3. CALIBRATE: Find T* on validation set
4. Return model with temperature T*
```

### 2.2 Complexity Analysis

- **Forward pass:** $O(d_{\text{model}})$ per sample
- **Softmax:** $O(K)$ per sample
- **Cross-entropy:** $O(K)$ per sample
- **Calibration (temp scaling):** $O(N_{\text{val}} \cdot K)$ one-time cost

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

- **State:** $s_t = \mathbf{I}_t$ — input image
- **Action:** $a_t = \hat{y}$ — predicted class label
- **Reward:** $r_t = \mathbb{1}[\hat{y} = y]$ — correct classification reward
- **Transition:** Next image drawn from the dataset (i.i.d.)

### 3.2 Why RL?

Softmax policies in RL are directly derived from the classification softmax: $\pi(a|s) = \text{softmax}(Q(s, \cdot) / T)$. Cross-entropy is the policy gradient loss for classification. Calibration ensures the policy's confidence reflects true success probability — critical for risk-sensitive RL.

---

## 4. Dataset

- **Name:** CIFAR-10, ImageNet
- **Size:** 50K/10K (CIFAR-10), 1.2M/50K (ImageNet)
- **Auto-download:**

```python
from torchvision import datasets, transforms
cifar = datasets.CIFAR10('./data', train=True, download=True,
                         transform=transforms.ToTensor())
```

---

## 5. Key Equations Summary

| Equation | Meaning |
|----------|---------|
| $p_k = \exp(z_k)/\sum_j\exp(z_j)$ | Softmax function |
| $\mathcal{L} = -\sum_k q_k\log p_k$ | Cross-entropy loss |
| $\nabla_{z_k}\mathcal{L} = p_k - q_k$ | Gradient of CE loss |
| $\text{ECE} = \sum_m(|B_m|/N)\|\text{acc}_m - \text{conf}_m\|$ | Expected calibration error |
| $P_{\text{cal}} = \text{softmax}(\mathbf{z}/T)$ | Temperature scaling |

---

## 6. References

- Platt, J. C. "Probabilistic Outputs for Support Vector Machines," *Advances in Large Margin Classifiers*, MIT Press, 2000.
- Guo, C. et al. "On Calibration of Modern Neural Networks," *ICML*, 2017.
- Bishop, C. M. *Pattern Recognition and Machine Learning*, Springer, 2006, Ch. 4.
- Goodfellow, I., Bengio, Y., & Courville, A. *Deep Learning*, MIT Press, 2016, Ch. 6.
