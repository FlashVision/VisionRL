![Module Logo](../logo.png)

# Neural Network Basics

## Overview

Neural networks are universal function approximators that form the backbone of deep reinforcement learning. This document states the universal approximation theorem, provides the complete backpropagation derivation with full chain rule through multiple layers, analyzes gradient flow and the vanishing/exploding gradient problem, and connects these foundations to RL function approximation.

## Prerequisites

- Multivariable calculus (chain rule, Jacobians)
- Linear algebra (matrix multiplication)
- Basic optimization (gradient descent)

---

## 1. Mathematical Foundations

### 1.1 Core Definition — Feedforward Neural Network

A feedforward neural network with $L$ layers computes:

$$f(\mathbf{x}; \boldsymbol{\theta}) = \sigma_L(\mathbf{W}_L \sigma_{L-1}(\mathbf{W}_{L-1} \cdots \sigma_1(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) \cdots + \mathbf{b}_{L-1}) + \mathbf{b}_L)$$

where $\boldsymbol{\theta} = \{(\mathbf{W}_l, \mathbf{b}_l)\}_{l=1}^L$ are the parameters and $\sigma_l$ are activation functions.

Layer-by-layer:

$$\mathbf{z}_l = \mathbf{W}_l \mathbf{a}_{l-1} + \mathbf{b}_l, \quad \mathbf{a}_l = \sigma_l(\mathbf{z}_l), \quad \mathbf{a}_0 = \mathbf{x}$$

### 1.2 Universal Approximation Theorem

**Theorem (Cybenko, 1989; Hornik, 1991):** Let $\sigma$ be a non-constant, bounded, continuous activation function. For any continuous function $f: [0,1]^d \to \mathbb{R}$ and any $\varepsilon > 0$, there exists a single-hidden-layer network with $N$ neurons:

$$g(\mathbf{x}) = \sum_{i=1}^{N} v_i \sigma(\mathbf{w}_i^T \mathbf{x} + b_i)$$

such that $\sup_{\mathbf{x} \in [0,1]^d} |f(\mathbf{x}) - g(\mathbf{x})| < \varepsilon$.

**Significance:** This guarantees that neural networks can represent any continuous function — the question is whether gradient descent can find the right parameters.

**Limitation:** The theorem is existential (non-constructive). It doesn't bound $N$ or guarantee learnability. In practice, deeper networks are exponentially more parameter-efficient than shallow ones for many function classes.

### 1.3 Full Backpropagation Derivation

**Goal:** Compute $\frac{\partial \mathcal{L}}{\partial \mathbf{W}_l}$ and $\frac{\partial \mathcal{L}}{\partial \mathbf{b}_l}$ for all layers $l$.

**Step 1 — Forward pass.** For $l = 1, \ldots, L$:

$$\mathbf{z}_l = \mathbf{W}_l \mathbf{a}_{l-1} + \mathbf{b}_l, \quad \mathbf{a}_l = \sigma_l(\mathbf{z}_l)$$

**Step 2 — Loss.** For a single sample with target $\mathbf{y}$:

$$\mathcal{L} = \ell(\mathbf{a}_L, \mathbf{y})$$

For MSE: $\mathcal{L} = \frac{1}{2}\|\mathbf{a}_L - \mathbf{y}\|^2$.

**Step 3 — Output layer error.** Define $\boldsymbol{\delta}_L = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_L}$:

$$\boldsymbol{\delta}_L = \frac{\partial \mathcal{L}}{\partial \mathbf{a}_L} \odot \sigma_L'(\mathbf{z}_L)$$

For MSE loss: $\boldsymbol{\delta}_L = (\mathbf{a}_L - \mathbf{y}) \odot \sigma_L'(\mathbf{z}_L)$.

**Step 4 — Backpropagate the error.** For $l = L-1, L-2, \ldots, 1$:

$$\boldsymbol{\delta}_l = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_l} = \left(\mathbf{W}_{l+1}^T \boldsymbol{\delta}_{l+1}\right) \odot \sigma_l'(\mathbf{z}_l)$$

**Derivation:** By the chain rule:

$$\frac{\partial \mathcal{L}}{\partial z_l^{(i)}} = \sum_j \frac{\partial \mathcal{L}}{\partial z_{l+1}^{(j)}} \cdot \frac{\partial z_{l+1}^{(j)}}{\partial a_l^{(i)}} \cdot \frac{\partial a_l^{(i)}}{\partial z_l^{(i)}}$$

Since $z_{l+1}^{(j)} = \sum_i W_{l+1}^{(j,i)} a_l^{(i)} + b_{l+1}^{(j)}$:

$$\frac{\partial z_{l+1}^{(j)}}{\partial a_l^{(i)}} = W_{l+1}^{(j,i)}$$

And $\frac{\partial a_l^{(i)}}{\partial z_l^{(i)}} = \sigma_l'(z_l^{(i)})$.

Therefore: $\delta_l^{(i)} = \sigma_l'(z_l^{(i)}) \sum_j W_{l+1}^{(j,i)} \delta_{l+1}^{(j)}$, which in matrix form is $\boldsymbol{\delta}_l = (\mathbf{W}_{l+1}^T\boldsymbol{\delta}_{l+1}) \odot \sigma_l'(\mathbf{z}_l)$.

**Step 5 — Parameter gradients.** Using $\mathbf{z}_l = \mathbf{W}_l\mathbf{a}_{l-1} + \mathbf{b}_l$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_l} = \boldsymbol{\delta}_l \mathbf{a}_{l-1}^T$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}_l} = \boldsymbol{\delta}_l$$

**Derivation:** $\frac{\partial \mathcal{L}}{\partial W_l^{(i,j)}} = \frac{\partial \mathcal{L}}{\partial z_l^{(i)}} \cdot \frac{\partial z_l^{(i)}}{\partial W_l^{(i,j)}} = \delta_l^{(i)} \cdot a_{l-1}^{(j)}$.

**Result (Complete Backpropagation):**

$$\boxed{\boldsymbol{\delta}_L = \nabla_{\mathbf{a}_L}\mathcal{L} \odot \sigma'_L(\mathbf{z}_L), \quad \boldsymbol{\delta}_l = (\mathbf{W}_{l+1}^T\boldsymbol{\delta}_{l+1}) \odot \sigma'_l(\mathbf{z}_l), \quad \nabla_{\mathbf{W}_l}\mathcal{L} = \boldsymbol{\delta}_l\mathbf{a}_{l-1}^T}$$
$\blacksquare$

### 1.4 Gradient Flow Analysis

**Step 1:** The gradient at layer $l$ is:

$$\boldsymbol{\delta}_l = \left(\prod_{k=l+1}^{L}\text{diag}(\sigma'_k(\mathbf{z}_k))\mathbf{W}_k^T\right)\boldsymbol{\delta}_L$$

**Step 2:** The gradient magnitude scales as:

$$\|\boldsymbol{\delta}_l\| \approx \|\boldsymbol{\delta}_L\| \prod_{k=l+1}^{L}\|\sigma'_k\| \cdot \|\mathbf{W}_k\|$$

**Step 3 — Vanishing gradients.** If $\|\sigma'_k\| \cdot \|\mathbf{W}_k\| < 1$ (e.g., sigmoid $\sigma' \leq 0.25$), gradients decay exponentially: $\|\boldsymbol{\delta}_l\| \sim c^{L-l} \to 0$.

**Step 4 — Exploding gradients.** If $\|\sigma'_k\| \cdot \|\mathbf{W}_k\| > 1$, gradients grow exponentially: $\|\boldsymbol{\delta}_l\| \sim c^{L-l} \to \infty$.

**Solutions:**
- **ReLU activation:** $\sigma'(z) \in \{0, 1\}$, avoiding the squashing effect.
- **Residual connections:** $\mathbf{a}_l = \sigma(\mathbf{z}_l) + \mathbf{a}_{l-1}$, providing gradient shortcuts.
- **Batch normalization:** Normalizes pre-activations to maintain gradient scale.
- **Careful initialization:** Xavier ($\text{Var}(W) = 1/n_{\text{in}}$) or He ($\text{Var}(W) = 2/n_{\text{in}}$).

---

## 2. Algorithm / Method

### 2.1 Pseudocode: Backpropagation

```
Algorithm: Backpropagation
Input: Network with L layers, input x, target y, loss ℓ
Output: Gradients ∂ℓ/∂W_l, ∂ℓ/∂b_l for all l

1. FORWARD PASS: a_0 = x
   For l = 1 to L:
     z_l = W_l · a_{l-1} + b_l
     a_l = σ_l(z_l)
2. COMPUTE LOSS: ℓ = loss(a_L, y)
3. BACKWARD PASS: δ_L = ∇_{a_L}ℓ ⊙ σ'_L(z_L)
   For l = L-1 down to 1:
     δ_l = (W_{l+1}ᵀ · δ_{l+1}) ⊙ σ'_l(z_l)
4. GRADIENTS:
   For l = 1 to L:
     ∂ℓ/∂W_l = δ_l · a_{l-1}ᵀ
     ∂ℓ/∂b_l = δ_l
5. Return gradients
```

### 2.2 Complexity Analysis

- **Forward pass:** $O(\sum_l n_l \cdot n_{l-1})$ (matrix-vector products)
- **Backward pass:** Same as forward pass (symmetric computation)
- **Space:** $O(\sum_l n_l)$ for storing activations and errors

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

- **State:** $s_t$ — input to the neural network (e.g., image)
- **Action:** Network output — value estimate $V(s;\theta)$ or policy $\pi(a|s;\theta)$
- **Reward:** RL loss gradient used in place of supervised loss
- **Transition:** Environment dynamics (not learned by the network itself)

### 3.2 Why RL?

Neural networks serve as function approximators in deep RL: $Q(s,a;\theta) \approx Q^*(s,a)$ and $\pi_\theta(a|s)$. Backpropagation computes the gradients needed for DQN (Module 06.1), policy gradient (Module 06.2), and actor-critic (Module 06.3) algorithms.

---

## 4. Dataset

- **Name:** MNIST, CIFAR-10
- **Size:** 60K/10K (MNIST), 50K/10K (CIFAR-10)
- **Auto-download:**

```python
from torchvision import datasets
mnist = datasets.MNIST('./data', download=True)
cifar = datasets.CIFAR10('./data', download=True)
```

---

## 5. Key Equations Summary

| Equation | Meaning |
|----------|---------|
| $\mathbf{a}_l = \sigma(\mathbf{W}_l\mathbf{a}_{l-1}+\mathbf{b}_l)$ | Forward pass |
| $\boldsymbol{\delta}_l = (\mathbf{W}_{l+1}^T\boldsymbol{\delta}_{l+1})\odot\sigma'(\mathbf{z}_l)$ | Error backpropagation |
| $\nabla_{\mathbf{W}_l}\mathcal{L} = \boldsymbol{\delta}_l\mathbf{a}_{l-1}^T$ | Weight gradient |
| $\|\boldsymbol{\delta}_l\| \sim \prod_{k>l}\|\sigma'_k\|\cdot\|\mathbf{W}_k\|$ | Gradient flow |

---

## 6. References

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. "Learning Representations by Back-propagating Errors," *Nature*, 323:533–536, 1986.
- Cybenko, G. "Approximation by Superpositions of a Sigmoidal Function," *Math. Control, Signals, Systems*, 2(4):303–314, 1989.
- Glorot, X. & Bengio, Y. "Understanding the Difficulty of Training Deep Feedforward Neural Networks," *AISTATS*, 2010.
- He, K. et al. "Delving Deep into Rectifiers," *ICCV*, 2015.
