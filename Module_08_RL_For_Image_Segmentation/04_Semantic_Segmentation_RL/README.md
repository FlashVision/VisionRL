![Module Logo](../logo.png)

# Semantic Segmentation with Reinforcement Learning

## Overview

This module applies reinforcement learning to semantic segmentation, where an RL agent learns to select optimal hyperparameters and post-processing strategies for dense pixel labeling. The mathematical framework covers fully convolutional network theory, dilated convolution receptive fields, and Conditional Random Field (CRF) inference for spatial consistency, all formulated within an RL framework for automated hyperparameter optimization.

## Prerequisites

- Convolutional neural networks (convolution, pooling, upsampling)
- Probabilistic graphical models (MRFs, CRFs, inference)
- Variational inference (mean-field approximation, KL divergence)
- Information theory (mutual information, entropy)
- Reinforcement learning (bandit problems, Bayesian optimization connection)

---

## 1. Mathematical Foundations

### 1.1 Fully Convolutional Network: Upsampling Mathematics

**Transposed Convolution (Deconvolution):** The upsampling operation maps feature maps from low resolution to high resolution.

**Step 1:** Standard convolution as matrix multiplication. For input $\mathbf{x} \in \mathbb{R}^{n^2}$ (flattened) and kernel $\mathbf{k}$, convolution is:

$$\mathbf{y} = \mathbf{C}\mathbf{x}$$

where $\mathbf{C} \in \mathbb{R}^{m^2 \times n^2}$ is the sparse convolution matrix (Toeplitz structure).

**Step 2:** The transposed convolution applies $\mathbf{C}^T$:

$$\hat{\mathbf{x}} = \mathbf{C}^T\mathbf{y}$$

This maps from $\mathbb{R}^{m^2}$ back to $\mathbb{R}^{n^2}$, effectively upsampling.

**Step 3:** Output size formula for transposed convolution with stride $s$, padding $p$, kernel size $k$:

$$o = (i - 1) \cdot s - 2p + k$$

**Step 4:** Bilinear interpolation initialization. For upsampling factor $f$, the bilinear kernel:

$$w(x, y) = (1 - |x|/f)(1 - |y|/f) \quad \text{for } |x|, |y| \leq f$$

**Step 5:** Skip connections fuse multi-scale features:

$$\mathbf{F}_{fused} = \mathbf{F}_{upsampled}^{deep} + \mathbf{W} \cdot \mathbf{F}_{shallow}$$

### 1.2 Dilated (Atrous) Convolution: Receptive Field Proof

**Definition:** A dilated convolution with rate $r$ and kernel $\mathbf{k}$ of size $K$:

$$(f *_r k)(x) = \sum_{i=-\lfloor K/2\rfloor}^{\lfloor K/2\rfloor} f(x + r \cdot i) \cdot k(i)$$

**Theorem:** A stack of $L$ dilated convolutions with rates $r_1, r_2, \ldots, r_L$ and kernel size $K$ has effective receptive field:

$$\text{RF} = 1 + \sum_{l=1}^L r_l \cdot (K - 1)$$

**Proof by induction:**

**Base case ($L=1$):** A single dilated convolution with rate $r_1$ and kernel $K$ sees:
- Positions: $\{0, \pm r_1, \pm 2r_1, \ldots, \pm \lfloor K/2\rfloor \cdot r_1\}$
- Span: $r_1(K-1) + 1$  ✓

**Inductive step:** Assume the first $L-1$ layers have RF $= 1 + \sum_{l=1}^{L-1}r_l(K-1)$. Layer $L$ with rate $r_L$ expands each position by $r_L(K-1)$:

$$\text{RF}_L = \text{RF}_{L-1} + r_L(K-1) = 1 + \sum_{l=1}^L r_l(K-1) \quad \blacksquare$$

**Example (DeepLab):** Rates $r = [1, 6, 12, 18]$ with $K = 3$:

$$\text{RF} = 1 + 2(1 + 6 + 12 + 18) = 75 \text{ pixels}$$

### 1.3 CRF as Post-Processing: Mean-Field Inference

**Definition:** The dense CRF energy for labeling $\mathbf{x}$ given image $\mathbf{I}$:

$$E(\mathbf{x}|\mathbf{I}) = \sum_i \psi_u(x_i) + \sum_{i<j}\psi_p(x_i, x_j)$$

**Unary potential** (from CNN softmax):

$$\psi_u(x_i = l) = -\log P(x_i = l | \mathbf{I})$$

**Pairwise potential** (bilateral + spatial):

$$\psi_p(x_i, x_j) = \mu(x_i, x_j)\left[w_1 \exp\left(-\frac{|p_i-p_j|^2}{2\theta_\alpha^2} - \frac{|I_i-I_j|^2}{2\theta_\beta^2}\right) + w_2\exp\left(-\frac{|p_i-p_j|^2}{2\theta_\gamma^2}\right)\right]$$

where $\mu(x_i, x_j) = \mathbb{1}[x_i \neq x_j]$ (Potts model).

**Mean-field inference derivation:**

**Step 1:** Approximate the true posterior $P(\mathbf{x}|\mathbf{I})$ with a fully factored distribution:

$$Q(\mathbf{x}) = \prod_i Q_i(x_i)$$

**Step 2:** Minimize $\text{KL}(Q \| P)$:

$$\text{KL}(Q\|P) = \sum_\mathbf{x} Q(\mathbf{x})\log\frac{Q(\mathbf{x})}{P(\mathbf{x}|\mathbf{I})}$$

**Step 3:** Taking the functional derivative with respect to $Q_i$ and setting to zero:

$$\log Q_i(x_i) = -\psi_u(x_i) - \sum_{j\neq i}\sum_{x_j} Q_j(x_j)\psi_p(x_i, x_j) + \text{const}$$

**Step 4:** Exponentiate and normalize:

$$Q_i(x_i = l) = \frac{1}{Z_i}\exp\left(-\psi_u(x_i = l) - \sum_{j\neq i}\sum_{l'}\psi_p(l, l')Q_j(x_j = l')\right)$$

**Step 5:** Iterate until convergence. The message passing step uses Gaussian filtering for efficient computation:

$$\tilde{Q}_i^{(m)}(l) = \sum_j k^{(m)}(f_i, f_j) Q_j(l)$$

which is a bilateral filter applied to the $Q$ distributions — computable in $O(N)$ using permutohedral lattice.

**Step 6:** Update rule:

$$\hat{Q}_i(l) \leftarrow \exp(-\psi_u(x_i = l))\exp\left(-\sum_m w_m \tilde{Q}_i^{(m)}(l)\right)$$

Normalize: $Q_i(l) \leftarrow \hat{Q}_i(l) / \sum_{l'}\hat{Q}_i(l')$.

### 1.4 Proof: Mean-Field Minimizes KL Divergence in Factored Family

**Theorem:** The fixed point of the mean-field update equations corresponds to a local minimum of $\text{KL}(Q\|P)$ within the family of fully-factored distributions.

**Proof:**

**Step 1:** The KL divergence in terms of the energy:

$$\text{KL}(Q\|P) = \sum_i \sum_l Q_i(l)\log Q_i(l) + E_Q[E(\mathbf{x})] + \log Z$$

**Step 2:** The first term is $-H(Q)$ (negative entropy). The update equations arise from setting the gradient with respect to each $Q_i$ to zero while maintaining normalization (Lagrange multiplier).

**Step 3:** Since KL divergence is convex in $Q_i$ for fixed $Q_{j\neq i}$, each coordinate update decreases (or maintains) the objective. By monotone convergence in a bounded-below objective, the algorithm converges to a fixed point. $\blacksquare$

---

## 2. Algorithm / Method

### 2.1 Pseudocode

```
Algorithm: RL for Semantic Segmentation Hyperparameter Selection
───────────────────────────────────────────────────────────────────
Input: Image I, CNN backbone
Output: Optimized segmentation and hyperparameters

State: s = (image_complexity, class_distribution, 
            boundary_density, current_mIoU)
Action: a = (dilation_rates[], CRF_θα, CRF_θβ, CRF_θγ,
             CRF_w1, CRF_w2, num_CRF_iterations)

Initialize policy π_θ (hyperparameter selector)

for episode = 1 to M do
    Sample validation image
    s₀ ← compute_image_stats(I)
    for t = 0 to T_max do
        aₜ ~ π_θ(·|sₜ)
        segmentation ← FCN_forward(I, dilation_rates_t)
        refined ← CRF_inference(segmentation, I, CRF_params_t)
        rₜ ← mIoU(refined, ground_truth)
        s_{t+1} ← update_state(refined, I)
    end for
    Update π_θ via REINFORCE with baseline
end for
```

### 2.2 Complexity Analysis

- **FCN forward pass:** $O(N \cdot C^2 \cdot K^2 \cdot L)$
- **Mean-field CRF (per iteration):** $O(N \cdot K_{CRF})$ with permutohedral lattice
- **Total CRF:** $O(N \cdot K_{CRF} \cdot T_{iter})$ for $T_{iter}$ iterations
- **Hyperparameter search (brute force):** $O(|\mathcal{A}|^d)$ exponential
- **RL agent decision:** $O(|\theta_{RL}|)$ — constant per image

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

- **State:** Image-level features (complexity, edge density, class balance) + current segmentation quality metrics
- **Action:** Hyperparameter configuration (CRF parameters, dilation rates, loss weights, learning rate)
- **Reward:** Mean IoU on validation set with selected hyperparameters
- **Transition:** Stochastic (different images yield different mIoU for same hyperparameters)

### 3.2 Why RL?

1. **Hyperparameter landscape:** CRF parameters interact non-linearly; RL explores this space efficiently
2. **Image-dependent optimization:** Optimal CRF parameters vary by image content (e.g., fine texture vs. large regions)
3. **Computational budget:** RL agent learns to allocate CRF iterations based on expected improvement
4. **End-to-end optimization:** Directly optimizes mIoU rather than surrogate losses

---

## 4. Dataset

| Dataset | Classes | Size | Resolution | Description |
|---------|---------|------|-----------|-------------|
| Cityscapes | 19 | 5,000 | 2048×1024 | Urban scenes |
| ADE20K | 150 | 25,210 | Variable | Scene parsing |
| PASCAL Context | 59 | 10,103 | Variable | Full scene labels |
| COCO Stuff | 171 | 164K | Variable | Things + stuff |

---

## 5. Key Equations Summary

| Equation | Description |
|----------|-------------|
| $\text{RF} = 1 + \sum_l r_l(K-1)$ | Dilated conv receptive field |
| $E(\mathbf{x}) = \sum_i\psi_u(x_i) + \sum_{i<j}\psi_p(x_i,x_j)$ | CRF energy |
| $Q_i(l) \propto \exp(-\psi_u - \sum_m w_m\tilde{Q}_i^{(m)})$ | Mean-field update |
| $o = (i-1)s - 2p + k$ | Transposed conv output size |
| $\text{KL}(Q\|P) = -H(Q) + E_Q[E] + \log Z$ | Variational objective |

---

## 6. References

1. Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. *CVPR*.
2. Chen, L.-C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. (2018). DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs. *IEEE TPAMI*, 40(4), 834-848.
3. Krähenbühl, P., & Koltun, V. (2011). Efficient inference in fully connected CRFs with Gaussian edge potentials. *NeurIPS*.
4. Yu, F., & Koltun, V. (2016). Multi-scale context aggregation by dilated convolutions. *ICLR*.
