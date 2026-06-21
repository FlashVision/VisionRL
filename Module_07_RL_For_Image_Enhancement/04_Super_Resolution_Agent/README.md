![Module Logo](../logo.png)

# Super-Resolution via Reinforcement Learning

## Overview

This module formulates single-image super-resolution (SISR) as a reinforcement learning problem where an agent selects upsampling strategies and enhancement operations to reconstruct high-resolution images from low-resolution inputs. The mathematical treatment covers the degradation model, MAP estimation theory, frequency-domain limitations, and perceptual loss functions that guide the RL agent's reward signal.

## Prerequisites

- Linear algebra (SVD, pseudoinverse, ill-conditioned systems)
- Signal processing (Nyquist theorem, sampling theory, aliasing)
- Optimization (regularization, proximal operators)
- Bayesian inference (MAP estimation, prior models)
- Deep learning (CNN architectures, loss functions)

---

## 1. Mathematical Foundations

### 1.1 Observation (Degradation) Model

The relationship between a high-resolution (HR) image $x \in \mathbb{R}^{rH \times rW}$ and its low-resolution (LR) observation $y \in \mathbb{R}^{H \times W}$ is:

$$y = DHx + n$$

where:
- $D \in \mathbb{R}^{HW \times rH \cdot rW}$ is the downsampling operator (decimation by factor $r$)
- $H \in \mathbb{R}^{rH \cdot rW \times rH \cdot rW}$ is a blur kernel (anti-aliasing filter)
- $n \sim \mathcal{N}(0, \sigma^2 I)$ is additive noise
- $r$ is the upscaling factor

### 1.2 MAP Estimation Derivation

**Step 1:** The inverse problem is to recover $x$ from $y$. By Bayes' theorem:

$$p(x|y) = \frac{p(y|x) \cdot p(x)}{p(y)}$$

**Step 2:** The MAP estimate maximizes the posterior:

$$\hat{x}_{MAP} = \arg\max_x \, p(x|y) = \arg\max_x \, [\log p(y|x) + \log p(x)]$$

**Step 3:** Under Gaussian noise, the log-likelihood is:

$$\log p(y|x) = -\frac{1}{2\sigma^2}\|y - DHx\|_2^2 + \text{const}$$

**Step 4:** The log-prior encodes our assumptions about natural images. For a generic regularizer $R(x)$:

$$\log p(x) \propto -\lambda R(x)$$

**Step 5:** The MAP estimate becomes:

$$\hat{x}_{MAP} = \arg\min_x \underbrace{\frac{1}{2}\|y - DHx\|_2^2}_{\text{data fidelity}} + \underbrace{\lambda R(x)}_{\text{regularization}}$$

**Step 6:** Common regularizers and their priors:

| Regularizer $R(x)$ | Implicit Prior | Properties |
|---|---|---|
| $\|\nabla x\|_2^2$ (Tikhonov) | Gaussian | Smooth solutions |
| $\|\nabla x\|_1$ (Total Variation) | Laplacian gradients | Edge-preserving |
| $\|\Psi x\|_1$ (Sparse transform) | Sparse in $\Psi$ basis | Wavelet/DCT sparsity |

**Step 7:** For Tikhonov regularization, the closed-form solution is:

$$\hat{x} = (H^TD^TDH + \lambda L^TL)^{-1} H^TD^T y$$

where $L$ is the discrete gradient operator.

**Step 8:** For TV regularization, solve via ADMM with auxiliary variable $z = \nabla x$:

$$\mathcal{L}(x, z, u) = \frac{1}{2}\|y - DHx\|_2^2 + \lambda\|z\|_1 + \frac{\rho}{2}\|\nabla x - z + u\|_2^2$$

### 1.3 Proof: Upsampling Cannot Create New Frequencies (Nyquist)

**Theorem:** Let $x(t)$ be bandlimited to $[-B, B]$ and sampled at rate $f_s = 2B/r$ (sub-Nyquist by factor $r$). No algorithm can recover frequency content in $[B/r, B]$ from the samples alone without additional prior information.

**Proof:**

**Step 1:** By the sampling theorem, $x_s(t) = x(t) \cdot \text{III}_{T_s}(t)$ where $\text{III}_{T_s}$ is the Dirac comb with period $T_s = 1/f_s$.

**Step 2:** In frequency domain:

$$X_s(f) = f_s \sum_{k=-\infty}^{\infty} X(f - kf_s)$$

**Step 3:** When $f_s < 2B$, the spectral copies overlap (aliasing):

$$X_s(f) = f_s[X(f) + X(f - f_s) + X(f + f_s) + \cdots]$$

**Step 4:** For $f \in [f_s/2, B]$, the spectral component $X(f)$ is summed with $X(f - f_s)$, creating an irrecoverable mixture:

$$X_s(f) = X(f) + X(f - f_s) \quad \text{for } f \in [f_s/2, f_s]$$

**Step 5:** This is a single equation with two unknowns. Without additional constraints (priors), the system is underdetermined. Therefore, no reconstruction algorithm can uniquely recover the original frequencies above $f_s/2$. $\blacksquare$

**Corollary:** Super-resolution algorithms that appear to create high-frequency detail are actually hallucinating content based on learned priors (training data statistics), not recovering true signal information.

### 1.4 SRCNN Loss and Perceptual Loss Derivation

**Pixel-wise loss (SRCNN):**

$$\mathcal{L}_{pixel} = \frac{1}{N}\sum_{i=1}^N \|F(y_i; \theta) - x_i\|_2^2$$

where $F(\cdot; \theta)$ is the SR network.

**Perceptual loss derivation:**

**Step 1:** Let $\phi_l(I)$ be the feature map at layer $l$ of a pre-trained VGG network, with dimensions $C_l \times H_l \times W_l$.

**Step 2:** The perceptual loss measures feature-space distance:

$$\mathcal{L}_{perceptual} = \frac{1}{C_l H_l W_l} \|\phi_l(F(y;\theta)) - \phi_l(x)\|_2^2$$

**Step 3:** The gradient with respect to the SR network output $\hat{x} = F(y;\theta)$:

$$\frac{\partial \mathcal{L}_{perceptual}}{\partial \hat{x}} = \frac{1}{C_l H_l W_l} \cdot J_{\phi_l}^T(\hat{x}) \cdot [\phi_l(\hat{x}) - \phi_l(x)]$$

where $J_{\phi_l}$ is the Jacobian of the feature extractor.

**Step 4:** The total loss combines pixel, perceptual, and adversarial terms:

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{pixel} + \lambda_2 \mathcal{L}_{perceptual} + \lambda_3 \mathcal{L}_{adversarial}$$

---

## 2. Algorithm / Method

### 2.1 Pseudocode

```
Algorithm: RL Super-Resolution Agent
─────────────────────────────────────────
Input: Low-resolution image y, scale factor r
Output: Super-resolved image x̂

State: s = (current_SR_image, residual_map, frequency_analysis, step_count)
Actions: {apply_SRCNN, apply_EDSR, apply_RealESRGAN, 
          sharpen(σ), denoise(h), adjust_contrast(α), stop}

Initialize actor-critic networks π_θ, V_φ

for episode = 1 to M do
    x̂₀ ← bicubic_upsample(y, r)    // Initial upsampling
    s₀ ← extract_features(x̂₀, y)
    for t = 0 to T_max do
        aₜ ~ π_θ(·|sₜ)
        x̂_{t+1} ← apply_action(x̂ₜ, aₜ)
        rₜ ← λ₁·ΔPSNR + λ₂·ΔSSIM + λ₃·Δperceptual
        s_{t+1} ← extract_features(x̂_{t+1}, y)
        if aₜ == stop: break
    end for
    Update θ, φ via PPO with GAE
end for
```

### 2.2 Complexity Analysis

- **Bicubic upsampling:** $O(r^2 HW \cdot k^2)$ with kernel size $k$
- **CNN-based SR:** $O(r^2 HW \cdot C^2 \cdot K^2 \cdot L)$ for $L$ layers with $C$ channels and $K \times K$ kernels
- **Perceptual loss:** $O(r^2 HW \cdot C_{VGG}^2)$ for VGG forward pass
- **Agent decision:** $O(|\theta|)$
- **Memory:** $O(r^2 HW \cdot C_{max})$ for intermediate feature maps

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

- **State:** Current SR image features, frequency spectrum residual, local sharpness map
- **Action:** Selection from a toolbox of SR operators + post-processing filters with continuous parameters
- **Reward:** Multi-metric improvement: $r_t = \alpha\Delta\text{PSNR} + \beta\Delta\text{SSIM} + \gamma\Delta\text{LPIPS}$
- **Transition:** Deterministic application of selected operation
- **Terminal:** Quality plateau detected or maximum steps reached

### 3.2 Why RL?

1. **Ill-posed problem:** Super-resolution has infinitely many solutions; RL can learn to prefer perceptually pleasing ones
2. **Multi-objective trade-off:** Pixel accuracy vs. perceptual quality vs. computation cost — RL naturally handles Pareto-optimal strategies
3. **Adaptive strategy:** Different image regions (texture, edges, flat) benefit from different SR approaches
4. **Sequential enhancement:** Post-processing after initial SR (sharpening, denoising) benefits from learned ordering

---

## 4. Dataset

| Dataset | Size | Scale Factors | Description |
|---------|------|--------------|-------------|
| DIV2K | 1,000 images | ×2, ×3, ×4 | High-quality diverse images |
| Set5/Set14 | 5/14 images | ×2, ×3, ×4 | Classic test benchmarks |
| Urban100 | 100 images | ×2, ×4 | Urban scenes with structure |
| Manga109 | 109 images | ×2, ×4 | Line art and text |

---

## 5. Key Equations Summary

| Equation | Description |
|----------|-------------|
| $y = DHx + n$ | Image degradation model |
| $\hat{x} = \arg\min \|y-DHx\|^2 + \lambda R(x)$ | MAP super-resolution |
| $\mathcal{L}_{perc} = \frac{1}{C_lH_lW_l}\|\phi_l(\hat{x}) - \phi_l(x)\|^2$ | Perceptual loss |
| $f_s \geq 2B$ (Nyquist) | Sampling theorem requirement |
| $\hat{x}_{Tik} = (H^TD^TDH + \lambda L^TL)^{-1}H^TD^Ty$ | Tikhonov solution |

---

## 6. References

1. Dong, C., Loy, C. C., He, K., & Tang, X. (2014). Learning a deep convolutional network for image super-resolution. *ECCV*.
2. Ledig, C., et al. (2017). Photo-realistic single image super-resolution using a generative adversarial network. *CVPR*.
3. Johnson, J., Alahi, A., & Fei-Fei, L. (2016). Perceptual losses for real-time style transfer and super-resolution. *ECCV*.
4. Shannon, C. E. (1949). Communication in the presence of noise. *Proceedings of the IRE*, 37(1), 10-21.
5. Wang, X., et al. (2018). ESRGAN: Enhanced super-resolution generative adversarial networks. *ECCV Workshops*.
