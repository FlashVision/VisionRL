![Module Logo](../logo.png)

# Brightness and Contrast Enhancement via Reinforcement Learning

## Overview

This module formulates image brightness and contrast enhancement as a sequential decision-making problem solved by a reinforcement learning agent. The agent learns to apply pixel-level transformations that maximize perceptual quality metrics (PSNR, SSIM) through iterative adjustments, treating each enhancement step as an action in a Markov Decision Process.

## Prerequisites

- Linear algebra (matrix operations, norms)
- Probability and statistics (expectation, variance, covariance)
- Calculus (partial derivatives, optimization)
- Basic RL concepts (MDP, policy, reward)
- Signal processing fundamentals (convolution, filtering)

---

## 1. Mathematical Foundations

### 1.1 Image Quality Metrics: Core Definitions

Let $I \in \mathbb{R}^{H \times W \times C}$ denote an image with height $H$, width $W$, and $C$ channels. For grayscale analysis, we consider single-channel images $I \in \mathbb{R}^{H \times W}$.

**Mean Squared Error (MSE):**

$$\text{MSE}(x, y) = \frac{1}{N} \sum_{i=1}^{N} (x_i - y_i)^2$$

where $x$ is the reference image, $y$ is the distorted image, and $N = H \times W$ is the total number of pixels.

### 1.2 PSNR Derivation from MSE

**Definition:** Peak Signal-to-Noise Ratio quantifies the ratio between the maximum possible signal power and the power of corrupting noise.

**Step 1:** Define the peak signal value. For an 8-bit image, $\text{MAX}_I = 2^8 - 1 = 255$.

**Step 2:** Express the signal-to-noise ratio in linear scale:

$$\text{SNR}_{linear} = \frac{\text{MAX}_I^2}{\text{MSE}(x, y)}$$

**Step 3:** Convert to decibel scale using logarithmic transformation:

$$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right)$$

**Step 4:** Expand using logarithm properties:

$$\text{PSNR} = 10 \cdot \log_{10}(\text{MAX}_I^2) - 10 \cdot \log_{10}(\text{MSE})$$

$$= 20 \cdot \log_{10}(\text{MAX}_I) - 10 \cdot \log_{10}(\text{MSE})$$

**Step 5:** Substitute the MSE expression:

$$\text{PSNR} = 20 \cdot \log_{10}(255) - 10 \cdot \log_{10}\left(\frac{1}{N}\sum_{i=1}^N (x_i - y_i)^2\right)$$

**Step 6:** Note that $20\log_{10}(255) \approx 48.13$ dB, so for a perfect reconstruction (MSE = 0), PSNR $\to \infty$.

**Step 7:** Establish the inverse relationship — halving the MSE increases PSNR by $10\log_{10}(2) \approx 3.01$ dB.

### 1.3 SSIM Full Derivation

The Structural Similarity Index decomposes image comparison into three independent components: luminance, contrast, and structure.

**Step 1: Luminance comparison.** Given pixel sets $x = \{x_i\}_{i=1}^N$ and $y = \{y_i\}_{i=1}^N$:

$$\mu_x = \frac{1}{N}\sum_{i=1}^N x_i, \quad \mu_y = \frac{1}{N}\sum_{i=1}^N y_i$$

The luminance comparison function:

$$l(x, y) = \frac{2\mu_x \mu_y + C_1}{\mu_x^2 + \mu_y^2 + C_1}$$

where $C_1 = (K_1 L)^2$ is a stabilization constant with $L$ being the dynamic range and $K_1 \ll 1$.

**Step 2: Contrast comparison.** Compute standard deviations:

$$\sigma_x = \left(\frac{1}{N-1}\sum_{i=1}^N (x_i - \mu_x)^2\right)^{1/2}$$

The contrast comparison function:

$$c(x, y) = \frac{2\sigma_x \sigma_y + C_2}{\sigma_x^2 + \sigma_y^2 + C_2}$$

where $C_2 = (K_2 L)^2$.

**Step 3: Structure comparison.** Compute the cross-correlation:

$$\sigma_{xy} = \frac{1}{N-1}\sum_{i=1}^N (x_i - \mu_x)(y_i - \mu_y)$$

The structure comparison function:

$$s(x, y) = \frac{\sigma_{xy} + C_3}{\sigma_x \sigma_y + C_3}$$

where typically $C_3 = C_2 / 2$.

**Step 4: Combine the three components:**

$$\text{SSIM}(x, y) = [l(x,y)]^\alpha \cdot [c(x,y)]^\beta \cdot [s(x,y)]^\gamma$$

With default parameters $\alpha = \beta = \gamma = 1$ and $C_3 = C_2/2$:

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

**Step 5: Proof that SSIM is bounded in $[-1, 1]$.**

### 1.4 Proof: SSIM $\in [-1, 1]$

**Theorem:** For any two signal patches $x, y$ with $C_1, C_2 > 0$, we have $-1 \leq \text{SSIM}(x, y) \leq 1$.

**Proof:**

*Upper bound:* Consider the luminance term. By the AM-GM inequality:

$$\mu_x^2 + \mu_y^2 \geq 2\mu_x\mu_y$$

Therefore:

$$l(x,y) = \frac{2\mu_x\mu_y + C_1}{\mu_x^2 + \mu_y^2 + C_1} \leq \frac{\mu_x^2 + \mu_y^2 + C_1}{\mu_x^2 + \mu_y^2 + C_1} = 1$$

For the structure term, by Cauchy-Schwarz: $|\sigma_{xy}| \leq \sigma_x \sigma_y$, thus:

$$|s(x,y)| = \left|\frac{\sigma_{xy} + C_3}{\sigma_x\sigma_y + C_3}\right| \leq \frac{|\sigma_{xy}| + C_3}{\sigma_x\sigma_y + C_3} \leq \frac{\sigma_x\sigma_y + C_3}{\sigma_x\sigma_y + C_3} = 1$$

Similarly for $c(x,y) \leq 1$ since $\sigma_x, \sigma_y \geq 0$.

Therefore $\text{SSIM}(x,y) \leq 1$.

*Lower bound:* The minimum of $\sigma_{xy}$ is $-\sigma_x\sigma_y$ (when $x$ and $y$ are perfectly anti-correlated). Then:

$$s(x,y) = \frac{-\sigma_x\sigma_y + C_3}{\sigma_x\sigma_y + C_3} \geq -1$$

Combined with $l(x,y) \geq 0$ and $c(x,y) \geq 0$ (since all variances are non-negative), we get $\text{SSIM} \geq -1$. $\blacksquare$

### 1.5 Histogram Specification as Optimal Transport

**Definition:** Let $p_s$ and $p_t$ be the source and target intensity distributions on $[0, L-1]$.

**Step 1:** Define cumulative distribution functions:

$$F_s(k) = \sum_{j=0}^{k} p_s(j), \quad F_t(k) = \sum_{j=0}^{k} p_t(j)$$

**Step 2:** The 1D Wasserstein distance (Earth Mover's Distance) is:

$$W_1(p_s, p_t) = \int_0^1 |F_s^{-1}(u) - F_t^{-1}(u)| \, du$$

**Step 3:** The optimal transport map $T^*$ that minimizes:

$$T^* = \arg\min_T \int |x - T(x)|^2 \, dp_s(x)$$

subject to $T_\# p_s = p_t$ (push-forward constraint).

**Step 4:** In 1D, the unique optimal map is:

$$T^*(x) = F_t^{-1}(F_s(x))$$

**Step 5:** This is exactly histogram specification — map each intensity through the source CDF and then the inverse target CDF.

**Step 6:** The $W_2^2$ cost under optimal transport equals:

$$W_2^2(p_s, p_t) = \int_0^1 |F_s^{-1}(u) - F_t^{-1}(u)|^2 \, du$$

---

## 2. Algorithm / Method

### 2.1 Pseudocode

```
Algorithm: RL Brightness-Contrast Agent
─────────────────────────────────────────
Input: Image I, target quality threshold τ
Output: Enhanced image I*

Initialize policy π_θ (actor network)
Initialize value function V_φ (critic network)

for episode = 1 to M do
    s₀ ← extract_features(I)
    for t = 0 to T_max do
        aₜ ~ π_θ(·|sₜ)          // Sample action (Δbrightness, Δcontrast)
        I_{t+1} ← apply_action(Iₜ, aₜ)
        rₜ ← α·ΔPSNR + β·ΔSSIM  // Compute reward
        s_{t+1} ← extract_features(I_{t+1})
        Store (sₜ, aₜ, rₜ, s_{t+1})
        if quality(I_{t+1}) ≥ τ: break
    end for
    Update θ, φ via PPO
end for
return I*
```

### 2.2 Complexity Analysis

- **State extraction:** $O(HWC)$ for computing image statistics
- **Action application:** $O(HW)$ per pixel transformation
- **Reward computation:** $O(HW)$ for PSNR; $O(HW \cdot k^2)$ for SSIM with window size $k$
- **Policy update:** $O(|\theta| \cdot B)$ where $B$ is batch size
- **Total per episode:** $O(T \cdot HW \cdot k^2)$ where $T$ is episode length

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

- **State space** $\mathcal{S}$: Image features — histogram, mean, variance, local contrast map
- **Action space** $\mathcal{A}$: Continuous adjustments $a = (\Delta\text{brightness}, \Delta\text{contrast}, \Delta\gamma) \in [-1, 1]^3$
- **Transition:** $P(s_{t+1}|s_t, a_t)$ is deterministic given the current image and action
- **Reward:** $r_t = \alpha \cdot \Delta\text{PSNR}_t + \beta \cdot \Delta\text{SSIM}_t + \lambda \cdot \mathbb{1}[\text{done}]$
- **Discount factor:** $\gamma \in [0.95, 0.99]$

### 3.2 Why RL?

1. **Sequential refinement:** Enhancement often requires multiple careful adjustments rather than a single transformation
2. **Non-differentiable metrics:** SSIM and perceptual quality are not easily optimized via gradient descent on the image directly
3. **Exploration-exploitation:** The agent can discover unconventional enhancement strategies
4. **Adaptability:** A trained policy generalizes across diverse image conditions without per-image optimization

---

## 4. Dataset

| Dataset | Size | Resolution | Description |
|---------|------|-----------|-------------|
| LOL (Low-Light) | 500 pairs | 400×600 | Low/normal light image pairs |
| MIT-Adobe FiveK | 5,000 images | Variable | Expert-retouched photos |
| DPED | 22,000 pairs | 1920×1080 | Phone-DSLR enhancement pairs |

---

## 5. Key Equations Summary

| Equation | Description |
|----------|-------------|
| $\text{PSNR} = 20\log_{10}\left(\frac{\text{MAX}_I}{\sqrt{\text{MSE}}}\right)$ | Peak Signal-to-Noise Ratio |
| $\text{SSIM} = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$ | Structural Similarity |
| $T^*(x) = F_t^{-1}(F_s(x))$ | Optimal histogram transport map |
| $r_t = \alpha\Delta\text{PSNR} + \beta\Delta\text{SSIM}$ | RL reward function |
| $W_2^2(p_s, p_t) = \int_0^1 \|F_s^{-1}(u) - F_t^{-1}(u)\|^2 du$ | Wasserstein-2 distance |

---

## 6. References

1. Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: from error visibility to structural similarity. *IEEE TIP*, 13(4), 600-612.
2. Villani, C. (2003). *Topics in Optimal Transport*. AMS.
3. Park, J., Lee, J. Y., Yoo, D., & Kweon, I. S. (2018). Distort-and-recover: Color enhancement using deep reinforcement learning. *CVPR*.
4. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.
