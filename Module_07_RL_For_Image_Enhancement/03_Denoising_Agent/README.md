![Module Logo](../logo.png)

# Image Denoising via Reinforcement Learning

## Overview

This module develops a reinforcement learning agent that performs sequential image denoising by selecting and applying optimal filtering operations at each step. The mathematical framework bridges classical denoising theory (Wiener filtering, Non-Local Means) with modern RL-based approaches, formulating noise removal as a multi-step decision process that maximizes image quality metrics.

## Prerequisites

- Signal processing (Fourier transform, power spectral density)
- Probability theory (conditional expectation, Bayesian estimation)
- Functional analysis (Hilbert spaces, projection operators)
- Optimization (convex optimization, variational methods)
- Reinforcement learning (Q-learning, policy gradients)

---

## 1. Mathematical Foundations

### 1.1 Noise Model

The observed noisy image follows the additive model:

$$y(x) = s(x) + n(x)$$

where $s(x)$ is the clean signal, $n(x) \sim \mathcal{N}(0, \sigma^2)$ is i.i.d. Gaussian noise, and $x \in \Omega \subset \mathbb{R}^2$ is the spatial domain.

### 1.2 Wiener Filter Derivation (Minimum MSE)

**Goal:** Find the linear filter $h$ that minimizes the mean squared error:

$$\hat{s} = h * y, \quad \text{minimize } E[\|s - \hat{s}\|^2]$$

**Step 1:** Work in the frequency domain. Let $H(\omega)$, $S(\omega)$, $Y(\omega)$, $N(\omega)$ denote Fourier transforms:

$$\hat{S}(\omega) = H(\omega) \cdot Y(\omega)$$

**Step 2:** Express the MSE in frequency domain (Parseval's theorem):

$$\text{MSE} = \frac{1}{(2\pi)^2}\int |S(\omega) - H(\omega)Y(\omega)|^2 \, d\omega$$

**Step 3:** Expand: Since $Y = S + N$:

$$\text{MSE} = \frac{1}{(2\pi)^2}\int |S(\omega) - H(\omega)(S(\omega) + N(\omega))|^2 \, d\omega$$

**Step 4:** Take expectation (assuming $S$ and $N$ are uncorrelated):

$$E[\text{MSE}] = \frac{1}{(2\pi)^2}\int \left[(1-H)^2 P_{ss}(\omega) + H^2 P_{nn}(\omega)\right] d\omega$$

where $P_{ss}(\omega) = E[|S(\omega)|^2]$ is the signal power spectral density.

**Step 5:** Differentiate with respect to $H(\omega)$ and set to zero:

$$\frac{\partial}{\partial H}\left[(1-H)^2 P_{ss} + H^2 P_{nn}\right] = -2(1-H)P_{ss} + 2H P_{nn} = 0$$

**Step 6:** Solve for $H$:

$$H_{Wiener}(\omega) = \frac{P_{ss}(\omega)}{P_{ss}(\omega) + P_{nn}(\omega)}$$

**Step 7:** Equivalently, in terms of SNR $= P_{ss}/P_{nn}$:

$$H_{Wiener}(\omega) = \frac{\text{SNR}(\omega)}{\text{SNR}(\omega) + 1} = 1 - \frac{1}{\text{SNR}(\omega) + 1}$$

**Step 8:** Verify optimality — the second derivative $\frac{\partial^2}{\partial H^2} = 2(P_{ss} + P_{nn}) > 0$, confirming this is a minimum. $\blacksquare$

### 1.3 Non-Local Means (NLM) Derivation

**Core Idea:** Estimate each pixel as a weighted average of all pixels in the image, where weights reflect patch similarity.

**Step 1:** Define the denoised estimate:

$$\hat{I}(x) = \sum_{y \in \Omega} w(x, y) \cdot I(y)$$

subject to the normalization constraint $\sum_y w(x, y) = 1$ and $w(x, y) \geq 0$.

**Step 2:** Define patch similarity. Let $P(x)$ denote the patch of size $(2d+1)^2$ centered at pixel $x$:

$$P(x) = \{I(x + \delta) : \|\delta\|_\infty \leq d\}$$

**Step 3:** The weight is defined via a Gaussian kernel on patch distances:

$$w(x, y) = \frac{1}{Z(x)} \exp\left(-\frac{\|P(x) - P(y)\|_{2,a}^2}{h^2}\right)$$

where $h$ is the filtering parameter, $Z(x) = \sum_y \exp(-\|P(x)-P(y)\|_{2,a}^2 / h^2)$ is the normalizing constant, and $\|\cdot\|_{2,a}$ is a Gaussian-weighted $L^2$ norm.

**Step 4:** Expand the weighted norm:

$$\|P(x) - P(y)\|_{2,a}^2 = \sum_{\delta} a(\delta) |I(x+\delta) - I(y+\delta)|^2$$

where $a(\delta) = \frac{1}{(2\pi\sigma_a^2)} e^{-|\delta|^2/(2\sigma_a^2)}$.

**Step 5:** Bias-variance analysis. The NLM estimate satisfies:

$$E[\hat{I}(x)] = \sum_y w_0(x,y) s(y) + O(\sigma^2/h^2)$$

$$\text{Var}[\hat{I}(x)] = \frac{\sigma^2}{\sum_y w(x,y)^2}$$

where $w_0$ are the oracle weights (from clean patches).

**Step 6:** Optimal $h$ balances bias and variance: $h^* \propto \sigma \sqrt{2d+1}$.

### 1.4 BM3D: Collaborative Filtering

**Step 1 (Grouping):** For reference patch $P_R$, find similar patches:

$$\mathcal{G}(P_R) = \{P_i : \|P_R - P_i\|^2 / |P| \leq \tau_{match}\}$$

**Step 2 (3D Transform):** Stack matched patches into a 3D array and apply a 3D transform (2D spatial + 1D across patches):

$$\Theta_{3D} = T_{1D} \circ T_{2D}$$

**Step 3 (Collaborative filtering):** Apply hard thresholding or Wiener filtering in the 3D transform domain:

$$\hat{\Theta}_{3D}(\omega) = \begin{cases} \Theta_{3D}(\omega) & \text{if } |\Theta_{3D}(\omega)| > \lambda\sigma \\ 0 & \text{otherwise} \end{cases}$$

**Step 4 (Aggregation):** Inverse transform and aggregate overlapping estimates with weights:

$$\hat{I}(x) = \frac{\sum_R w_R \cdot \hat{P}_R(x)}{\sum_R w_R \cdot \mathbb{1}[x \in P_R]}$$

where $w_R = 1/\|\hat{\Theta}_{3D,R}\|_0$ (inverse of number of non-zero coefficients).

### 1.5 Proof: Wiener Filter is Optimal among Linear Estimators

**Theorem:** Among all linear estimators $\hat{s} = h * y$, the Wiener filter achieves the minimum MSE.

**Proof:** The MSE functional is:

$$J[H] = \int \left[(1-H(\omega))^2 P_{ss}(\omega) + H(\omega)^2 P_{nn}(\omega)\right] d\omega$$

This is a quadratic functional in $H(\omega)$ for each $\omega$. Since the integrand is a sum of squared terms weighted by positive spectral densities, $J[H]$ is strictly convex. The unique minimizer satisfies the first-order condition derived above. By strict convexity, no other linear filter can achieve lower MSE. $\blacksquare$

---

## 2. Algorithm / Method

### 2.1 Pseudocode

```
Algorithm: Sequential Denoising RL Agent
─────────────────────────────────────────
Input: Noisy image y, noise level estimate σ
Output: Denoised image ŝ

State: s = (current_image, noise_map, iteration_count, quality_history)
Actions: {NLM(h₁), NLM(h₂), BM3D(σ₁), Bilateral(σ_s, σ_r), 
          Gaussian(σ_g), Median(k), Stop}

for episode = 1 to M do
    s₀ ← (y, σ_map, 0, [])
    for t = 0 to T_max do
        aₜ ← π_θ(sₜ)                    // Select filter + params
        I_{t+1} ← apply_filter(Iₜ, aₜ)
        rₜ ← PSNR(I_{t+1}, s) - PSNR(Iₜ, s)  // PSNR improvement
        if aₜ == Stop or rₜ < ε: break
        sₜ₊₁ ← update_state(I_{t+1})
    end for
    Update θ via policy gradient
end for
```

### 2.2 Complexity Analysis

- **NLM:** $O(N \cdot |S| \cdot |P|)$ where $|S|$ is search window, $|P|$ is patch size
- **BM3D:** $O(N \cdot |S| \cdot |P| \cdot \log|P|)$ (FFT-based)
- **Wiener filter:** $O(N \log N)$ (via FFT)
- **Agent decision:** $O(|\theta|)$ per step
- **Total:** $O(T \cdot N \cdot |S| \cdot |P|)$ for $T$ steps

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

- **State:** Current image features + estimated residual noise map + quality trajectory
- **Action:** Choice of denoising filter and its parameters from a discrete-continuous hybrid space
- **Reward:** $r_t = \text{PSNR}(I_{t+1}, I_{clean}) - \text{PSNR}(I_t, I_{clean}) - c$ where $c > 0$ is a per-step cost encouraging efficiency
- **Terminal condition:** Agent selects "Stop" or PSNR improvement falls below threshold

### 3.2 Why RL?

1. **Heterogeneous noise:** Real images have spatially varying noise; the agent can apply different strategies locally
2. **Sequential refinement:** Over-denoising causes artifacts; RL learns when to stop
3. **Filter selection:** No single filter is optimal for all noise types; the agent learns a mixture strategy
4. **Diminishing returns:** Each denoising step has decreasing benefit; RL balances quality vs. computation

---

## 4. Dataset

| Dataset | Size | Noise Type | Description |
|---------|------|-----------|-------------|
| BSD68 | 68 images | Synthetic Gaussian | Standard benchmark |
| Set12 | 12 images | σ ∈ {15, 25, 50} | Classic test set |
| SIDD | 30,000 pairs | Real camera noise | Smartphone images |
| DND | 50 images | Real noise | Diverse scenes |

---

## 5. Key Equations Summary

| Equation | Description |
|----------|-------------|
| $H_W(\omega) = \frac{P_{ss}}{P_{ss} + P_{nn}}$ | Wiener filter (frequency domain) |
| $\hat{I}(x) = \sum_y w(x,y)I(y)$ | Non-Local Means estimator |
| $w(x,y) = \frac{1}{Z}\exp(-\frac{\|P(x)-P(y)\|^2}{h^2})$ | NLM weight function |
| $r_t = \Delta\text{PSNR}_t - c$ | RL step reward with cost |
| $h^* \propto \sigma\sqrt{2d+1}$ | Optimal NLM bandwidth |

---

## 6. References

1. Buades, A., Coll, B., & Morel, J.-M. (2005). A non-local algorithm for image denoising. *CVPR*.
2. Dabov, K., Foi, A., Katkovnik, V., & Egiazarian, K. (2007). Image denoising by sparse 3-D transform-domain collaborative filtering. *IEEE TIP*, 16(8), 2080-2095.
3. Wiener, N. (1949). *Extrapolation, Interpolation, and Smoothing of Stationary Time Series*. MIT Press.
4. Yu, K., Dong, C., Lin, L., & Loy, C. C. (2019). Crafting a toolchain for image restoration by deep reinforcement learning. *CVPR*.
