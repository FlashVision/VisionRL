![Module Logo](../logo.png)

# Noise and Denoising

## Overview

Image noise arises from sensor physics, transmission errors, and quantization. This document derives maximum likelihood estimators for the major noise models (Gaussian, Poisson, salt-and-pepper), provides a complete derivation of the Wiener filter from the minimum mean squared error (MMSE) principle, proves its optimality, and establishes the theoretical foundations of collaborative filtering methods like BM3D.

## Prerequisites

- Probability and statistics (ML estimation, expectation)
- Fourier analysis and power spectral density
- Optimization (calculus of variations)
- Module 02.1 (convolution and filtering)

---

## 1. Mathematical Foundations

### 1.1 Core Definition — Noise Models

A noisy image is modeled as:

$$g(x, y) = f(x, y) + n(x, y) \quad \text{(additive noise)}$$

or

$$g(x, y) = f(x, y) \cdot n(x, y) \quad \text{(multiplicative noise)}$$

where $f$ is the clean image and $n$ is the noise process.

### 1.2 Maximum Likelihood Estimation for Noise Parameters

#### Gaussian Noise Model

**Model:** $g_i = f_i + n_i$ where $n_i \sim \mathcal{N}(0, \sigma^2)$ i.i.d.

**Step 1:** The likelihood of observing $N$ pixels given clean values $\{f_i\}$ and noise variance $\sigma^2$:

$$L(\sigma^2) = \prod_{i=1}^{N} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(g_i - f_i)^2}{2\sigma^2}\right)$$

**Step 2:** The log-likelihood:

$$\ell(\sigma^2) = -\frac{N}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{N}(g_i - f_i)^2$$

**Step 3:** Differentiate with respect to $\sigma^2$ and set to zero:

$$\frac{\partial \ell}{\partial \sigma^2} = -\frac{N}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^{N}(g_i - f_i)^2 = 0$$

**Step 4:** Solve:

$$\hat{\sigma}^2_{\text{ML}} = \frac{1}{N}\sum_{i=1}^{N}(g_i - f_i)^2$$

#### Poisson Noise Model

**Model:** For photon-counting sensors, $g_i \sim \text{Poisson}(f_i)$.

**Step 1:** The PMF is $P(g_i | f_i) = \frac{f_i^{g_i} e^{-f_i}}{g_i!}$.

**Step 2:** Log-likelihood: $\ell(\{f_i\}) = \sum_i [g_i \log f_i - f_i - \log(g_i!)]$.

**Step 3:** ML estimate: $\frac{\partial \ell}{\partial f_i} = \frac{g_i}{f_i} - 1 = 0 \implies \hat{f}_i = g_i$.

**Key property:** $\text{Var}(g_i) = f_i$ — the noise variance equals the signal intensity. This is the defining characteristic of shot noise in low-light imaging.

#### Salt-and-Pepper Noise Model

**Model:** Each pixel is independently corrupted:

$$g_i = \begin{cases} 0 & \text{with probability } p/2 \\ L-1 & \text{with probability } p/2 \\ f_i & \text{with probability } 1-p \end{cases}$$

**ML estimate of $p$:** Count pixels at extreme values: $\hat{p} = \frac{\#\{i : g_i = 0 \text{ or } g_i = L-1\}}{N}$ (assuming $f$ rarely takes extreme values).

### 1.3 Full Derivation of the Wiener Filter

**Problem:** Given noisy observation $G(\omega) = F(\omega) + N(\omega)$ in the frequency domain, find the filter $W(\omega)$ that minimizes the mean squared error:

$$\text{MSE} = \mathbb{E}\left[|F(\omega) - W(\omega)G(\omega)|^2\right]$$

**Step 1:** Expand the MSE:

$$\text{MSE} = \mathbb{E}\left[|F - WG|^2\right] = \mathbb{E}\left[|F - W(F + N)|^2\right]$$

$$= \mathbb{E}\left[|(1-W)F - WN|^2\right]$$

**Step 2:** Expand the squared magnitude (assuming $F$ and $N$ are uncorrelated):

$$= |1-W|^2 \mathbb{E}[|F|^2] + |W|^2 \mathbb{E}[|N|^2]$$

$$= |1-W|^2 S_{ff}(\omega) + |W|^2 S_{nn}(\omega)$$

where $S_{ff}(\omega) = \mathbb{E}[|F(\omega)|^2]$ is the power spectral density (PSD) of the signal and $S_{nn}(\omega) = \mathbb{E}[|N(\omega)|^2]$ is the PSD of the noise.

**Step 3:** Minimize with respect to $W(\omega)$. Treat $W$ as a real variable (for real-valued filters) and differentiate:

$$\frac{\partial \text{MSE}}{\partial W} = -2(1-W)S_{ff} + 2W S_{nn} = 0$$

**Step 4:** Solve for $W$:

$$-S_{ff} + WS_{ff} + WS_{nn} = 0$$

$$W(S_{ff} + S_{nn}) = S_{ff}$$

$$W_{\text{Wiener}}(\omega) = \frac{S_{ff}(\omega)}{S_{ff}(\omega) + S_{nn}(\omega)}$$

**Step 5:** Equivalently, in terms of the signal-to-noise ratio $\text{SNR}(\omega) = S_{ff}(\omega)/S_{nn}(\omega)$:

$$W_{\text{Wiener}}(\omega) = \frac{\text{SNR}(\omega)}{\text{SNR}(\omega) + 1}$$

**Result:**

$$\boxed{W_{\text{Wiener}}(\omega) = \frac{S_{ff}(\omega)}{S_{ff}(\omega) + S_{nn}(\omega)} = \frac{1}{1 + 1/\text{SNR}(\omega)}}$$
$\blacksquare$

### 1.4 Proof of Wiener Filter Optimality

**Theorem:** The Wiener filter is the MMSE-optimal linear filter.

**Proof:**

**Step 1:** We showed in Section 1.3 that $W_{\text{Wiener}}$ is a critical point of the MSE functional. The second derivative:

$$\frac{\partial^2 \text{MSE}}{\partial W^2} = 2(S_{ff} + S_{nn}) > 0$$

confirms it is a minimum (strict convexity).

**Step 2:** Compute the minimum MSE achieved:

$$\text{MSE}_{\min} = |1 - W^*|^2 S_{ff} + |W^*|^2 S_{nn}$$

Substituting $W^* = S_{ff}/(S_{ff} + S_{nn})$:

$$= \left(\frac{S_{nn}}{S_{ff} + S_{nn}}\right)^2 S_{ff} + \left(\frac{S_{ff}}{S_{ff} + S_{nn}}\right)^2 S_{nn} = \frac{S_{ff} S_{nn}}{S_{ff} + S_{nn}}$$

**Step 3:** Behavior analysis:
- When $\text{SNR} \gg 1$: $W \approx 1$ (pass the signal through unchanged).
- When $\text{SNR} \ll 1$: $W \approx 0$ (suppress noise by attenuating the signal at that frequency).

The Wiener filter is an adaptive frequency-domain filter that passes frequencies where the signal dominates and attenuates frequencies where noise dominates.
$\blacksquare$

### 1.5 Generalized Wiener Filter (with Blur)

For a degraded observation $G = HF + N$ where $H(\omega)$ is the blur transfer function:

$$W_{\text{gen}}(\omega) = \frac{H^*(\omega)}{|H(\omega)|^2 + S_{nn}(\omega)/S_{ff}(\omega)}$$

**Derivation:** Repeat the MMSE derivation with $G = HF + N$:

$$\text{MSE} = \mathbb{E}[|F - WG|^2] = |1 - WH|^2 S_{ff} + |W|^2 S_{nn}$$

$$\frac{\partial}{\partial W} = -2H^*(1 - WH)S_{ff} + 2WS_{nn} = 0$$

$$W = \frac{H^* S_{ff}}{|H|^2 S_{ff} + S_{nn}}$$

### 1.6 BM3D Collaborative Filtering Theory

BM3D (Block-Matching and 3D Filtering) exploits non-local self-similarity:

**Step 1 — Block matching:** For each reference patch $P_{\text{ref}}$, find similar patches using the distance $d(P_i, P_{\text{ref}}) = \|P_i - P_{\text{ref}}\|_2^2 / |P|$.

**Step 2 — 3D grouping:** Stack matched patches into a 3D array $\mathbf{G} \in \mathbb{R}^{p \times p \times K}$.

**Step 3 — Collaborative filtering:** Apply a 3D transform (e.g., 2D DCT + 1D Hadamard along the stacking dimension), threshold/Wiener filter the coefficients, and invert:

$$\hat{\mathbf{G}} = \mathcal{T}_{3D}^{-1}\left[W_{\text{Wiener}} \cdot \mathcal{T}_{3D}(\mathbf{G})\right]$$

**Theoretical justification:** Similar patches can be viewed as multiple noisy observations of the same underlying signal. Averaging $K$ independent noisy copies reduces noise by $\sqrt{K}$. The 3D transform further exploits spectral sparsity within the group, achieving near-optimal denoising rates.

---

## 2. Algorithm / Method

### 2.1 Pseudocode: Wiener Denoising

```
Algorithm: Wiener_Filter_Denoising
Input: Noisy image g, estimated noise PSD S_nn, estimated signal PSD S_ff
Output: Denoised image f_hat

1. G = FFT2(g)
2. W = S_ff / (S_ff + S_nn)
3. F_hat = W · G
4. f_hat = real(IFFT2(F_hat))
5. Return f_hat
```

### 2.2 Complexity Analysis

- **Wiener filter:** $O(MN\log(MN))$ — dominated by FFT
- **BM3D:** $O(MN \cdot K \cdot p^2)$ where $K$ = matched patches, $p$ = patch size
- **Space:** $O(MN)$ for Wiener; $O(MNKp^2)$ for BM3D patch groups

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

- **State:** $s_t = \mathbf{I}_t$ — current (partially denoised) image
- **Action:** $a_t \in \{\text{Wiener}(\sigma), \text{median}(k), \text{bilateral}(\sigma_s, \sigma_r), \text{NLM}(h)\}$
- **Reward:** $r_t = \text{PSNR}(\mathbf{I}_{t+1}, \mathbf{I}_{\text{clean}}) - \text{PSNR}(\mathbf{I}_t, \mathbf{I}_{\text{clean}})$
- **Transition:** Apply selected denoising filter

### 3.2 Why RL?

Different noise types (Gaussian, Poisson, salt-and-pepper) require different optimal filters. An RL agent can learn to identify the noise type from image statistics and select the appropriate denoising method and parameters adaptively, achieving better results than any single fixed denoiser.

---

## 4. Dataset

- **Name:** CIFAR-10 or scikit-image built-ins with synthetic noise
- **Size:** 60,000 images (CIFAR-10) or standard test images
- **Auto-download:**

```python
from skimage import data, util
camera = data.camera()
noisy = util.random_noise(camera, mode='gaussian', var=0.01)
```

---

## 5. Key Equations Summary

| Equation | Meaning |
|----------|---------|
| $g = f + n$, $n \sim \mathcal{N}(0, \sigma^2)$ | Additive Gaussian noise model |
| $\hat{\sigma}^2 = \frac{1}{N}\sum(g_i - f_i)^2$ | ML noise variance estimate |
| $W(\omega) = \frac{S_{ff}}{S_{ff} + S_{nn}}$ | Wiener filter (MMSE optimal) |
| $\text{MSE}_{\min} = \frac{S_{ff}S_{nn}}{S_{ff}+S_{nn}}$ | Minimum achievable MSE |
| $\text{Var}(g_i) = f_i$ | Poisson noise variance property |

---

## 6. References

- Wiener, N. *Extrapolation, Interpolation, and Smoothing of Stationary Time Series*, MIT Press, 1949.
- Dabov, K., Foi, A., Katkovnik, V., & Egiazarian, K. "Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering," *IEEE Trans. Image Processing*, 16(8):2080–2095, 2007.
- Buades, A., Coll, B., & Morel, J.-M. "A Non-Local Algorithm for Image Denoising," *CVPR*, 2005.
- Gonzalez, R. C. & Woods, R. E. *Digital Image Processing*, 4th ed., Pearson, 2018, Ch. 5.
