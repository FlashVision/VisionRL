![Module Logo](../logo.png)

# What Is An Image?

## Overview

An image is a sampled, quantized representation of a continuous two-dimensional irradiance function produced by physical light–surface–sensor interaction. This document derives the complete image formation pipeline from first principles, provides a rigorous proof of the Nyquist–Shannon sampling theorem, develops quantization theory with a full SQNR derivation, and connects these foundations to reinforcement learning state representations.

## Prerequisites

- Real analysis (limits, continuity, Riemann integration)
- Fourier analysis (Fourier transform, convolution)
- Basic probability and statistics
- Linear algebra fundamentals

---

## 1. Mathematical Foundations

### 1.1 Core Definition — The Image Formation Model

A monochromatic image is modeled as the product of illumination and reflectance:

$$f(x, y) = i(x, y) \cdot r(x, y)$$

where $f: \mathbb{R}^2 \to \mathbb{R}_{\geq 0}$ is the image intensity, $i: \mathbb{R}^2 \to \mathbb{R}_{>0}$ is the illumination function satisfying $0 < i(x,y) < \infty$, and $r: \mathbb{R}^2 \to (0,1)$ is the reflectance function.

For a point light source at position $\mathbf{p}_s$ with radiant intensity $I_0$, the illumination at a surface point $\mathbf{p}$ with surface normal $\hat{\mathbf{n}}$ is given by the rendering equation (Lambertian case):

$$i(\mathbf{p}) = \frac{I_0}{\|\mathbf{p} - \mathbf{p}_s\|^2} \max(\hat{\mathbf{n}} \cdot \hat{\mathbf{l}}, \; 0)$$

where $\hat{\mathbf{l}} = \frac{\mathbf{p}_s - \mathbf{p}}{\|\mathbf{p}_s - \mathbf{p}\|}$ is the unit vector toward the light source.

### 1.2 Full Derivation of the Nyquist–Shannon Sampling Theorem

**Theorem (Nyquist–Shannon):** Let $f(x)$ be a band-limited signal with $F(\omega) = 0$ for $|\omega| > \omega_{\max}$. Then $f(x)$ can be perfectly reconstructed from its samples $f(nT)$ taken at intervals $T < \frac{\pi}{\omega_{\max}}$ (equivalently, $f_s > 2f_{\max}$).

**Proof:**

**Step 1:** Define the sampled signal using the Dirac comb. The sampling operation multiplies $f(x)$ by a Dirac comb:

$$f_s(x) = f(x) \cdot \mathrm{III}_T(x) = f(x) \sum_{n=-\infty}^{\infty} \delta(x - nT)$$

$$f_s(x) = \sum_{n=-\infty}^{\infty} f(nT) \, \delta(x - nT)$$

**Step 2:** Compute the Fourier transform of the sampled signal. Multiplication in the spatial domain becomes convolution in the frequency domain. The Fourier transform of $\mathrm{III}_T(x)$ is:

$$\mathcal{F}\{\mathrm{III}_T(x)\} = \frac{1}{T} \mathrm{III}_{1/T}(\omega) = \frac{1}{T} \sum_{k=-\infty}^{\infty} \delta\!\left(\omega - \frac{k}{T}\right)$$

Therefore:

$$F_s(\omega) = F(\omega) * \frac{1}{T}\sum_{k=-\infty}^{\infty} \delta\!\left(\omega - \frac{k}{T}\right) = \frac{1}{T}\sum_{k=-\infty}^{\infty} F\!\left(\omega - \frac{k}{T}\right)$$

**Step 3:** Analyze the spectral replicas. The spectrum $F_s(\omega)$ consists of shifted copies of $F(\omega)$ centered at $\omega = k/T$ for each integer $k$. If $\frac{1}{T} > 2f_{\max}$ (i.e., $f_s > 2f_{\max}$), these copies do not overlap.

**Step 4:** Reconstruct by ideal low-pass filtering. Apply an ideal low-pass filter $H(\omega)$ with cutoff $f_c = \frac{1}{2T}$:

$$H(\omega) = \begin{cases} T & \text{if } |\omega| \leq \frac{1}{2T} \\ 0 & \text{otherwise} \end{cases}$$

Then $F(\omega) = F_s(\omega) \cdot H(\omega)$, which recovers only the $k=0$ replica: $F(\omega)$.

**Step 5:** Derive the reconstruction formula. The inverse Fourier transform of the ideal low-pass filter is the sinc function:

$$h(x) = \mathcal{F}^{-1}\{H(\omega)\} = \operatorname{sinc}\!\left(\frac{x}{T}\right) = \frac{\sin(\pi x / T)}{\pi x / T}$$

The reconstructed signal is the convolution:

$$f(x) = f_s(x) * h(x) = \sum_{n=-\infty}^{\infty} f(nT) \, \operatorname{sinc}\!\left(\frac{x - nT}{T}\right)$$

**Step 6:** Verify exactness at sample points. At $x = mT$:

$$f(mT) = \sum_{n=-\infty}^{\infty} f(nT) \, \operatorname{sinc}(m - n) = f(mT) \cdot 1 + \sum_{n \neq m} f(nT) \cdot 0 = f(mT)$$

since $\operatorname{sinc}(0) = 1$ and $\operatorname{sinc}(k) = 0$ for all nonzero integers $k$.

**Step 7:** Show aliasing when the condition is violated. If $f_s \leq 2f_{\max}$, adjacent spectral replicas overlap. The overlapping region produces:

$$F_s(\omega) = F(\omega) + F\!\left(\omega - \frac{1}{T}\right) + \cdots \neq F(\omega)$$

This frequency folding is irreversible — the original spectrum cannot be recovered.

**Result (Whittaker–Shannon Interpolation Formula):**

$$\boxed{f(x) = \sum_{n=-\infty}^{\infty} f(nT) \, \operatorname{sinc}\!\left(\frac{x - nT}{T}\right), \quad T < \frac{1}{2f_{\max}}}$$
$\blacksquare$

**Intuition:** The sampling theorem states that a band-limited signal lives in a finite-dimensional subspace of $L^2(\mathbb{R})$. The samples $\{f(nT)\}$ are the coordinates in the orthogonal sinc basis. Sampling above the Nyquist rate guarantees the basis is complete; below it, basis functions overlap (alias) and information is irretrievably lost.

### 1.3 Extension to Two Dimensions

For a 2D image $f(x, y)$ with maximum spatial frequencies $f_{\max}^x$ and $f_{\max}^y$:

$$f(x, y) = \sum_{m=-\infty}^{\infty} \sum_{n=-\infty}^{\infty} f(mT_x, nT_y) \, \operatorname{sinc}\!\left(\frac{x - mT_x}{T_x}\right) \operatorname{sinc}\!\left(\frac{y - nT_y}{T_y}\right)$$

provided $T_x < \frac{1}{2f_{\max}^x}$ and $T_y < \frac{1}{2f_{\max}^y}$.

### 1.4 Quantization Theory and SQNR Derivation

**Definition:** Uniform quantization maps a continuous amplitude $f \in [f_{\min}, f_{\max}]$ to one of $L = 2^b$ levels:

$$Q(f) = \Delta \left\lfloor \frac{f - f_{\min}}{\Delta} + \frac{1}{2} \right\rfloor + f_{\min}, \quad \Delta = \frac{f_{\max} - f_{\min}}{L - 1}$$

**Step 1:** Model the quantization error. The error $e = f - Q(f)$ satisfies $|e| \leq \Delta/2$. Under the assumption that $f$ is uniformly distributed within each quantization interval (valid for fine quantization), $e$ is uniformly distributed on $[-\Delta/2, \Delta/2]$.

**Step 2:** Compute the error variance.

$$\sigma_e^2 = \mathbb{E}[e^2] = \int_{-\Delta/2}^{\Delta/2} e^2 \cdot \frac{1}{\Delta} \, de = \frac{1}{\Delta}\left[\frac{e^3}{3}\right]_{-\Delta/2}^{\Delta/2} = \frac{1}{\Delta} \cdot \frac{2(\Delta/2)^3}{3} = \frac{\Delta^2}{12}$$

**Step 3:** Express $\Delta$ in terms of bit depth. For a signal with full-scale range $V_{pp} = f_{\max} - f_{\min}$:

$$\Delta = \frac{V_{pp}}{2^b - 1} \approx \frac{V_{pp}}{2^b} \quad \text{(for large } b\text{)}$$

**Step 4:** Compute signal power. For a full-scale sinusoidal signal $f(t) = \frac{V_{pp}}{2}\sin(2\pi f_0 t)$:

$$\sigma_f^2 = \frac{V_{pp}^2}{8}$$

**Step 5:** Derive the SQNR.

$$\text{SQNR} = \frac{\sigma_f^2}{\sigma_e^2} = \frac{V_{pp}^2 / 8}{V_{pp}^2 / (12 \cdot 2^{2b})} = \frac{12 \cdot 2^{2b}}{8} = \frac{3}{2} \cdot 2^{2b}$$

**Step 6:** Convert to decibels.

$$\text{SQNR}_{\text{dB}} = 10 \log_{10}\!\left(\frac{3}{2} \cdot 2^{2b}\right) = 10 \log_{10}\!\left(\frac{3}{2}\right) + 10 \cdot 2b \cdot \log_{10}(2)$$

$$= 10(0.1761) + 20b(0.30103)$$

$$\boxed{\text{SQNR}_{\text{dB}} = 6.02b + 1.76 \; \text{dB}}$$
$\blacksquare$

**Intuition:** Each additional bit of quantization depth doubles the number of representable levels, halving the step size $\Delta$ and quartering the noise power $\sigma_e^2 \propto \Delta^2$. This gives exactly $10\log_{10}(4) \approx 6.02$ dB improvement per bit.

### 1.5 Proof That Quantization Error is Uncorrelated with the Signal

**Theorem:** Under the uniform error model, $\mathbb{E}[e \cdot f] = 0$ (the quantization error is uncorrelated with the input signal).

**Proof:** Within each quantization interval $[q_k - \Delta/2, \; q_k + \Delta/2]$, the input $f$ is uniformly distributed. The error is $e = f - q_k$, so:

$$\mathbb{E}[e \cdot f \mid f \in \text{interval } k] = \mathbb{E}[e(q_k + e)] = q_k \underbrace{\mathbb{E}[e]}_{=0} + \underbrace{\mathbb{E}[e^2]}_{=\Delta^2/12}$$

Wait — this shows $\mathbb{E}[ef] = \mathbb{E}[e^2] \neq 0$. The precise statement is: $e$ is uncorrelated with $q_k$ (the quantized value), not with $f$ itself. By the total expectation theorem:

$$\text{Cov}(e, q_k) = \mathbb{E}[e \cdot q_k] - \mathbb{E}[e]\mathbb{E}[q_k] = 0 - 0 = 0$$

since $\mathbb{E}[e \mid \text{interval } k] = 0$ and $q_k$ is a constant within each interval.
$\blacksquare$

---

## 2. Algorithm / Method

### 2.1 Image Formation Pipeline

```
Algorithm: Digital Image Formation
Input: Continuous scene irradiance f(x,y), sensor parameters (M, N, b)
Output: Digital image I ∈ {0,...,2^b - 1}^{M×N}

1. OPTICAL FILTERING: Apply anti-aliasing filter h_AA
     f_filtered(x,y) = (f * h_AA)(x,y)
2. SPATIAL SAMPLING: Sample on M×N grid
     For m = 0 to M-1, n = 0 to N-1:
       f[m,n] = f_filtered(m·Δx, n·Δy)
3. QUANTIZATION: Map to discrete levels
     For each (m,n):
       I[m,n] = round(f[m,n] / Δ) · Δ
       I[m,n] = clip(I[m,n], 0, 2^b - 1)
4. Return I
```

### 2.2 Complexity Analysis

- **Sampling:** $O(MN)$ — one evaluation per pixel
- **Quantization:** $O(MN)$ — one rounding operation per pixel
- **Anti-aliasing filter (2D convolution with $k \times k$ kernel):** $O(MNk^2)$
- **Storage:** $O(MNb)$ bits $= O(MN)$ for fixed bit depth

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

- **State:** $s_t = \mathbf{I}_t \in \{0, \ldots, 255\}^{M \times N \times C}$ — the current image observation
- **Action:** $a_t \in \mathcal{A}$ — an image processing operation (e.g., adjust brightness, apply filter)
- **Reward:** $r_t = \mathcal{Q}(\mathbf{I}_{t+1}, \mathbf{I}_{\text{ref}}) - \mathcal{Q}(\mathbf{I}_t, \mathbf{I}_{\text{ref}})$ where $\mathcal{Q}$ is a quality metric (PSNR, SSIM)
- **Transition:** $s_{t+1} = T(s_t, a_t)$ — deterministic application of the chosen action

### 3.2 Why RL?

Traditional image processing applies fixed pipelines. RL enables adaptive, image-specific processing where the agent learns to select operations based on the current image state. Understanding the image formation model (what information was lost during sampling and quantization) informs reward design and state representation choices.

---

## 4. Dataset

- **Name:** scikit-image built-in images
- **Size:** Multiple standard images (512×512 grayscale, 512×512×3 RGB)
- **Auto-download:**

```python
from skimage import data
astronaut = data.astronaut()   # 512×512×3 RGB
camera = data.camera()         # 512×512 grayscale
coins = data.coins()           # 303×384 grayscale
```

---

## 5. Key Equations Summary

| Equation | Meaning |
|----------|---------|
| $f(x,y) = i(x,y) \cdot r(x,y)$ | Image formation model |
| $f_s > 2f_{\max}$ | Nyquist sampling condition |
| $f(x) = \sum_n f(nT)\operatorname{sinc}\!\left(\frac{x-nT}{T}\right)$ | Whittaker–Shannon reconstruction |
| $\sigma_e^2 = \Delta^2 / 12$ | Quantization noise variance |
| $\text{SQNR} = 6.02b + 1.76$ dB | Signal-to-quantization-noise ratio |
| $\Delta = V_{pp} / (2^b - 1)$ | Quantization step size |

---

## 6. References

- Gonzalez, R. C. & Woods, R. E. *Digital Image Processing*, 4th ed., Pearson, 2018.
- Shannon, C. E. "Communication in the Presence of Noise," *Proc. IRE*, 37(1):10–21, 1949.
- Oppenheim, A. V. & Willsky, A. S. *Signals and Systems*, 2nd ed., Prentice Hall, 1997.
- Widrow, B. & Kollár, I. *Quantization Noise*, Cambridge University Press, 2008.
