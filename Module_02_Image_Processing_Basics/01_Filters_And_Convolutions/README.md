![Module Logo](../logo.png)

# Filters and Convolution

## Overview

Convolution is the foundational operation of image processing and convolutional neural networks. This document proves the convolution theorem (spatial convolution equals frequency-domain multiplication), derives the Gaussian filter from the heat diffusion equation, proves the separability of 2D Gaussian convolution, and establishes computational complexity results for separable filtering.

## Prerequisites

- Fourier analysis (Fourier transform, inverse transform)
- Calculus (partial differential equations basics)
- Linear algebra (matrix–vector products)
- Module 01 (image representation)

---

## 1. Mathematical Foundations

### 1.1 Core Definition — Discrete 2D Convolution

The discrete 2D convolution of an image $f[m,n]$ with a kernel $h[m,n]$ is:

$$(f * h)[m, n] = \sum_{i=-\infty}^{\infty} \sum_{j=-\infty}^{\infty} f[i, j] \, h[m - i, n - j]$$

For a finite kernel of size $(2K+1) \times (2K+1)$:

$$(f * h)[m, n] = \sum_{i=-K}^{K} \sum_{j=-K}^{K} f[m - i, n - j] \, h[i, j]$$

### 1.2 Proof of the Convolution Theorem

**Theorem:** Convolution in the spatial domain corresponds to multiplication in the frequency domain:

$$\mathcal{F}\{f * h\} = \mathcal{F}\{f\} \cdot \mathcal{F}\{h\}$$

**Proof:**

**Step 1:** Write the Fourier transform of the convolution:

$$\mathcal{F}\{f * h\}(\omega_x, \omega_y) = \sum_{m} \sum_{n} \left[\sum_{i} \sum_{j} f[i,j] \, h[m-i, n-j]\right] e^{-j2\pi(\omega_x m + \omega_y n)}$$

**Step 2:** Interchange the order of summation (justified by absolute convergence for finite-energy signals):

$$= \sum_{i} \sum_{j} f[i,j] \sum_{m} \sum_{n} h[m-i, n-j] \, e^{-j2\pi(\omega_x m + \omega_y n)}$$

**Step 3:** Substitute $m' = m - i$, $n' = n - j$ in the inner sums:

$$\sum_{m} \sum_{n} h[m-i, n-j] \, e^{-j2\pi(\omega_x m + \omega_y n)} = \sum_{m'} \sum_{n'} h[m', n'] \, e^{-j2\pi(\omega_x(m'+i) + \omega_y(n'+j))}$$

**Step 4:** Factor the exponential:

$$= e^{-j2\pi(\omega_x i + \omega_y j)} \sum_{m'} \sum_{n'} h[m', n'] \, e^{-j2\pi(\omega_x m' + \omega_y n')}$$

$$= e^{-j2\pi(\omega_x i + \omega_y j)} \cdot H(\omega_x, \omega_y)$$

**Step 5:** Substitute back:

$$\mathcal{F}\{f * h\} = \sum_{i} \sum_{j} f[i,j] \, e^{-j2\pi(\omega_x i + \omega_y j)} \cdot H(\omega_x, \omega_y)$$

$$= F(\omega_x, \omega_y) \cdot H(\omega_x, \omega_y)$$

**Result:**

$$\boxed{\mathcal{F}\{f * h\} = F(\omega_x, \omega_y) \cdot H(\omega_x, \omega_y)}$$
$\blacksquare$

**Corollary:** The dual theorem also holds: $\mathcal{F}\{f \cdot h\} = F * H$ (pointwise multiplication in space corresponds to convolution in frequency).

**Intuition:** Every linear shift-invariant system can be characterized by its frequency response $H(\omega)$. Convolution "filters" the input by amplifying or attenuating each frequency component according to $H(\omega)$. The convolution theorem enables $O(MN \log(MN))$ filtering via FFT instead of $O(MNK^2)$ direct computation.

### 1.3 Derivation of the Gaussian Filter from the Diffusion Equation

**Step 1:** The isotropic heat (diffusion) equation in 2D is:

$$\frac{\partial u}{\partial t} = D \nabla^2 u = D\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$

where $u(x, y, t)$ is the temperature (or image intensity) and $D$ is the diffusion coefficient.

**Step 2:** Take the Fourier transform in $(x, y)$:

$$\frac{\partial \hat{u}}{\partial t} = D(-4\pi^2(\omega_x^2 + \omega_y^2)) \hat{u} = -D \cdot 4\pi^2 \|\boldsymbol{\omega}\|^2 \hat{u}$$

**Step 3:** This is a first-order ODE in $t$ with solution:

$$\hat{u}(\omega_x, \omega_y, t) = \hat{u}_0(\omega_x, \omega_y) \cdot e^{-4\pi^2 D t (\omega_x^2 + \omega_y^2)}$$

**Step 4:** The impulse response (Green's function) has $\hat{u}_0 = 1$. Its inverse Fourier transform is:

$$G(x, y; t) = \mathcal{F}^{-1}\left\{e^{-4\pi^2 Dt(\omega_x^2 + \omega_y^2)}\right\}$$

**Step 5:** Evaluate using the known transform pair $\mathcal{F}^{-1}\{e^{-\alpha\omega^2}\} = \sqrt{\pi/\alpha} \, e^{-\pi^2 x^2/\alpha}$:

$$G(x, y; t) = \frac{1}{4\pi D t} \exp\!\left(-\frac{x^2 + y^2}{4Dt}\right)$$

**Step 6:** Identify with the Gaussian kernel by setting $\sigma^2 = 2Dt$:

$$G_\sigma(x, y) = \frac{1}{2\pi\sigma^2} \exp\!\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

**Result:** Gaussian filtering is equivalent to solving the diffusion equation. The parameter $\sigma$ controls the diffusion time (blur level).

$$\boxed{G_\sigma(x,y) = \frac{1}{2\pi\sigma^2}\exp\!\left(-\frac{x^2+y^2}{2\sigma^2}\right)}$$
$\blacksquare$

**Intuition:** Blurring with a Gaussian is literally letting the image "diffuse" like heat. Longer diffusion time (larger $\sigma$) produces more blurring. This explains why Gaussian blur is the natural choice for scale-space analysis.

### 1.4 Proof of Separability of the Gaussian Filter

**Theorem:** The 2D Gaussian kernel can be decomposed as the outer product of two 1D Gaussians:

$$G_\sigma(x, y) = g_\sigma(x) \cdot g_\sigma(y)$$

where $g_\sigma(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-x^2/(2\sigma^2)}$.

**Proof:**

**Step 1:** Write the 2D Gaussian:

$$G_\sigma(x, y) = \frac{1}{2\pi\sigma^2} \exp\!\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

**Step 2:** Factor the exponential:

$$= \frac{1}{2\pi\sigma^2} \exp\!\left(-\frac{x^2}{2\sigma^2}\right) \exp\!\left(-\frac{y^2}{2\sigma^2}\right)$$

**Step 3:** Factor the normalization constant:

$$= \left[\frac{1}{\sqrt{2\pi}\sigma} \exp\!\left(-\frac{x^2}{2\sigma^2}\right)\right] \cdot \left[\frac{1}{\sqrt{2\pi}\sigma} \exp\!\left(-\frac{y^2}{2\sigma^2}\right)\right]$$

$$= g_\sigma(x) \cdot g_\sigma(y)$$

**Step 4:** Therefore, 2D Gaussian convolution can be performed as two successive 1D convolutions:

$$(f * G_\sigma)[m,n] = ((f *_x g_\sigma) *_y g_\sigma)[m,n]$$

where $*_x$ denotes convolution along rows and $*_y$ along columns.
$\blacksquare$

**Complexity Consequence:**

- Direct 2D convolution with kernel size $(2K+1)^2$: $O(MN \cdot (2K+1)^2) = O(MNK^2)$
- Separable convolution: $O(MN \cdot 2(2K+1)) = O(MNK)$

The separable decomposition reduces the complexity from quadratic to linear in the kernel size.

### 1.5 Proof That Only the Gaussian Is Separable and Rotationally Symmetric

**Theorem:** The Gaussian is the only kernel that is simultaneously separable and rotationally symmetric.

**Proof:**

**Step 1:** Rotational symmetry requires $h(x, y) = \phi(x^2 + y^2)$ for some function $\phi$.

**Step 2:** Separability requires $h(x, y) = f(x) \cdot g(y)$.

**Step 3:** Combining: $f(x)g(y) = \phi(x^2 + y^2)$. Set $y = 0$: $f(x)g(0) = \phi(x^2)$, so $f(x) = \frac{\phi(x^2)}{g(0)}$.

**Step 4:** Set $x = 0$: $f(0)g(y) = \phi(y^2)$, so $g(y) = \frac{\phi(y^2)}{f(0)}$.

**Step 5:** Substitute back: $\phi(x^2) \cdot \phi(y^2) = f(0)g(0) \cdot \phi(x^2 + y^2)$.

**Step 6:** Let $\psi(u) = \log \phi(u)$. Then: $\psi(x^2) + \psi(y^2) = C + \psi(x^2 + y^2)$.

**Step 7:** This is Cauchy's functional equation $\psi(a) + \psi(b) = C + \psi(a + b)$ with $\psi'(u) = \tilde{\psi}(u) - C$ satisfying $\tilde{\psi}(a + b) = \tilde{\psi}(a) + \tilde{\psi}(b)$. The only continuous solution is $\tilde{\psi}(u) = \alpha u$, giving $\phi(u) = e^{\alpha u + \beta}$, which is a Gaussian (with $\alpha < 0$ for integrability).
$\blacksquare$

---

## 2. Algorithm / Method

### 2.1 Pseudocode: Separable Gaussian Convolution

```
Algorithm: Separable_Gaussian_Blur
Input: Image I ∈ ℝ^{M×N}, standard deviation σ
Output: Blurred image I_blur

1. KERNEL SIZE: K = ⌈3σ⌉
2. BUILD 1D KERNEL: For i = -K to K:
     g[i] = exp(-i² / (2σ²))
   Normalize: g = g / sum(g)
3. ROW CONVOLUTION: For each row m:
     I_temp[m,:] = convolve1d(I[m,:], g)
4. COLUMN CONVOLUTION: For each column n:
     I_blur[:,n] = convolve1d(I_temp[:,n], g)
5. Return I_blur
```

### 2.2 Complexity Analysis

- **Direct 2D convolution:** $O(MN(2K+1)^2)$ — Time; $O((2K+1)^2)$ — Space
- **Separable convolution:** $O(2MN(2K+1))$ — Time; $O(2K+1)$ — Space
- **FFT-based convolution:** $O(MN\log(MN))$ — Time; $O(MN)$ — Space

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

- **State:** $s_t = \mathbf{I}_t$ — the current image
- **Action:** $a_t = (\text{filter\_type}, \sigma)$ — choice of filter and its parameter
- **Reward:** $r_t = \text{PSNR}(\mathbf{I}_{t+1}, \mathbf{I}_{\text{clean}}) - \text{PSNR}(\mathbf{I}_t, \mathbf{I}_{\text{clean}})$
- **Transition:** $s_{t+1} = s_t * h_{a_t}$ — convolution with the selected filter

### 3.2 Why RL?

Filter selection and parameter tuning (kernel size, $\sigma$) are traditionally done manually or via grid search. RL can learn adaptive filtering policies that select different filters for different image regions based on local content, noise level, and edge structure.

---

## 4. Dataset

- **Name:** scikit-image built-in images
- **Size:** Standard test images
- **Auto-download:**

```python
from skimage import data, filters
camera = data.camera()
blurred = filters.gaussian(camera, sigma=2)
```

---

## 5. Key Equations Summary

| Equation | Meaning |
|----------|---------|
| $(f*h)[m,n] = \sum_{i,j} f[i,j]h[m-i,n-j]$ | 2D discrete convolution |
| $\mathcal{F}\{f*h\} = F \cdot H$ | Convolution theorem |
| $G_\sigma(x,y) = \frac{1}{2\pi\sigma^2}e^{-(x^2+y^2)/(2\sigma^2)}$ | 2D Gaussian kernel |
| $G_\sigma(x,y) = g_\sigma(x) \cdot g_\sigma(y)$ | Separability of Gaussian |
| $\partial u/\partial t = D\nabla^2 u$ | Diffusion equation (Gaussian origin) |

---

## 6. References

- Oppenheim, A. V. & Schafer, R. W. *Discrete-Time Signal Processing*, 3rd ed., Pearson, 2010.
- Lindeberg, T. "Scale-Space Theory: A Basic Tool for Analyzing Structures at Different Scales," *J. Applied Statistics*, 1994.
- Gonzalez, R. C. & Woods, R. E. *Digital Image Processing*, 4th ed., Pearson, 2018, Ch. 3.
- Weickert, J. *Anisotropic Diffusion in Image Processing*, Teubner, 1998.
