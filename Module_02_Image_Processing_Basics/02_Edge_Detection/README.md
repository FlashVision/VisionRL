![Module Logo](../logo.png)

# Edge Detection

## Overview

Edges are the most information-dense features in images, encoding object boundaries, texture boundaries, and depth discontinuities. This document derives the Sobel operator from Taylor series expansion of image gradients, provides the complete Canny edge detection theory with proofs for non-maximum suppression and hysteresis thresholding, and derives the Laplacian of Gaussian (LoG) as the optimal edge detector under scale-space theory.

## Prerequisites

- Multivariable calculus (gradients, Hessians, Taylor expansion)
- Linear algebra (eigenvalues of $2 \times 2$ matrices)
- Convolution theory (Module 02.1)
- Gaussian filtering (Module 02.1)

---

## 1. Mathematical Foundations

### 1.1 Core Definition — Image Gradient

The gradient of a continuous image $f(x, y)$ is:

$$\nabla f = \begin{pmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{pmatrix}$$

The gradient magnitude and direction are:

$$|\nabla f| = \sqrt{\left(\frac{\partial f}{\partial x}\right)^2 + \left(\frac{\partial f}{\partial y}\right)^2}, \quad \theta = \arctan\!\left(\frac{\partial f / \partial y}{\partial f / \partial x}\right)$$

Edges occur at locations where $|\nabla f|$ is large.

### 1.2 Full Derivation of the Sobel Operator from Taylor Expansion

**Step 1:** Consider a $3 \times 3$ neighborhood around pixel $(m, n)$. Label the nine pixel values using the notation:

$$\begin{pmatrix} f_{-1,-1} & f_{-1,0} & f_{-1,1} \\ f_{0,-1} & f_{0,0} & f_{0,1} \\ f_{1,-1} & f_{1,0} & f_{1,1} \end{pmatrix}$$

**Step 2:** Taylor-expand $f(m+i, n+j)$ around $(m, n)$:

$$f(m+i, n+j) = f + i f_x + j f_y + \frac{1}{2}(i^2 f_{xx} + 2ij f_{xy} + j^2 f_{yy}) + \cdots$$

where $f_x = \partial f/\partial x$, etc., all evaluated at $(m, n)$.

**Step 3:** Estimate $\partial f/\partial x$ using finite differences. The simplest central difference is:

$$f_x \approx \frac{f(m, n+1) - f(m, n-1)}{2}$$

**Step 4:** Improve the estimate by averaging over three rows to reduce noise. The Sobel operator uses a weighted average with weights $(1, 2, 1)$ (a discrete approximation to Gaussian smoothing):

$$f_x \approx \frac{1}{4}\left[(f_{-1,1} - f_{-1,-1}) + 2(f_{0,1} - f_{0,-1}) + (f_{1,1} - f_{1,-1})\right]$$

**Step 5:** This is equivalent to convolution with the Sobel kernel:

$$\mathbf{G}_x = \frac{1}{4}\begin{pmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{pmatrix}$$

**Step 6:** Verify this as separable convolution. The Sobel $x$-kernel factors as:

$$\mathbf{G}_x = \frac{1}{4}\begin{pmatrix} 1 \\ 2 \\ 1 \end{pmatrix} \begin{pmatrix} -1 & 0 & 1 \end{pmatrix} = \mathbf{s} \cdot \mathbf{d}^T$$

where $\mathbf{s} = \frac{1}{4}(1, 2, 1)^T$ is smoothing (binomial approximation to Gaussian) and $\mathbf{d} = (-1, 0, 1)^T$ is the central difference. The $y$-kernel is $\mathbf{G}_y = \mathbf{d} \cdot \mathbf{s}^T$ (differentiation along $y$, smoothing along $x$).

**Step 7:** The Sobel gradient magnitude is:

$$|\nabla f|_{\text{Sobel}} = \sqrt{(f * \mathbf{G}_x)^2 + (f * \mathbf{G}_y)^2}$$

**Result:** The Sobel operator computes a noise-smoothed gradient by combining Gaussian-like smoothing perpendicular to the differentiation direction with central-difference differentiation along it.
$\blacksquare$

### 1.3 Canny Edge Detection — Complete Theory

Canny (1986) derived the optimal edge detector by posing three criteria and solving a variational optimization problem.

**Canny's Three Criteria:**

1. **Good detection:** Maximize the signal-to-noise ratio (minimize missed edges and false positives).
2. **Good localization:** Detected edges should be as close as possible to the true edge location.
3. **Single response:** Only one response per true edge.

**Step 1 — Detection criterion.** For a step edge of amplitude $A$ corrupted by Gaussian noise $n(x)$ with variance $\sigma_n^2$, the SNR of a 1D filter $h(x)$ supported on $[-W, W]$ is:

$$\text{SNR} = \frac{A \left|\int_{-W}^{0} h(x) \, dx\right|}{\sigma_n \sqrt{\int_{-W}^{W} h^2(x) \, dx}}$$

**Step 2 — Localization criterion.** The localization is measured by:

$$\text{LOC} = \frac{A |h'(0)|}{\sigma_n \sqrt{\int_{-W}^{W} h'^2(x) \, dx}}$$

**Step 3 — Joint optimization.** Canny maximized the product $\Sigma = \text{SNR} \times \text{LOC}$ subject to the single-response constraint (mean distance between zero-crossings of the filter output equals $2W$). Using calculus of variations, the optimal filter is the first derivative of a Gaussian:

$$h_{\text{Canny}}(x) = -\frac{x}{\sigma^2} e^{-x^2/(2\sigma^2)}$$

which is precisely $-G_\sigma'(x)$.

**Step 4 — Non-Maximum Suppression (NMS).** After computing gradient magnitude $|\nabla f|$ and direction $\theta$:

- For each pixel, check the two neighbors along the gradient direction $\theta$.
- If $|\nabla f|$ at the pixel is not strictly greater than both neighbors, suppress it (set to zero).

**Proof that NMS produces single-pixel-wide edges:** Along the gradient direction, the gradient magnitude profile of a smoothed step edge is a single-peaked function (Gaussian derivative is unimodal). NMS retains only the maximum, which is a single point. Since the gradient direction is perpendicular to the edge, this produces a one-pixel-wide ridge at the edge location.

**Step 5 — Hysteresis Thresholding.** Use two thresholds $T_{\text{low}} < T_{\text{high}}$:

1. **Strong edges:** Pixels with $|\nabla f| \geq T_{\text{high}}$ — definitely edges.
2. **Weak edges:** Pixels with $T_{\text{low}} \leq |\nabla f| < T_{\text{high}}$ — edges only if connected to a strong edge.
3. **Non-edges:** Pixels with $|\nabla f| < T_{\text{low}}$ — discarded.

**Why hysteresis works:** A single threshold either misses weak but real edges (too high) or includes noise (too low). Hysteresis uses connectivity: weak edges connected to strong edges are likely real (continuity prior). This is formally equivalent to graph-based connected component analysis on the strong+weak edge map.

### 1.4 Laplacian of Gaussian (LoG) Derivation

**Step 1:** The Laplacian operator detects zero-crossings of the second derivative:

$$\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$$

**Step 2:** Apply Gaussian smoothing before the Laplacian for noise robustness:

$$\nabla^2 (G_\sigma * f) = (\nabla^2 G_\sigma) * f$$

The interchange is valid because differentiation and convolution are both linear operations, and $G_\sigma$ is infinitely differentiable.

**Step 3:** Compute $\nabla^2 G_\sigma$:

$$\frac{\partial^2 G_\sigma}{\partial x^2} = \frac{1}{2\pi\sigma^2}\left(\frac{x^2}{\sigma^4} - \frac{1}{\sigma^2}\right)e^{-(x^2+y^2)/(2\sigma^2)}$$

Similarly for $\partial^2/\partial y^2$. Summing:

$$\nabla^2 G_\sigma(x,y) = \frac{1}{2\pi\sigma^4}\left(\frac{x^2 + y^2}{\sigma^2} - 2\right)e^{-(x^2+y^2)/(2\sigma^2)}$$

**Step 4:** Let $r^2 = x^2 + y^2$:

$$\text{LoG}(x, y) = -\frac{1}{\pi\sigma^4}\left(1 - \frac{r^2}{2\sigma^2}\right)e^{-r^2/(2\sigma^2)}$$

(the negative sign is conventional so that edges produce positive responses).

**Step 5:** The zero-crossing of the LoG occurs at $r = \sigma\sqrt{2}$, which defines the characteristic scale of detected edges.

**Result:**

$$\boxed{\text{LoG}(x,y) = -\frac{1}{\pi\sigma^4}\left(1 - \frac{x^2+y^2}{2\sigma^2}\right)e^{-(x^2+y^2)/(2\sigma^2)}}$$
$\blacksquare$

---

## 2. Algorithm / Method

### 2.1 Pseudocode: Canny Edge Detection

```
Algorithm: Canny_Edge_Detection
Input: Image I, σ, T_low, T_high
Output: Binary edge map E

1. SMOOTH: I_s = I * G_σ
2. GRADIENT: G_x = I_s * Sobel_x, G_y = I_s * Sobel_y
3. MAGNITUDE: M = sqrt(G_x² + G_y²)
4. DIRECTION: θ = atan2(G_y, G_x), quantize to {0°, 45°, 90°, 135°}
5. NMS: For each pixel (m,n):
     Check neighbors along direction θ[m,n]
     If M[m,n] < either neighbor: M_nms[m,n] = 0
     Else: M_nms[m,n] = M[m,n]
6. HYSTERESIS:
     Strong = {(m,n) : M_nms[m,n] ≥ T_high}
     Weak = {(m,n) : T_low ≤ M_nms[m,n] < T_high}
     E = Strong ∪ {weak pixels connected to Strong}
7. Return E
```

### 2.2 Complexity Analysis

- **Gaussian smoothing:** $O(MNK)$ (separable, $K = \lceil 3\sigma \rceil$)
- **Gradient computation:** $O(MN)$
- **NMS:** $O(MN)$
- **Hysteresis (BFS/DFS):** $O(MN)$
- **Total:** $O(MN(K + 1)) = O(MNK)$

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

- **State:** $s_t = (\mathbf{I}, \sigma_t, T_{\text{low},t}, T_{\text{high},t})$ — image and current detector parameters
- **Action:** $a_t \in \{\text{increase\_}\sigma, \text{decrease\_}\sigma, \text{adjust\_thresholds}, \ldots\}$
- **Reward:** $r_t = F_1(\text{detected edges}, \text{ground truth})$ — edge detection F1 score
- **Transition:** Recompute edges with updated parameters

### 3.2 Why RL?

Canny parameters ($\sigma$, $T_{\text{low}}$, $T_{\text{high}}$) are typically set manually. RL can learn image-adaptive parameter selection policies: noisy images need larger $\sigma$, low-contrast edges need lower thresholds. The agent learns to balance detection rate and false alarm rate adaptively.

---

## 4. Dataset

- **Name:** BSDS500 (Berkeley Segmentation Dataset) or scikit-image built-ins
- **Size:** 500 images with human-annotated edge maps
- **Auto-download:**

```python
from skimage import data, feature
camera = data.camera()
edges = feature.canny(camera, sigma=1.5)
```

---

## 5. Key Equations Summary

| Equation | Meaning |
|----------|---------|
| $\nabla f = (f_x, f_y)^T$ | Image gradient vector |
| $\mathbf{G}_x = \mathbf{s} \cdot \mathbf{d}^T$ | Sobel as separable smoothing + differentiation |
| $h_{\text{Canny}} = -G_\sigma'(x)$ | Optimal edge filter (Canny derivation) |
| $\text{LoG} = \nabla^2 G_\sigma$ | Laplacian of Gaussian |
| $\text{SNR} \times \text{LOC}$ | Canny optimality criterion |

---

## 6. References

- Canny, J. "A Computational Approach to Edge Detection," *IEEE Trans. PAMI*, 8(6):679–698, 1986.
- Marr, D. & Hildreth, E. "Theory of Edge Detection," *Proc. Royal Society B*, 207:187–217, 1980.
- Gonzalez, R. C. & Woods, R. E. *Digital Image Processing*, 4th ed., Pearson, 2018, Ch. 10.
- Lindeberg, T. *Scale-Space Theory in Computer Vision*, Springer, 1994.
