![Module Logo](../logo.png)

# Image Histograms and Statistics

## Overview

The histogram of an image encodes the empirical distribution of pixel intensities, providing a compact statistical summary that drives many image processing algorithms. This document derives the full theory of histogram equalization as a CDF-based mapping, proves that the equalized histogram maximizes entropy, develops the complete entropy derivation from information theory, and establishes the connection between histogram transformations and RL state normalization.

## Prerequisites

- Probability theory (PDFs, CDFs, random variables)
- Information theory basics (entropy concept)
- Calculus (change of variables, integration)
- Module 01.1–01.4 (image fundamentals)

---

## 1. Mathematical Foundations

### 1.1 Core Definition — Image Histogram

For a discrete image $\mathbf{I}$ with $L$ intensity levels $\{0, 1, \ldots, L-1\}$, the histogram is the function:

$$h(k) = \#\{(m, n) : \mathbf{I}(m, n) = k\}, \quad k = 0, 1, \ldots, L-1$$

The normalized histogram is the empirical probability mass function:

$$p(k) = \frac{h(k)}{MN}, \quad \sum_{k=0}^{L-1} p(k) = 1$$

### 1.2 Full Entropy Derivation

**Definition (Shannon Entropy):** The entropy of a discrete random variable $X$ with PMF $p(x)$ is:

$$H(X) = -\sum_{x} p(x) \log_2 p(x)$$

**Derivation from axioms:**

**Step 1:** Shannon required a measure of "uncertainty" $H(p_1, \ldots, p_n)$ satisfying three axioms:
1. **Continuity:** $H$ is continuous in the $p_i$.
2. **Monotonicity:** For equiprobable outcomes, $H(1/n, \ldots, 1/n)$ is monotonically increasing in $n$.
3. **Recursion (grouping):** $H(p_1, \ldots, p_n) = H(p_1 + p_2, p_3, \ldots, p_n) + (p_1 + p_2) H\!\left(\frac{p_1}{p_1+p_2}, \frac{p_2}{p_1+p_2}\right)$.

**Step 2:** For equiprobable outcomes, define $A(n) = H(1/n, \ldots, 1/n)$. By the grouping axiom applied to $n = s^m$ equiprobable outcomes grouped into $s$-way trees of depth $m$:

$$A(s^m) = m \cdot A(s)$$

**Step 3:** For arbitrary integers $n, s$, choose $m$ such that $s^m \leq n \leq s^{m+1}$. By monotonicity:

$$m \cdot A(s) \leq A(n) \leq (m+1) \cdot A(s)$$

Dividing by $A(s)$ (assumed positive by monotonicity and $A(2) > 0$):

$$m \leq \frac{A(n)}{A(s)} \leq m + 1$$

Also $m \leq \frac{\log n}{\log s} \leq m + 1$, so:

$$\left|\frac{A(n)}{A(s)} - \frac{\log n}{\log s}\right| \leq 1$$

**Step 4:** The same bound holds for $s^m$ in place of $n$, so as $m \to \infty$:

$$\frac{A(n)}{A(s)} = \frac{\log n}{\log s} \implies A(n) = c \log n$$

for some constant $c > 0$ (choosing $c = A(s) / \log s$).

**Step 5:** For the equiprobable case, $A(n) = c \log n = -c \sum_{i=1}^{n} \frac{1}{n} \log \frac{1}{n} = -c\sum_i p_i \log p_i$.

**Step 6:** By the recursion axiom, this extends to non-equiprobable distributions:

$$H(p_1, \ldots, p_n) = -c \sum_{i=1}^{n} p_i \log p_i$$

Choosing $c = 1$ and base-2 logarithms gives entropy in bits.

**Result:**

$$\boxed{H(X) = -\sum_{k=0}^{L-1} p(k) \log_2 p(k)}$$
$\blacksquare$

**Intuition:** Entropy quantifies the average "surprise" per pixel. A completely uniform image ($p(k) = 1$ for one $k$) has $H = 0$ bits (no surprise). A uniform histogram ($p(k) = 1/L$ for all $k$) has maximum entropy $H = \log_2 L$ bits (maximum surprise — every intensity is equally likely).

### 1.3 Histogram Equalization as CDF Mapping

**Goal:** Find a monotone mapping $T: [0, L-1] \to [0, L-1]$ such that the output image has a (approximately) uniform histogram.

**Continuous case derivation:**

**Step 1:** Let $r \in [0, 1]$ be a continuous intensity with PDF $p_r(r)$ and CDF $P_r(r) = \int_0^r p_r(t) \, dt$.

**Step 2:** Define the transformation $s = T(r) = P_r(r)$ (the CDF mapping).

**Step 3:** Since $T$ is monotonically increasing (because $p_r \geq 0$), use the change-of-variables formula. For $s = T(r)$:

$$p_s(s) = p_r(r) \left|\frac{dr}{ds}\right|$$

**Step 4:** Compute $dr/ds$. Since $s = P_r(r)$:

$$\frac{ds}{dr} = p_r(r) \implies \frac{dr}{ds} = \frac{1}{p_r(r)}$$

**Step 5:** Substitute:

$$p_s(s) = p_r(r) \cdot \frac{1}{p_r(r)} = 1$$

**Step 6:** Verify the domain. Since $P_r: [0,1] \to [0,1]$ is monotone from $P_r(0) = 0$ to $P_r(1) = 1$, the output $s$ is uniformly distributed on $[0, 1]$.

**Result:** The CDF mapping $T(r) = P_r(r)$ produces a uniformly distributed output.
$\blacksquare$

**Discrete case:** For a discrete image with $L$ levels:

$$s_k = T(k) = (L - 1) \sum_{j=0}^{k} p(j) = (L - 1) \cdot P_r(k)$$

rounded to the nearest integer level.

### 1.4 Proof That the Equalized Histogram Maximizes Entropy

**Theorem:** Among all intensity distributions on $\{0, 1, \ldots, L-1\}$, the uniform distribution maximizes Shannon entropy.

**Proof (via Lagrange multipliers):**

**Step 1:** We maximize $H = -\sum_{k=0}^{L-1} p_k \log p_k$ subject to $\sum_{k=0}^{L-1} p_k = 1$ and $p_k \geq 0$.

**Step 2:** Form the Lagrangian:

$$\mathcal{L}(p_0, \ldots, p_{L-1}, \lambda) = -\sum_{k=0}^{L-1} p_k \log p_k - \lambda\left(\sum_{k=0}^{L-1} p_k - 1\right)$$

**Step 3:** Take the partial derivative with respect to $p_k$ and set to zero:

$$\frac{\partial \mathcal{L}}{\partial p_k} = -\log p_k - \frac{1}{\ln 2} - \lambda = 0$$

(using $\frac{d}{dp}[p \log_2 p] = \log_2 p + 1/\ln 2$)

$$\log p_k = -\frac{1}{\ln 2} - \lambda = \text{constant}$$

**Step 4:** Since $\log p_k$ is the same for all $k$, we have $p_k = p^*$ for all $k$. Using the constraint $\sum p_k = 1$:

$$Lp^* = 1 \implies p^* = \frac{1}{L}$$

**Step 5:** Verify this is a maximum (not minimum). The Hessian of $H$ with respect to $\mathbf{p}$ is:

$$\frac{\partial^2 H}{\partial p_j \partial p_k} = -\frac{\delta_{jk}}{p_k \ln 2}$$

which is negative definite (diagonal with negative entries), confirming a maximum.

**Step 6:** The maximum entropy is:

$$H_{\max} = -\sum_{k=0}^{L-1} \frac{1}{L} \log_2 \frac{1}{L} = \log_2 L$$

**Result:** Histogram equalization produces a uniform distribution, which achieves $H = \log_2 L$ bits, the maximum possible entropy.
$\blacksquare$

**Alternative proof via Jensen's inequality:**

Since $-\log$ is strictly convex, Jensen's inequality gives:

$$H = \mathbb{E}[-\log p(X)] \leq -\log \mathbb{E}[p(X)]$$

But a cleaner route uses the Gibbs inequality (KL divergence non-negativity):

$$D_{\text{KL}}(p \| u) = \sum_k p_k \log \frac{p_k}{1/L} = \log L - H(p) \geq 0$$

Therefore $H(p) \leq \log L$, with equality iff $p = u$ (uniform).
$\blacksquare$

### 1.5 Histogram Specification (Matching)

**Problem:** Transform an image so its histogram matches a target distribution $p_z(z)$.

**Step 1:** Equalize the source: $s = T(r) = P_r(r)$.

**Step 2:** Compute the target CDF: $G(z) = P_z(z)$.

**Step 3:** Invert: $z = G^{-1}(s) = G^{-1}(P_r(r))$.

**Proof of correctness:** Since $s$ is uniform (by Step 1) and $G^{-1}$ maps $\text{Uniform}[0,1]$ to a random variable with CDF $G$, the output $z$ has the target distribution $p_z$.

---

## 2. Algorithm / Method

### 2.1 Pseudocode: Histogram Equalization

```
Algorithm: Histogram_Equalization
Input: Image I ∈ {0,...,L-1}^{M×N}
Output: Equalized image I_eq

1. COMPUTE HISTOGRAM: For k = 0 to L-1:
     h[k] = count of pixels with value k
2. NORMALIZE: p[k] = h[k] / (M·N)
3. COMPUTE CDF: P[0] = p[0]
     For k = 1 to L-1: P[k] = P[k-1] + p[k]
4. MAP: For each pixel (m,n):
     I_eq[m,n] = round((L-1) · P[I[m,n]])
5. Return I_eq
```

### 2.2 Complexity Analysis

- **Time:** $O(MN + L)$ — one pass to compute histogram, one pass to map pixels
- **Space:** $O(L)$ for the histogram and CDF arrays

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

- **State:** $s_t = (\mathbf{I}_t, h_t)$ — the current image and its histogram
- **Action:** $a_t \in \{\text{equalize}, \text{stretch\_contrast}, \text{gamma}(\gamma), \text{clip}(c_{\text{lo}}, c_{\text{hi}})\}$
- **Reward:** $r_t = \alpha \cdot \Delta H + \beta \cdot \Delta\text{PSNR}$ — weighted improvement in entropy and quality
- **Transition:** Apply the histogram transformation to produce $\mathbf{I}_{t+1}$

### 3.2 Why RL?

Histogram equalization is a one-shot global operation that may produce over-enhanced results. RL agents can learn adaptive, multi-step histogram modifications: partial equalization, CLAHE-like local adjustments, or content-aware tone mapping. The entropy-maximization property provides a natural information-theoretic reward signal.

---

## 4. Dataset

- **Name:** scikit-image built-in images
- **Size:** Standard test images with varying contrast characteristics
- **Auto-download:**

```python
from skimage import data, exposure
camera = data.camera()
equalized = exposure.equalize_hist(camera)
```

---

## 5. Key Equations Summary

| Equation | Meaning |
|----------|---------|
| $p(k) = h(k) / MN$ | Normalized histogram (empirical PMF) |
| $H = -\sum_k p(k) \log_2 p(k)$ | Shannon entropy of image |
| $s = T(r) = (L-1) \cdot P_r(r)$ | Histogram equalization transform |
| $p_s(s) = 1$ (uniform) | Output distribution after equalization |
| $H_{\max} = \log_2 L$ | Maximum entropy (uniform distribution) |
| $D_{\text{KL}}(p\|u) = \log L - H(p) \geq 0$ | Gibbs inequality proof of max entropy |

---

## 6. References

- Shannon, C. E. "A Mathematical Theory of Communication," *Bell System Technical Journal*, 27:379–423, 1948.
- Gonzalez, R. C. & Woods, R. E. *Digital Image Processing*, 4th ed., Pearson, 2018, Ch. 3.
- Pizer, S. M. et al. "Adaptive Histogram Equalization and Its Variations," *Computer Vision, Graphics, and Image Processing*, 39(3):355–368, 1987.
- Cover, T. M. & Thomas, J. A. *Elements of Information Theory*, 2nd ed., Wiley, 2006.
