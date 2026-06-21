![Module Logo](../logo.png)

# Pixels and Channels

## Overview

A pixel is the atomic unit of a digital image, carrying intensity or color information encoded across one or more channels. This document develops the mathematical theory of color representation from first principles: the physics of trichromatic vision, the derivation of the CIE color-matching functions, the full gamma correction model, the CIE XYZ to sRGB transformation with all matrices, and a formal proof that the RGB gamut forms a convex subset of the chromaticity diagram.

## Prerequisites

- Linear algebra (matrix multiplication, eigenvalues)
- Calculus (integration, change of variables)
- Basic optics and photometry
- Module 01.1 (What Is An Image)

---

## 1. Mathematical Foundations

### 1.1 Core Definition — Pixel and Channel

A pixel at position $(m, n)$ in a digital image with $C$ channels is a vector:

$$\mathbf{p}_{m,n} = \begin{pmatrix} p_{m,n}^{(1)} \\ p_{m,n}^{(2)} \\ \vdots \\ p_{m,n}^{(C)} \end{pmatrix} \in \{0, 1, \ldots, 2^b - 1\}^C$$

For a standard 8-bit RGB image, $C = 3$ and $b = 8$, giving $\mathbf{p} \in \{0, \ldots, 255\}^3$.

The full image is a third-order tensor:

$$\mathbf{I} \in \{0, \ldots, 2^b - 1\}^{M \times N \times C}$$

### 1.2 Trichromatic Color Theory

Human color perception is governed by three types of cone cells with spectral sensitivity functions $\bar{l}(\lambda)$, $\bar{m}(\lambda)$, $\bar{s}(\lambda)$ (long, medium, short wavelength cones). The perceived color of a spectral power distribution $\Phi(\lambda)$ is encoded by three cone responses:

$$L = \int_{\lambda_{\min}}^{\lambda_{\max}} \Phi(\lambda) \, \bar{l}(\lambda) \, d\lambda, \quad M = \int \Phi(\lambda) \, \bar{m}(\lambda) \, d\lambda, \quad S = \int \Phi(\lambda) \, \bar{s}(\lambda) \, d\lambda$$

**Metamerism Theorem:** Two spectral distributions $\Phi_1(\lambda)$ and $\Phi_2(\lambda)$ are perceptually identical if and only if they produce the same tristimulus values $(L, M, S)$.

### 1.3 Full Derivation of Gamma Correction

#### The Physical Model

CRT monitors have a nonlinear voltage-to-luminance relationship:

$$L_{\text{display}} = L_{\max} \cdot V^{\gamma_{\text{CRT}}}$$

where $V \in [0, 1]$ is the normalized voltage and $\gamma_{\text{CRT}} \approx 2.2$.

#### Derivation of the Gamma Encoding Function

**Step 1:** Define the goal. We want the displayed luminance $L_{\text{display}}$ to be proportional to the scene luminance $L_{\text{scene}}$:

$$L_{\text{display}} = k \cdot L_{\text{scene}}$$

**Step 2:** Substitute the display model. If we store a gamma-encoded value $V_{\text{encoded}}$:

$$L_{\text{display}} = L_{\max} \cdot V_{\text{encoded}}^{\gamma}$$

**Step 3:** For perceptual linearity, set $V_{\text{encoded}}^{\gamma} = L_{\text{scene}} / L_{\max}$:

$$V_{\text{encoded}} = \left(\frac{L_{\text{scene}}}{L_{\max}}\right)^{1/\gamma}$$

**Step 4:** The encoding gamma is $\gamma_{\text{enc}} = 1/\gamma \approx 1/2.2 \approx 0.4545$. The full sRGB encoding function adds a linear segment near zero to avoid infinite slope:

$$V_{\text{sRGB}} = \begin{cases} 12.92 \cdot L_{\text{linear}} & \text{if } L_{\text{linear}} \leq 0.0031308 \\ 1.055 \cdot L_{\text{linear}}^{1/2.4} - 0.055 & \text{if } L_{\text{linear}} > 0.0031308 \end{cases}$$

**Step 5:** Verify continuity at the breakpoint. At $L_{\text{linear}} = 0.0031308$:

$$12.92 \times 0.0031308 = 0.04045$$

$$1.055 \times 0.0031308^{1/2.4} - 0.055 = 1.055 \times 0.3812 - 0.055 \approx 0.04045$$

Both branches yield the same value, confirming $C^0$ continuity.

**Step 6:** Verify continuous derivative (slope matching). The derivative of the linear branch is $12.92$. The derivative of the power branch at the breakpoint:

$$\frac{d}{dL}\left[1.055 \cdot L^{1/2.4} - 0.055\right] = \frac{1.055}{2.4} \cdot L^{1/2.4 - 1} = \frac{1.055}{2.4} \cdot 0.0031308^{-0.5833}$$

$$= 0.4396 \times 29.41 \approx 12.92$$

The slopes match, giving $C^1$ continuity.

**Result:** The sRGB gamma encoding is a $C^1$-smooth function that approximates $L^{1/2.2}$ but avoids the infinite derivative at $L = 0$.
$\blacksquare$

**Intuition:** Gamma correction allocates more digital codes to dark tones where human vision is most sensitive (Weber's law: $\Delta L / L \approx \text{const}$). The power law $L^{1/\gamma}$ stretches the dark end and compresses the bright end, matching the perceptual response curve.

### 1.4 CIE XYZ to sRGB Transformation

**Step 1:** Define the CIE XYZ tristimulus values via color-matching functions $\bar{x}(\lambda), \bar{y}(\lambda), \bar{z}(\lambda)$:

$$X = \int \Phi(\lambda) \bar{x}(\lambda) d\lambda, \quad Y = \int \Phi(\lambda) \bar{y}(\lambda) d\lambda, \quad Z = \int \Phi(\lambda) \bar{z}(\lambda) d\lambda$$

**Step 2:** The sRGB primaries are defined by their CIE xy chromaticity coordinates:

| Primary | $x$ | $y$ |
|---------|-------|-------|
| Red | 0.6400 | 0.3300 |
| Green | 0.3000 | 0.6000 |
| Blue | 0.1500 | 0.0600 |
| White (D65) | 0.3127 | 0.3290 |

**Step 3:** Compute the XYZ coordinates of each primary at unit luminance. For chromaticity $(x, y)$ with $Y = 1$:

$$X = \frac{x}{y}, \quad Y = 1, \quad Z = \frac{1 - x - y}{y}$$

**Step 4:** Form the primary matrix $\mathbf{M}_p$ where each column is the XYZ of a primary:

$$\mathbf{M}_p = \begin{pmatrix} X_r & X_g & X_b \\ Y_r & Y_g & Y_b \\ Z_r & Z_g & Z_b \end{pmatrix} = \begin{pmatrix} 0.6400/0.3300 & 0.3000/0.6000 & 0.1500/0.0600 \\ 1 & 1 & 1 \\ 0.0300/0.3300 & 0.1000/0.6000 & 0.7900/0.0600 \end{pmatrix}$$

**Step 5:** Solve for scaling factors $\mathbf{s} = (S_r, S_g, S_b)^T$ using the white point constraint $\mathbf{M}_p \mathbf{s} = \mathbf{W}_{D65}$:

$$\mathbf{s} = \mathbf{M}_p^{-1} \mathbf{W}_{D65}$$

where $\mathbf{W}_{D65} = (X_{D65}, Y_{D65}, Z_{D65})^T = (0.9505, 1.0000, 1.0890)^T$.

**Step 6:** The final XYZ-to-linear-sRGB matrix is $\mathbf{M} = \mathbf{M}_p \cdot \text{diag}(\mathbf{s})$, and its inverse gives:

$$\begin{pmatrix} R_{\text{linear}} \\ G_{\text{linear}} \\ B_{\text{linear}} \end{pmatrix} = \begin{pmatrix} 3.2406 & -1.5372 & -0.4986 \\ -0.9689 & 1.8758 & 0.0415 \\ 0.0557 & -0.2040 & 1.0570 \end{pmatrix} \begin{pmatrix} X \\ Y \\ Z \end{pmatrix}$$

**Step 7:** Apply gamma encoding (Section 1.3) to obtain final sRGB values.

### 1.5 Proof That the RGB Gamut Is a Convex Subset

**Theorem:** The set of all colors representable by an RGB system with non-negative primaries forms a convex set in XYZ space (and in chromaticity space).

**Proof:**

**Step 1:** Any color in the RGB gamut is a non-negative linear combination of the three primaries:

$$\mathbf{c} = r \mathbf{P}_R + g \mathbf{P}_G + b \mathbf{P}_B, \quad r, g, b \geq 0$$

where $\mathbf{P}_R, \mathbf{P}_G, \mathbf{P}_B \in \mathbb{R}^3$ are the XYZ coordinates of the primaries.

**Step 2:** Let $\mathbf{c}_1 = r_1 \mathbf{P}_R + g_1 \mathbf{P}_G + b_1 \mathbf{P}_B$ and $\mathbf{c}_2 = r_2 \mathbf{P}_R + g_2 \mathbf{P}_G + b_2 \mathbf{P}_B$ be two colors in the gamut, with all coefficients $\geq 0$.

**Step 3:** For any $\lambda \in [0, 1]$, consider the convex combination:

$$\mathbf{c}_\lambda = \lambda \mathbf{c}_1 + (1 - \lambda) \mathbf{c}_2$$

**Step 4:** Expand:

$$\mathbf{c}_\lambda = [\lambda r_1 + (1-\lambda)r_2]\,\mathbf{P}_R + [\lambda g_1 + (1-\lambda)g_2]\,\mathbf{P}_G + [\lambda b_1 + (1-\lambda)b_2]\,\mathbf{P}_B$$

**Step 5:** Each coefficient $\lambda r_1 + (1-\lambda)r_2 \geq 0$ since $r_1, r_2 \geq 0$ and $\lambda \in [0,1]$. Similarly for $g$ and $b$.

**Step 6:** Therefore $\mathbf{c}_\lambda$ is a non-negative combination of the primaries, so $\mathbf{c}_\lambda$ is in the gamut.

**Step 7:** Since every convex combination of gamut colors is in the gamut, the gamut is convex.

**Result:** The RGB gamut is a convex cone in XYZ space. When normalized to the chromaticity plane ($x + y + z = 1$), it projects to a convex triangle (the gamut triangle).
$\blacksquare$

**Intuition:** Convexity means that mixing two representable colors always produces a representable color. Geometrically, the gamut in chromaticity space is the triangle with vertices at the three primary chromaticities. Any color outside this triangle cannot be displayed by the given set of primaries.

---

## 2. Algorithm / Method

### 2.1 Channel Separation and Recombination

```
Algorithm: RGB Channel Operations
Input: Image I ∈ ℝ^{M×N×3}
Output: Per-channel statistics and recombined image

1. SEPARATE: R = I[:,:,0], G = I[:,:,1], B = I[:,:,2]
2. STATISTICS: For each channel c ∈ {R, G, B}:
     μ_c = (1/MN) Σ_{m,n} c[m,n]
     σ_c = sqrt((1/MN) Σ_{m,n} (c[m,n] - μ_c)²)
3. NORMALIZE: c_norm = (c - μ_c) / σ_c
4. RECOMBINE: I_out = stack(R_out, G_out, B_out, axis=2)
```

### 2.2 Complexity Analysis

- **Channel separation/recombination:** $O(MNC)$ — linear scan
- **Per-channel statistics:** $O(MN)$ per channel
- **Color space conversion (matrix multiply per pixel):** $O(MN \cdot C^2)$ — typically $O(9MN)$ for $3 \times 3$ matrix
- **Space:** $O(MNC)$ for the image tensor

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

- **State:** $s_t = \mathbf{I}_t \in \mathbb{R}^{M \times N \times C}$ — the multi-channel image observation
- **Action:** $a_t \in \{\text{adjust\_R}, \text{adjust\_G}, \text{adjust\_B}, \text{swap\_channels}, \text{convert\_colorspace}, \ldots\}$
- **Reward:** $r_t = -\|\mathbf{I}_{t+1} - \mathbf{I}_{\text{target}}\|_F^2 + \|\mathbf{I}_t - \mathbf{I}_{\text{target}}\|_F^2$ (Frobenius norm improvement)
- **Transition:** Deterministic application of the selected channel manipulation

### 3.2 Why RL?

Channel manipulations are order-dependent (e.g., gamma correction before white balance vs. after). RL can learn optimal sequencing of per-channel operations that traditional fixed pipelines cannot adapt. The convexity of the RGB gamut guarantees that interpolating between agent-proposed color corrections stays within the representable gamut.

---

## 4. Dataset

- **Name:** scikit-image built-in images
- **Size:** Various (astronaut 512×512×3, coffee 400×600×3)
- **Auto-download:**

```python
from skimage import data
astronaut = data.astronaut()   # RGB, uint8
coffee = data.coffee()         # RGB, uint8
```

---

## 5. Key Equations Summary

| Equation | Meaning |
|----------|---------|
| $\mathbf{p}_{m,n} \in \{0,\ldots,255\}^C$ | Pixel as a $C$-dimensional vector |
| $V_{\text{sRGB}} = 1.055 \cdot L^{1/2.4} - 0.055$ | sRGB gamma encoding (high range) |
| $\mathbf{RGB}_{\text{lin}} = \mathbf{M}^{-1}_{XYZ \to sRGB} \cdot \mathbf{XYZ}$ | Color space conversion |
| $\mathbf{c}_\lambda = \lambda\mathbf{c}_1 + (1-\lambda)\mathbf{c}_2 \in \text{Gamut}$ | Gamut convexity |
| $X = \int \Phi(\lambda)\bar{x}(\lambda)\,d\lambda$ | CIE tristimulus value |

---

## 6. References

- Poynton, C. *Digital Video and HD: Algorithms and Interfaces*, 2nd ed., Morgan Kaufmann, 2012.
- Wyszecki, G. & Stiles, W. S. *Color Science*, 2nd ed., Wiley-Interscience, 2000.
- IEC 61966-2-1:1999 — sRGB standard specification.
- Stockman, A. & Sharpe, L. T. "The Spectral Sensitivities of the Middle- and Long-Wavelength-Sensitive Cones," *Vision Research*, 2000.
