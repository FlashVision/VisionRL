![Module Logo](../logo.png)

# Color Correction via Reinforcement Learning

## Overview

This module addresses the color correction problem through reinforcement learning, where an agent learns to apply optimal color transformations to achieve perceptually accurate color reproduction. The approach combines chromatic adaptation theory, color constancy algorithms, and sequential decision-making to produce an adaptive color correction pipeline.

## Prerequisites

- Linear algebra (eigenvalue decomposition, matrix transformations)
- Color science (CIE color spaces, illumination models)
- Optimization theory (convex optimization basics)
- Reinforcement learning (policy gradient methods)
- Statistics (maximum likelihood estimation)

---

## 1. Mathematical Foundations

### 1.1 Chromatic Adaptation: Von Kries Model

**Core Definition:** Chromatic adaptation models how the human visual system adjusts to the color of illumination. The Von Kries model assumes independent gain control in cone response channels.

Let $\rho = (\rho_L, \rho_M, \rho_S)^T$ be the cone responses under illuminant $E_1$, and let $\rho' = (\rho_L', \rho_M', \rho_S')^T$ be the adapted responses under illuminant $E_2$.

**Step 1:** Transform from XYZ to cone space using the transformation matrix $M$:

$$\begin{pmatrix} L \\ M \\ S \end{pmatrix} = \mathbf{M} \begin{pmatrix} X \\ Y \\ Z \end{pmatrix}$$

For the Bradford transform:

$$\mathbf{M}_{Bradford} = \begin{pmatrix} 0.8951 & 0.2664 & -0.1614 \\ -0.7502 & 1.7135 & 0.0367 \\ 0.0389 & -0.0685 & 1.0296 \end{pmatrix}$$

**Step 2:** Compute the diagonal adaptation matrix. Given source white point $W_s = (L_s, M_s, S_s)^T$ and destination white point $W_d = (L_d, M_d, S_d)^T$:

$$\mathbf{D} = \begin{pmatrix} L_d/L_s & 0 & 0 \\ 0 & M_d/M_s & 0 \\ 0 & 0 & S_d/S_s \end{pmatrix}$$

**Step 3:** The complete chromatic adaptation transform (CAT) in XYZ space:

$$\mathbf{T}_{CAT} = \mathbf{M}^{-1} \cdot \mathbf{D} \cdot \mathbf{M}$$

**Step 4:** Apply to any color:

$$(X_d, Y_d, Z_d)^T = \mathbf{T}_{CAT} \cdot (X_s, Y_s, Z_s)^T$$

**Step 5:** The adapted color preserves the appearance under the new illuminant by compensating for the spectral shift.

### 1.2 Color Constancy Problem

**Definition:** Given an observed image $I(x, y)$ formed under unknown illuminant $L(\lambda)$ with surface reflectance $S(x, y, \lambda)$ and camera sensitivity $C_k(\lambda)$:

$$I_k(x, y) = \int_\lambda L(\lambda) S(x, y, \lambda) C_k(\lambda) \, d\lambda, \quad k \in \{R, G, B\}$$

**Step 1 (Lambertian assumption):** Under the simplified diagonal model:

$$I_k(x, y) = e_k \cdot \rho_k(x, y)$$

where $e_k$ is the illuminant color and $\rho_k$ is the surface reflectance contribution.

**Step 2 (Gray-World hypothesis):** Assume the average reflectance is achromatic:

$$\hat{e}_k = \frac{1}{N}\sum_{x,y} I_k(x, y)$$

**Step 3 (Generalized Gray-World — Minkowski norm):**

$$\hat{e}_k = \left(\frac{1}{N}\sum_{x,y} |I_k(x,y)|^p\right)^{1/p}$$

Setting $p = 1$ gives Gray-World, $p \to \infty$ gives Max-RGB, and the Shades-of-Gray method uses intermediate $p$.

**Step 4 (Gray-Edge hypothesis):** The average edge in a scene is achromatic:

$$\hat{e}_k = \left(\frac{1}{N}\sum_{x,y} \left|\frac{\partial^n I_k(x,y)}{\partial x^n}\right|^p\right)^{1/p}$$

**Step 5:** The corrected image is:

$$I_k^{corrected}(x, y) = \frac{I_k(x, y)}{\hat{e}_k} \cdot e_k^{target}$$

### 1.3 Color Difference Metrics: CIELAB $\Delta E$

**Step 1:** Convert from XYZ to CIELAB. Define $f(t)$:

$$f(t) = \begin{cases} t^{1/3} & \text{if } t > \delta^3 \\ \frac{t}{3\delta^2} + \frac{4}{29} & \text{otherwise} \end{cases}$$

where $\delta = 6/29$.

**Step 2:** Compute CIELAB coordinates:

$$L^* = 116 \cdot f(Y/Y_n) - 16$$
$$a^* = 500 \cdot [f(X/X_n) - f(Y/Y_n)]$$
$$b^* = 200 \cdot [f(Y/Y_n) - f(Z/Z_n)]$$

**Step 3:** The CIE76 color difference:

$$\Delta E_{76}^* = \sqrt{(\Delta L^*)^2 + (\Delta a^*)^2 + (\Delta b^*)^2}$$

**Step 4:** CIE94 introduces weighting:

$$\Delta E_{94}^* = \sqrt{\left(\frac{\Delta L^*}{k_L S_L}\right)^2 + \left(\frac{\Delta C^*}{k_C S_C}\right)^2 + \left(\frac{\Delta H^*}{k_H S_H}\right)^2}$$

where $S_L = 1$, $S_C = 1 + 0.045 C_1^*$, $S_H = 1 + 0.015 C_1^*$.

**Step 5:** CIEDE2000 adds rotation term for blue region:

$$\Delta E_{00}^* = \sqrt{\left(\frac{\Delta L'}{k_L S_L}\right)^2 + \left(\frac{\Delta C'}{k_C S_C}\right)^2 + \left(\frac{\Delta H'}{k_H S_H}\right)^2 + R_T \frac{\Delta C'}{k_C S_C}\frac{\Delta H'}{k_H S_H}}$$

### 1.4 Proof: Von Kries Adaptation Preserves Relative Cone Ratios

**Theorem:** Under Von Kries adaptation, the ratio of cone signals between any two surfaces is preserved across illuminant changes.

**Proof:** Let surfaces $A$ and $B$ have cone responses $(L_A, M_A, S_A)$ and $(L_B, M_B, S_B)$ under illuminant 1. Under illuminant 2, the Von Kries transform gives:

$$L_A' = \frac{L_d}{L_s} L_A, \quad L_B' = \frac{L_d}{L_s} L_B$$

Therefore:

$$\frac{L_A'}{L_B'} = \frac{\frac{L_d}{L_s} L_A}{\frac{L_d}{L_s} L_B} = \frac{L_A}{L_B}$$

The same holds for $M$ and $S$ channels. Thus relative cone ratios are invariant under diagonal adaptation. $\blacksquare$

---

## 2. Algorithm / Method

### 2.1 Pseudocode

```
Algorithm: RL Color Correction Agent
─────────────────────────────────────────
Input: Image I with unknown illuminant
Output: Color-corrected image I*

State: s = (color_histogram_R, color_histogram_G, color_histogram_B,
            spatial_color_moments, white_point_estimate)
Action: a = 3×3 color correction matrix entries (9 continuous values)

Initialize policy π_θ (outputs 3×3 matrix parameters)
Initialize critic V_φ

for episode = 1 to M do
    s₀ ← extract_color_features(I)
    for t = 0 to T_max do
        Aₜ ~ π_θ(·|sₜ)           // Sample 3×3 correction matrix
        Aₜ ← project_valid(Aₜ)    // Ensure valid color transform
        I_{t+1} ← Aₜ · I_t        // Apply matrix to each pixel
        rₜ ← -ΔE(I_{t+1}, I_ref)  // Negative color difference
        s_{t+1} ← extract_color_features(I_{t+1})
    end for
    Update π_θ, V_φ via PPO
end for
```

### 2.2 Complexity Analysis

- **Color feature extraction:** $O(HW \cdot C + B)$ where $B$ is histogram bins
- **Matrix application:** $O(HW \cdot C^2)$ = $O(9HW)$ for 3×3 color matrix
- **$\Delta E$ computation:** $O(HW)$ per pixel (XYZ→Lab transform)
- **Policy network forward pass:** $O(|\theta|)$
- **Total per step:** $O(HW \cdot C^2)$ dominated by matrix application

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

- **State** $\mathcal{S}$: Color histogram (256×3), spatial color moments (mean, var, skew per channel), estimated white point
- **Action** $\mathcal{A}$: Elements of a 3×3 color correction matrix $\mathbf{A} \in \mathbb{R}^{3\times 3}$, constrained to preserve non-negativity
- **Transition:** $P(s'|s, a)$: deterministic — apply matrix $\mathbf{A}$ pixel-wise
- **Reward:** $R(s, a) = -\frac{1}{N}\sum_{i=1}^N \Delta E_{00}(p_i^{corrected}, p_i^{reference})$
- **Discount:** $\gamma = 0.97$

### 3.2 Why RL?

1. **Illuminant ambiguity:** Multiple plausible illuminants exist; RL explores the correction space
2. **Sequential refinement:** Coarse-to-fine color correction mirrors how human editors work
3. **Non-linear color spaces:** Perceptual color differences ($\Delta E$) are non-linear, making direct optimization difficult
4. **Scene-dependent adaptation:** The agent learns scene-specific correction strategies

---

## 4. Dataset

| Dataset | Size | Description |
|---------|------|-------------|
| ColorChecker | 568 images | Calibrated with Macbeth chart |
| Gehler-Shi | 568 images | Ground truth illuminants |
| NUS Multi-Cam | 1,736 images | 8 cameras, diverse illuminants |
| Cube+ | 1,707 images | SpyderCube reference |

---

## 5. Key Equations Summary

| Equation | Description |
|----------|-------------|
| $\mathbf{T}_{CAT} = \mathbf{M}^{-1}\mathbf{D}\mathbf{M}$ | Chromatic adaptation transform |
| $I_k(x,y) = e_k \cdot \rho_k(x,y)$ | Diagonal illumination model |
| $\Delta E_{00}^*$ | CIEDE2000 color difference |
| $L^* = 116f(Y/Y_n) - 16$ | CIELAB lightness |
| $\hat{e}_k = (\frac{1}{N}\sum|I_k|^p)^{1/p}$ | Generalized Gray-World |

---

## 6. References

1. Von Kries, J. (1902). Chromatic Adaptation. *Festschrift der Albrecht-Ludwigs-Universität*.
2. Finlayson, G. D., & Trezzi, E. (2004). Shades of gray and colour constancy. *Color and Imaging Conference*.
3. Luo, M. R., Cui, G., & Rigg, B. (2001). The development of the CIE 2000 colour-difference formula: CIEDE2000. *Color Research & Application*, 26(5), 340-350.
4. Bianco, S., Cusano, C., & Schettini, R. (2017). Color constancy using CNNs. *CVPR Workshops*.
5. Afifi, M., & Brown, M. S. (2019). What else can fool deep learning? *ICCV*.
