![Module Logo](../logo.png)

# Color Spaces

## Overview

Color spaces are coordinate systems for representing color, each designed to optimize a particular property: hardware compatibility (RGB), perceptual uniformity (CIELAB), or intuitive manipulation (HSV). This document provides complete mathematical derivations for the major color space conversions, including the geometric construction of the HSV hexcone model, the full RGB→XYZ→LAB pipeline with all transformation matrices, and a proof that CIELAB achieves approximate perceptual uniformity via the $\Delta E$ metric.

## Prerequisites

- Linear algebra (matrix transformations, change of basis)
- Trigonometry (atan2, polar coordinates)
- Module 01.1 and 01.2 (image formation, pixel channels)
- Basic color theory (trichromacy)

---

## 1. Mathematical Foundations

### 1.1 Core Definition — Color Space as a Coordinate System

A color space is a mapping $\phi: \mathcal{C} \to \mathbb{R}^3$ from the set of perceivable colors $\mathcal{C}$ to a three-dimensional coordinate system. Different color spaces are related by invertible transformations $\phi_2 = T \circ \phi_1$.

### 1.2 Full Derivation of RGB → HSV Conversion

The HSV (Hue, Saturation, Value) color space represents colors on a hexagonal cone (hexcone model).

**Step 1:** Normalize RGB values to $[0, 1]$: $R, G, B \in [0, 1]$.

**Step 2:** Compute the value (brightness) as the maximum channel:

$$V = \max(R, G, B)$$

**Step 3:** Define auxiliary variables:

$$C_{\max} = \max(R, G, B), \quad C_{\min} = \min(R, G, B), \quad \Delta = C_{\max} - C_{\min}$$

$\Delta$ is the chroma — the difference between the most and least dominant color channels.

**Step 4:** Derive the saturation. Saturation measures how far a color is from a neutral gray of the same brightness:

$$S = \begin{cases} 0 & \text{if } V = 0 \\ \frac{\Delta}{V} = \frac{C_{\max} - C_{\min}}{C_{\max}} & \text{if } V > 0 \end{cases}$$

**Geometric interpretation:** $S = 0$ corresponds to the achromatic axis (center of the hexcone); $S = 1$ corresponds to fully saturated colors on the surface.

**Step 5:** Derive the hue angle. The hue $H$ is computed by projecting the RGB cube onto a regular hexagon. Consider the hexagonal cross-section of the RGB cube perpendicular to the main diagonal $(1,1,1)$.

Define the intermediate variable based on which channel is maximum:

$$H' = \begin{cases} \frac{G - B}{\Delta} \mod 6 & \text{if } C_{\max} = R \\ \frac{B - R}{\Delta} + 2 & \text{if } C_{\max} = G \\ \frac{R - G}{\Delta} + 4 & \text{if } C_{\max} = B \end{cases}$$

$$H = 60° \times H'$$

**Step 6:** Verify the geometric construction. The six sectors of the hexagon correspond to:

| $H'$ range | Dominant | Rising/Falling |
|-----------|----------|----------------|
| $[0, 1)$ | R max | G rising |
| $[1, 2)$ | G max | R falling |
| $[2, 3)$ | G max | B rising |
| $[3, 4)$ | B max | G falling |
| $[4, 5)$ | B max | R rising |
| $[5, 6)$ | R max | B falling |

**Step 7:** Verify periodicity and boundary conditions. At $(R, G, B) = (1, 0, 0)$ (pure red):

$$\Delta = 1, \quad H' = \frac{0 - 0}{1} = 0, \quad H = 0°$$

At $(R, G, B) = (1, 1, 0)$ (yellow):

$$\Delta = 1, \quad H' = \frac{1 - 0}{1} = 1, \quad H = 60°$$

At $(R, G, B) = (0, 1, 0)$ (green): $H' = \frac{0 - 0}{1} + 2 = 2$, $H = 120°$. The hue traverses the full $360°$ cycle.

**Result:**

$$\boxed{\text{HSV} = \left(60° \times H', \; \frac{\Delta}{C_{\max}}, \; C_{\max}\right)}$$
$\blacksquare$

**Intuition:** HSV separates chromatic information (H, S) from achromatic information (V). The hexcone geometry arises because the RGB cube, viewed along its main diagonal, projects to a regular hexagon. Hue is the angular position on this hexagon, saturation is the radial distance from center, and value is the height along the diagonal.

### 1.3 Full Derivation of RGB → CIELAB via XYZ

The CIELAB (or $L^*a^*b^*$) color space is designed for perceptual uniformity. The conversion proceeds in two stages: RGB → XYZ → LAB.

#### Stage 1: Linear sRGB → XYZ

**Step 1:** Remove gamma encoding to obtain linear RGB (inverse of the sRGB transfer function):

$$C_{\text{linear}} = \begin{cases} C_{\text{sRGB}} / 12.92 & \text{if } C_{\text{sRGB}} \leq 0.04045 \\ \left(\frac{C_{\text{sRGB}} + 0.055}{1.055}\right)^{2.4} & \text{if } C_{\text{sRGB}} > 0.04045 \end{cases}$$

for each channel $C \in \{R, G, B\}$.

**Step 2:** Apply the linear transformation to CIE XYZ:

$$\begin{pmatrix} X \\ Y \\ Z \end{pmatrix} = \begin{pmatrix} 0.4124564 & 0.3575761 & 0.1804375 \\ 0.2126729 & 0.7151522 & 0.0721750 \\ 0.0193339 & 0.1191920 & 0.9503041 \end{pmatrix} \begin{pmatrix} R_{\text{lin}} \\ G_{\text{lin}} \\ B_{\text{lin}} \end{pmatrix}$$

**Step 3:** Note that $Y = 0.2126R + 0.7152G + 0.0722B$ corresponds to the luminous efficiency function — the second row encodes the photopic luminosity coefficients.

#### Stage 2: XYZ → CIELAB

**Step 4:** Normalize by the reference white point $D65$: $(X_n, Y_n, Z_n) = (0.95047, 1.00000, 1.08883)$.

**Step 5:** Apply the nonlinear compression function $f$:

$$f(t) = \begin{cases} t^{1/3} & \text{if } t > \delta^3 \\ \frac{t}{3\delta^2} + \frac{4}{29} & \text{if } t \leq \delta^3 \end{cases}$$

where $\delta = 6/29 \approx 0.20690$.

**Step 6:** Verify the continuity of $f$ at $t = \delta^3$. At $t = \delta^3 = (6/29)^3$:

- Cube root branch: $(\delta^3)^{1/3} = \delta = 6/29$
- Linear branch: $\frac{\delta^3}{3\delta^2} + \frac{4}{29} = \frac{\delta}{3} + \frac{4}{29} = \frac{6}{87} + \frac{4}{29} = \frac{6}{87} + \frac{12}{87} = \frac{18}{87} = \frac{6}{29}$

Both branches yield $6/29$. $\checkmark$

**Step 7:** Verify the derivative continuity at $t = \delta^3$:

- Cube root derivative: $\frac{1}{3}t^{-2/3}\big|_{t=\delta^3} = \frac{1}{3\delta^2}$
- Linear derivative: $\frac{1}{3\delta^2}$

Derivatives match, confirming $C^1$ continuity. $\checkmark$

**Step 8:** Compute the CIELAB coordinates:

$$L^* = 116 \, f\!\left(\frac{Y}{Y_n}\right) - 16$$

$$a^* = 500 \left[f\!\left(\frac{X}{X_n}\right) - f\!\left(\frac{Y}{Y_n}\right)\right]$$

$$b^* = 200 \left[f\!\left(\frac{Y}{Y_n}\right) - f\!\left(\frac{Z}{Z_n}\right)\right]$$

**Step 9:** Verify the range. $L^*$ ranges from $0$ (black) to $100$ (white). For the white point $(X_n, Y_n, Z_n)$: $f(1) = 1$, so $L^* = 116(1) - 16 = 100$, $a^* = 0$, $b^* = 0$. $\checkmark$

**Result:**

$$\boxed{L^* = 116 f(Y/Y_n) - 16, \quad a^* = 500[f(X/X_n) - f(Y/Y_n)], \quad b^* = 200[f(Y/Y_n) - f(Z/Z_n)]}$$
$\blacksquare$

### 1.4 Proof of Perceptual Uniformity — The $\Delta E$ Derivation

**Theorem:** The Euclidean distance in CIELAB space approximates perceived color difference.

**Definition:** The CIE76 color difference between two colors $(L_1^*, a_1^*, b_1^*)$ and $(L_2^*, a_2^*, b_2^*)$ is:

$$\Delta E_{76}^* = \sqrt{(L_2^* - L_1^*)^2 + (a_2^* - a_1^*)^2 + (b_2^* - b_1^*)^2}$$

**Proof of approximate perceptual uniformity:**

**Step 1:** The cube-root function $f(t) = t^{1/3}$ models Stevens' power law for brightness perception. Stevens' psychophysical law states that perceived magnitude $\psi$ of a stimulus of intensity $I$ follows:

$$\psi = k \cdot I^n$$

For brightness, $n \approx 1/3$, hence the cube-root compression.

**Step 2:** The scaling factors $(116, 500, 200)$ are calibrated so that one unit of $\Delta E$ corresponds to approximately one just-noticeable difference (JND) under standard viewing conditions.

**Step 3:** Define the CIELAB metric tensor. In XYZ space, the perceptual metric is non-Euclidean. The transformation to LAB is designed so that:

$$ds^2_{\text{perceptual}} \approx dL^{*2} + da^{*2} + db^{*2}$$

The Jacobian of the XYZ → LAB transformation $\mathbf{J}$ satisfies:

$$\mathbf{G}_{\text{LAB}} = \mathbf{J}^{-T} \mathbf{G}_{\text{XYZ}} \mathbf{J}^{-1} \approx \mathbf{I}_3$$

where $\mathbf{G}_{\text{XYZ}}$ is the perceptual metric tensor in XYZ coordinates (derived from MacAdam ellipses) and $\mathbf{I}_3$ is the identity.

**Step 4:** Empirical validation: MacAdam ellipses (regions of indistinguishable colors in chromaticity space) become approximately circular and equal-sized in CIELAB, confirming that Euclidean distance approximates perceptual difference.

**Step 5:** Perceptual thresholds in CIELAB:

| $\Delta E^*$ | Perception |
|-------------|------------|
| $< 1$ | Imperceptible |
| $1$–$2$ | Barely perceptible |
| $2$–$3.5$ | Noticeable |
| $3.5$–$5$ | Clearly noticeable |
| $> 5$ | Different color |

$\blacksquare$

**Intuition:** CIELAB "warps" the XYZ space so that equal Euclidean distances correspond to equal perceptual differences. The cube-root compression models the nonlinear response of the human visual system, while the scaling constants align the metric with empirical JND data.

### 1.5 CIEDE2000 — Refined Perceptual Distance

The CIEDE2000 formula refines $\Delta E_{76}^*$ with corrections for lightness, chroma, and hue weighting:

$$\Delta E_{00}^* = \sqrt{\left(\frac{\Delta L'}{k_L S_L}\right)^2 + \left(\frac{\Delta C'}{k_C S_C}\right)^2 + \left(\frac{\Delta H'}{k_H S_H}\right)^2 + R_T \frac{\Delta C'}{k_C S_C}\frac{\Delta H'}{k_H S_H}}$$

where $S_L, S_C, S_H$ are weighting functions and $R_T$ is a rotation term correcting for the interaction between chroma and hue differences in the blue region.

---

## 2. Algorithm / Method

### 2.1 Pseudocode: RGB → CIELAB Conversion

```
Algorithm: RGB_to_CIELAB
Input: (R, G, B) in [0, 255]
Output: (L*, a*, b*)

1. LINEARIZE: For each C ∈ {R, G, B}:
     c = C / 255
     if c ≤ 0.04045: c_lin = c / 12.92
     else: c_lin = ((c + 0.055) / 1.055)^2.4
2. XYZ TRANSFORM:
     [X, Y, Z]ᵀ = M_sRGB_to_XYZ · [R_lin, G_lin, B_lin]ᵀ
3. NORMALIZE: x = X/X_n, y = Y/Y_n, z = Z/Z_n
4. COMPRESS: Apply f(t) to each of x, y, z
5. COMPUTE LAB:
     L* = 116·f(y) − 16
     a* = 500·(f(x) − f(y))
     b* = 200·(f(y) − f(z))
6. Return (L*, a*, b*)
```

### 2.2 Complexity Analysis

- **Time:** $O(1)$ per pixel (fixed number of arithmetic operations), $O(MN)$ for entire image
- **Space:** $O(1)$ auxiliary per pixel, $O(MN)$ for output

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

- **State:** $s_t = \mathbf{I}_t$ in a chosen color space (RGB, HSV, or LAB)
- **Action:** Color correction operations — e.g., shift hue by $\delta H$, adjust saturation by factor $\alpha$
- **Reward:** $r_t = -\Delta E^*(\mathbf{I}_{t+1}, \mathbf{I}_{\text{target}})$ — negative perceptual distance to target
- **Transition:** Deterministic color transformation

### 3.2 Why RL?

Perceptual uniformity of CIELAB makes it ideal for RL reward design: equal $\Delta E$ values represent equal perceptual improvements regardless of color region. HSV decouples hue, saturation, and value, enabling more interpretable action spaces. An RL agent can learn to sequentially correct white balance, saturation, and contrast in the optimal order.

---

## 4. Dataset

- **Name:** scikit-image built-in images
- **Size:** Various standard test images
- **Auto-download:**

```python
from skimage import data, color
rgb_img = data.astronaut()
lab_img = color.rgb2lab(rgb_img)
hsv_img = color.rgb2hsv(rgb_img)
```

---

## 5. Key Equations Summary

| Equation | Meaning |
|----------|---------|
| $H = 60° \times H'$ where $H'$ is piecewise on max channel | RGB to HSV hue |
| $S = \Delta / C_{\max}$ | HSV saturation |
| $L^* = 116\,f(Y/Y_n) - 16$ | CIELAB lightness |
| $\Delta E^* = \sqrt{\Delta L^{*2} + \Delta a^{*2} + \Delta b^{*2}}$ | CIE76 color difference |
| $f(t) = t^{1/3}$ for $t > \delta^3$ | Cube-root perceptual compression |
| $\mathbf{XYZ} = \mathbf{M} \cdot \mathbf{RGB}_{\text{lin}}$ | Linear color space transformation |

---

## 6. References

- CIE 15:2004 — *Colorimetry*, 3rd edition, Commission Internationale de l'Éclairage.
- Sharma, G., Wu, W., & Dalal, E. N. "The CIEDE2000 Color-Difference Formula," *Color Research & Application*, 30(1):21–30, 2005.
- Smith, A. R. "Color Gamut Transform Pairs," *SIGGRAPH*, 1978.
- Fairchild, M. D. *Color Appearance Models*, 3rd ed., Wiley, 2013.
