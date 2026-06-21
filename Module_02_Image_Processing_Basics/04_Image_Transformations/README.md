![Module Logo](../logo.png)

# Image Transformations

## Overview

Geometric transformations map pixel coordinates from one image to another, enabling rotation, scaling, shearing, and perspective correction. This document proves that affine transformations form a group under composition, derives the full theory of homogeneous coordinates, proves error bounds for bilinear interpolation, and establishes the connection between transformation groups and RL action spaces.

## Prerequisites

- Linear algebra (matrix groups, determinants)
- Group theory basics (closure, identity, inverse)
- Calculus (Taylor series for error analysis)
- Module 01 (image as matrix)

---

## 1. Mathematical Foundations

### 1.1 Core Definition — Affine Transformation

An affine transformation maps a point $\mathbf{x} = (x, y)^T$ to $\mathbf{x}' = (x', y')^T$ via:

$$\mathbf{x}' = \mathbf{A}\mathbf{x} + \mathbf{t}$$

where $\mathbf{A} \in \mathbb{R}^{2 \times 2}$ is a linear transformation matrix and $\mathbf{t} \in \mathbb{R}^2$ is a translation vector. The six parameters $(a_{11}, a_{12}, a_{21}, a_{22}, t_x, t_y)$ define the transformation.

### 1.2 Derivation of Homogeneous Coordinates

**Problem:** Translations are not linear transformations ($\mathbf{x}' = \mathbf{x} + \mathbf{t}$ cannot be written as $\mathbf{x}' = \mathbf{M}\mathbf{x}$ for any $2 \times 2$ matrix $\mathbf{M}$).

**Solution:** Embed $\mathbb{R}^2$ into $\mathbb{R}^3$ using homogeneous coordinates.

**Step 1:** Represent the 2D point $(x, y)$ as the 3D vector $\tilde{\mathbf{x}} = (x, y, 1)^T$.

**Step 2:** The affine transformation becomes a matrix multiplication:

$$\tilde{\mathbf{x}}' = \begin{pmatrix} a_{11} & a_{12} & t_x \\ a_{21} & a_{22} & t_y \\ 0 & 0 & 1 \end{pmatrix} \tilde{\mathbf{x}} = \mathbf{M} \tilde{\mathbf{x}}$$

**Step 3:** Recover the Euclidean point: $(x', y') = (x'_h / w, y'_h / w)$ where $\tilde{\mathbf{x}}' = (x'_h, y'_h, w)^T$.

**Step 4:** For affine transformations, $w = 1$ always, so no division is needed.

**Step 5:** For projective (perspective) transformations, $w \neq 1$:

$$\mathbf{H} = \begin{pmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{pmatrix}$$

and $(x', y') = (x'_h/w, y'_h/w)$ — this is a true projective transformation (homography).

### 1.3 Proof That Affine Transformations Form a Group

**Theorem:** The set of affine transformations $\text{Aff}(2)$ with composition forms a group.

**Proof:** We verify the four group axioms using the $3 \times 3$ homogeneous matrix representation.

**Step 1 — Closure:** Let $\mathbf{M}_1, \mathbf{M}_2 \in \text{Aff}(2)$:

$$\mathbf{M}_1 = \begin{pmatrix} \mathbf{A}_1 & \mathbf{t}_1 \\ \mathbf{0}^T & 1 \end{pmatrix}, \quad \mathbf{M}_2 = \begin{pmatrix} \mathbf{A}_2 & \mathbf{t}_2 \\ \mathbf{0}^T & 1 \end{pmatrix}$$

$$\mathbf{M}_1 \mathbf{M}_2 = \begin{pmatrix} \mathbf{A}_1\mathbf{A}_2 & \mathbf{A}_1\mathbf{t}_2 + \mathbf{t}_1 \\ \mathbf{0}^T & 1 \end{pmatrix}$$

The result has the same structure (last row is $(0, 0, 1)$), so $\mathbf{M}_1\mathbf{M}_2 \in \text{Aff}(2)$. $\checkmark$

**Step 2 — Associativity:** Matrix multiplication is associative. $\checkmark$

**Step 3 — Identity:** $\mathbf{I}_3 = \begin{pmatrix} \mathbf{I}_2 & \mathbf{0} \\ \mathbf{0}^T & 1 \end{pmatrix} \in \text{Aff}(2)$. $\checkmark$

**Step 4 — Inverse:** For $\mathbf{M}$ with $\det(\mathbf{A}) \neq 0$:

$$\mathbf{M}^{-1} = \begin{pmatrix} \mathbf{A}^{-1} & -\mathbf{A}^{-1}\mathbf{t} \\ \mathbf{0}^T & 1 \end{pmatrix}$$

Verify: $\mathbf{M}\mathbf{M}^{-1} = \begin{pmatrix} \mathbf{A}\mathbf{A}^{-1} & -\mathbf{A}\mathbf{A}^{-1}\mathbf{t} + \mathbf{t} \\ \mathbf{0}^T & 1 \end{pmatrix} = \mathbf{I}_3$. $\checkmark$

**Result:** $(\text{Aff}(2), \circ)$ is a group. Moreover, it is a Lie group of dimension 6.
$\blacksquare$

**Subgroup hierarchy:**

$$\text{Translations} \subset \text{Rigid (Euclidean)} \subset \text{Similarity} \subset \text{Affine} \subset \text{Projective}$$

| Group | DOF | Preserves |
|-------|-----|-----------|
| Translation | 2 | Orientation, lengths, angles |
| Euclidean | 3 | Lengths, angles |
| Similarity | 4 | Angles, ratios |
| Affine | 6 | Parallelism, ratios on lines |
| Projective | 8 | Cross-ratio, collinearity |

### 1.4 Bilinear Interpolation and Error Bound Proof

When the mapped coordinates $(x', y')$ are non-integer, we must interpolate.

**Step 1:** Let $f$ be the image function. For non-integer position $(x, y)$, let $x_0 = \lfloor x \rfloor$, $y_0 = \lfloor y \rfloor$, $\alpha = x - x_0$, $\beta = y - y_0$.

**Step 2:** Bilinear interpolation:

$$\hat{f}(x, y) = (1-\alpha)(1-\beta)f_{00} + \alpha(1-\beta)f_{10} + (1-\alpha)\beta f_{01} + \alpha\beta f_{11}$$

where $f_{ij} = f(x_0 + i, y_0 + j)$.

**Step 3:** Verify that this is exact at the four corners:
- At $(\alpha, \beta) = (0, 0)$: $\hat{f} = f_{00}$. $\checkmark$
- At $(\alpha, \beta) = (1, 0)$: $\hat{f} = f_{10}$. $\checkmark$
- At $(\alpha, \beta) = (0, 1)$: $\hat{f} = f_{01}$. $\checkmark$
- At $(\alpha, \beta) = (1, 1)$: $\hat{f} = f_{11}$. $\checkmark$

**Step 4 — Error bound.** Assume $f \in C^2$ (twice continuously differentiable). Taylor expand around $(x_0, y_0)$:

$$f(x_0 + \alpha, y_0 + \beta) = f_{00} + \alpha f_x + \beta f_y + \frac{1}{2}(\alpha^2 f_{xx} + 2\alpha\beta f_{xy} + \beta^2 f_{yy}) + O(h^3)$$

where all derivatives are evaluated at $(x_0, y_0)$ and $h = 1$ (pixel spacing).

**Step 5:** The bilinear interpolant uses only the function values at four corners. By expanding each corner value in Taylor series and substituting into the bilinear formula, one can show:

$$\hat{f}(x, y) = f(x, y) - \alpha\beta(1-\alpha)(f_{xy}) h^2 + O(h^3)$$

The dominant error term involves the mixed partial derivative $f_{xy}$.

**Step 6:** The maximum interpolation error over the unit cell is:

$$|f(x, y) - \hat{f}(x, y)| \leq \frac{h^2}{4} \max|f_{xy}| + O(h^3)$$

The $1/4$ factor comes from maximizing $\alpha(1-\alpha)\beta(1-\beta)$ over $[0,1]^2$, which achieves its maximum of $1/16$ at $\alpha = \beta = 1/2$. The bound tightens to:

$$\boxed{|e_{\text{bilinear}}| \leq \frac{h^2}{8}\left(\max|f_{xx}| + \max|f_{yy}|\right) + \frac{h^2}{4}\max|f_{xy}|}$$
$\blacksquare$

**Intuition:** Bilinear interpolation is second-order accurate ($O(h^2)$). The error is largest where the image has strong diagonal features (large $f_{xy}$). For smooth images, the error is small; for images with sharp edges, it can produce visible blurring.

---

## 2. Algorithm / Method

### 2.1 Pseudocode: Affine Warp with Bilinear Interpolation

```
Algorithm: Affine_Warp
Input: Image I, affine matrix M (3×3)
Output: Warped image I'

1. INVERT: M_inv = inverse(M)
2. For each output pixel (m', n'):
     [x, y, 1]ᵀ = M_inv · [m', n', 1]ᵀ
     x0, y0 = floor(x), floor(y)
     α = x - x0, β = y - y0
     I'[m', n'] = (1-α)(1-β)·I[x0,y0] + α(1-β)·I[x0+1,y0]
                  + (1-α)β·I[x0,y0+1] + αβ·I[x0+1,y0+1]
3. Return I'
```

### 2.2 Complexity Analysis

- **Time:** $O(M'N')$ — one inverse mapping + interpolation per output pixel
- **Space:** $O(M'N')$ for the output image
- **Matrix inverse:** $O(1)$ for $3 \times 3$ matrix (constant-time formula)

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

- **State:** $s_t = (\mathbf{I}_t, \mathbf{M}_t)$ — current image and accumulated transformation
- **Action:** $a_t \in \{\text{rotate}(\theta), \text{scale}(s), \text{translate}(t_x, t_y), \text{shear}(\lambda)\}$
- **Reward:** $r_t = \text{alignment\_score}(\mathbf{I}_{t+1}, \mathbf{I}_{\text{ref}})$ (e.g., normalized cross-correlation)
- **Transition:** $\mathbf{M}_{t+1} = \mathbf{M}_{a_t} \cdot \mathbf{M}_t$ — compose transformations

### 3.2 Why RL?

The group structure of affine transformations means sequential actions compose via matrix multiplication. RL naturally handles this sequential decision process, learning to align images through iterative refinement rather than one-shot estimation.

---

## 4. Dataset

- **Name:** scikit-image built-in images + synthetic transformations
- **Size:** Standard test images
- **Auto-download:**

```python
from skimage import data, transform
camera = data.camera()
rotated = transform.rotate(camera, angle=30)
```

---

## 5. Key Equations Summary

| Equation | Meaning |
|----------|---------|
| $\tilde{\mathbf{x}}' = \mathbf{M}\tilde{\mathbf{x}}$ | Affine transform in homogeneous coords |
| $\det(\mathbf{A}) \neq 0 \implies \mathbf{M}^{-1}$ exists | Invertibility condition |
| $\mathbf{M}_1\mathbf{M}_2 \in \text{Aff}(2)$ | Group closure |
| $\hat{f} = (1-\alpha)(1-\beta)f_{00} + \cdots$ | Bilinear interpolation |
| $\|e\| \leq O(h^2 \max\|f_{xy}\|)$ | Bilinear interpolation error bound |

---

## 6. References

- Hartley, R. & Zisserman, A. *Multiple View Geometry in Computer Vision*, 2nd ed., Cambridge, 2004.
- Szeliski, R. *Computer Vision: Algorithms and Applications*, 2nd ed., Springer, 2022.
- Gonzalez, R. C. & Woods, R. E. *Digital Image Processing*, 4th ed., Pearson, 2018, Ch. 2.
- Keys, R. "Cubic Convolution Interpolation," *IEEE Trans. ASSP*, 29(6):1153–1160, 1981.
