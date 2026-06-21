![Module Logo](../logo.png)

# Image Inpainting Agent via Reinforcement Learning

## Overview

This module develops an RL agent for image inpainting that decides the fill order and source patch selection for completing missing image regions. The mathematical framework covers diffusion-based PDE inpainting, Markov Random Field energy minimization for texture synthesis, partial convolution operations, and formulates the sequential inpainting process as an MDP where the agent learns optimal filling strategies.

## Prerequisites

- Partial differential equations (diffusion, Laplace equation, Navier-Stokes)
- Markov Random Fields (Gibbs distributions, clique potentials)
- Convolution theory (masking, partial operations)
- Variational methods (total variation, Euler-Lagrange)
- Reinforcement learning (sequential decision-making, planning)

---

## 1. Mathematical Foundations

### 1.1 Diffusion Equation for Inpainting (PDE Derivation)

**Problem:** Given image $I$ with known region $\Omega_k$ and missing region $\Omega_m$ (with boundary $\partial\Omega_m$), find $u: \Omega_m \to \mathbb{R}$ that smoothly extends the image.

**Step 1 (Laplace equation — simplest model):** Find $u$ minimizing:

$$\min_u \int_{\Omega_m} |\nabla u|^2 \, dx$$

subject to $u|_{\partial\Omega_m} = I|_{\partial\Omega_m}$ (Dirichlet BC).

**Step 2:** The Euler-Lagrange equation yields:

$$\Delta u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0 \quad \text{in } \Omega_m$$

This produces smooth (harmonic) interpolation but blurs textures.

**Step 3 (Heat equation — iterative diffusion):** Solve via time evolution:

$$\frac{\partial u}{\partial t} = \Delta u, \quad u(x, 0) = I(x), \quad u|_{\partial\Omega_m} = I|_{\partial\Omega_m}$$

**Step 4 (Total Variation inpainting):** For edge-preserving inpainting:

$$\min_u \int_{\Omega_m} |\nabla u| \, dx \quad \text{s.t. } u|_{\partial\Omega_m} = I|_{\partial\Omega_m}$$

The Euler-Lagrange equation:

$$\text{div}\left(\frac{\nabla u}{|\nabla u|}\right) = 0$$

This is the curvature equation — level curves of $u$ have zero curvature (straight lines connecting boundaries).

**Step 5 (Navier-Stokes analogy — Bertalmio et al.):** Propagate isophotes (lines of equal intensity) into the missing region:

$$\frac{\partial I}{\partial t} = \nabla^\perp I \cdot \nabla(\Delta I)$$

where $\nabla^\perp I = (-I_y, I_x)$ is the isophote direction and $\Delta I$ is the smoothness (Laplacian). This transports information along isophote directions.

**Step 6 (Numerical scheme):** Discretize with upwind finite differences:

$$I^{n+1}(x) = I^n(x) + \Delta t \cdot \delta\vec{N} \cdot \nabla(\Delta I^n)$$

where $\vec{N}$ is the isophote unit normal.

### 1.2 Texture Synthesis: MRF Energy Minimization

**Step 1 (MRF model):** The image is modeled as a Markov Random Field where the value at pixel $x$ depends only on its neighborhood $\mathcal{N}(x)$:

$$P(I(x) | I(\Omega \setminus \{x\})) = P(I(x) | I(\mathcal{N}(x)))$$

**Step 2 (Gibbs distribution):**

$$P(I) = \frac{1}{Z}\exp(-E(I)/T)$$

$$E(I) = \sum_{c \in \mathcal{C}} V_c(I)$$

where $\mathcal{C}$ is the set of cliques and $V_c$ is the clique potential.

**Step 3 (Patch-based energy):** For texture inpainting, define the energy based on patch similarity:

$$E(I) = \sum_{x \in \Omega_m} \min_{y \in \Omega_k} d(P_x, P_y)$$

where $P_x$ is the patch centered at $x$ and $d$ is a patch distance metric:

$$d(P_x, P_y) = \sum_{(i,j)\in\text{patch}} m(i,j)\|I(x+i,j) - I(y+i,j)\|^2$$

with mask $m(i,j) = 1$ for known pixels in the patch.

**Step 4 (Optimization via EM-like algorithm):**
- E-step: Find best matching patch for each location
- M-step: Update pixel values by averaging overlapping patch contributions

### 1.3 Partial Convolution Mathematics

**Step 1:** Standard convolution with feature map $X$ and mask $M$ ($M=1$ for valid, $M=0$ for holes):

$$Y(x) = \sum_k W_k \cdot (X \odot M)(x + k)$$

**Step 2 (Partial convolution):** Normalize by the fraction of valid inputs:

$$Y(x) = \begin{cases} \frac{\sum_k W_k(X \odot M)(x+k)}{\sum_k M(x+k)} \cdot \frac{|\mathcal{K}|}{1} + b & \text{if } \sum_k M(x+k) > 0 \\ 0 & \text{otherwise} \end{cases}$$

where $|\mathcal{K}|$ is the kernel size.

**Step 3 (Mask update rule):** After each partial convolution layer, update the mask:

$$M'(x) = \begin{cases} 1 & \text{if } \sum_k M(x+k) > 0 \\ 0 & \text{otherwise} \end{cases}$$

**Step 4:** As depth increases, the valid region expands by the kernel radius each layer, eventually filling the entire mask.

**Step 5 (Property):** Partial convolution is unbiased — it produces the same output regardless of how many neighbors are masked, unlike zero-padding which biases toward zero.

### 1.4 RL Agent: Fill Order and Source Selection

**Step 1 (Priority-based fill order):** The classical Criminisi algorithm computes priority:

$$P(x) = C(x) \cdot D(x)$$

where $C(x) = \frac{|\mathcal{N}(x) \cap \Omega_k|}{|\mathcal{N}(x)|}$ is confidence and $D(x) = |\nabla I_p^{\perp} \cdot \mathbf{n}_x|$ is data term (isophote strength at boundary).

**Step 2 (RL reformulation):** Let the agent learn the priority function:

$$P_\theta(x) = \pi_\theta(x | \text{current\_image}, \text{mask})$$

**Step 3 (Source patch selection as action):**

$$a_t = (x_t^{fill}, y_t^{source}) \in \partial\Omega_m \times \Omega_k$$

The agent selects both where to fill and what to fill with.

---

## 2. Algorithm / Method

### 2.1 Pseudocode

```
Algorithm: RL Image Inpainting Agent
─────────────────────────────────────────
Input: Image I with mask M (M=0 for missing regions)
Output: Completed image I*

State: s = (I_current, M_current, boundary_features, confidence_map)
Action: a = (fill_location, source_patch_location, patch_size)

Initialize policy π_θ (fill order + source selection)

for episode = 1 to num_episodes do
    I₀, M₀ ← (masked_image, mask)
    for t = 0 to T (until mask fully filled) do
        // Select boundary pixel to fill and source patch
        (x_fill, x_source, size) ~ π_θ(·|sₜ)
        
        // Copy source patch to fill location
        I_{t+1} ← copy_patch(Iₜ, x_source, x_fill, size)
        M_{t+1} ← update_mask(Mₜ, x_fill, size)
        
        // Reward: local quality + global coherence
        rₜ ← SSIM_local + λ·perceptual_coherence - c
        s_{t+1} ← update_state(I_{t+1}, M_{t+1})
    end for
    Update π_θ via policy gradient
end for
```

### 2.2 Complexity Analysis

- **PDE inpainting:** $O(N \cdot T_{iter})$ per diffusion step
- **Patch matching (brute force):** $O(|\Omega_k| \cdot |\partial\Omega_m| \cdot p^2)$ for patch size $p$
- **Partial convolution (one layer):** $O(N \cdot C^2 \cdot K^2)$
- **RL agent decision:** $O(|\theta|)$ per fill step
- **Total inpainting:** $O(|\Omega_m|/p^2 \cdot |\Omega_k| \cdot p^2) = O(|\Omega_m| \cdot |\Omega_k|)$

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

- **State:** Current partially filled image + remaining mask + boundary analysis
- **Action:** (fill location on boundary, source patch in known region, patch size)
- **Reward:** Local SSIM at filled region + absence of visible seams + perceptual quality
- **Transition:** Deterministic patch copy operation
- **Terminal:** All missing pixels filled

### 3.2 Why RL?

1. **Fill order matters:** Incorrect order propagates errors; RL learns globally optimal ordering
2. **Context-dependent source selection:** The best source patch depends on what has been filled previously
3. **Multi-scale strategy:** Agent learns when to use large patches (uniform regions) vs. small patches (details)
4. **Beyond hand-crafted priorities:** Learned priority outperforms Criminisi's heuristic on complex scenes

---

## 4. Dataset

| Dataset | Size | Mask Types | Description |
|---------|------|-----------|-------------|
| Places2 | 10M | Irregular | Scene completion |
| CelebA-HQ | 30K | Center/random | Face inpainting |
| Paris StreetView | 14.9K | Center | Facade completion |
| DTD | 5,640 | Random | Texture completion |

---

## 5. Key Equations Summary

| Equation | Description |
|----------|-------------|
| $\Delta u = 0$ (in $\Omega_m$) | Harmonic inpainting |
| $\frac{\partial I}{\partial t} = \nabla^\perp I\cdot\nabla(\Delta I)$ | Navier-Stokes inpainting |
| $E(I) = \sum_{x\in\Omega_m}\min_{y\in\Omega_k}d(P_x,P_y)$ | MRF texture energy |
| $Y(x) = \frac{\sum_k W_k(X\odot M)}{sum_k M}\cdot|\mathcal{K}| + b$ | Partial convolution |
| $P(x) = C(x)\cdot D(x)$ | Criminisi priority |

---

## 6. References

1. Bertalmio, M., Sapiro, G., Caselles, V., & Ballester, C. (2000). Image inpainting. *SIGGRAPH*.
2. Criminisi, A., Pérez, P., & Toyama, K. (2004). Region filling and object removal by exemplar-based image inpainting. *IEEE TIP*, 13(9), 1200-1212.
3. Liu, G., Reda, F. A., Shih, K. J., et al. (2018). Image inpainting for irregular holes using partial convolutions. *ECCV*.
4. Yu, J., Lin, Z., Yang, J., et al. (2018). Generative image inpainting with contextual attention. *CVPR*.
