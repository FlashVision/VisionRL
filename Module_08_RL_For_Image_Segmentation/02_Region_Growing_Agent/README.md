![Module Logo](../logo.png)

# Region Growing Agent for Image Segmentation

## Overview

This module develops an RL agent that performs image segmentation via region growing, deciding at each step whether to include neighboring pixels into a growing region. The mathematical framework connects region growing to graph-based optimization, the Mumford-Shah functional for piecewise smooth segmentation, and level set methods that evolve region boundaries via partial differential equations.

## Prerequisites

- Graph theory (connectivity, minimum cuts, graph search)
- Variational calculus (Euler-Lagrange equations, functionals)
- Partial differential equations (Hamilton-Jacobi, level sets)
- Topology (connected components, Jordan curve theorem)
- Reinforcement learning (Q-learning, DQN)

---

## 1. Mathematical Foundations

### 1.1 Region Growing as Graph Search

**Definition:** Model the image as a weighted graph $G = (V, E, w)$ where:
- $V = \{(i,j) : 1 \leq i \leq H, 1 \leq j \leq W\}$ (pixels as vertices)
- $E = \{((i,j),(i',j')) : |(i-i')| + |(j-j')| \leq 1\}$ (4-connectivity)
- $w(e) = |I(v_1) - I(v_2)|$ for edge $e = (v_1, v_2)$ (intensity difference)

**Optimality Condition:** A region $R \subseteq V$ is optimally segmented if:

$$\max_{v \in R, u \notin R, (v,u)\in E} w(v,u) > \min_{v,u \in R, (v,u) \in E} w(v,u)$$

This states that the weakest internal connection is stronger than the strongest boundary connection.

**Step 1 (Predicate):** Define the growing predicate $P(R, v)$ for adding vertex $v$ to region $R$:

$$P(R, v) = \mathbb{1}\left[|I(v) - \mu_R| < \tau \cdot \sigma_R\right]$$

where $\mu_R = \frac{1}{|R|}\sum_{u\in R} I(u)$ and $\sigma_R$ is the region standard deviation.

**Step 2 (Priority ordering):** Grow using a priority queue with priority:

$$\text{priority}(v) = -|I(v) - \mu_R|$$

**Step 3 (Stopping criterion):** Stop when no neighbor satisfies the predicate, or equivalently when the minimum boundary gradient exceeds the threshold.

### 1.2 Mumford-Shah Functional

**Definition:** The Mumford-Shah energy functional for piecewise smooth image segmentation:

$$E(u, \Gamma) = \underbrace{\int_{\Omega \setminus \Gamma} |\nabla u|^2 \, dx}_{\text{smoothness}} + \underbrace{\alpha \int_{\Omega} (u - f)^2 \, dx}_{\text{fidelity}} + \underbrace{\beta \mathcal{H}^1(\Gamma)}_{\text{boundary length}}$$

where:
- $u: \Omega \to \mathbb{R}$ is the piecewise smooth approximation
- $f: \Omega \to \mathbb{R}$ is the observed image
- $\Gamma$ is the set of discontinuities (boundaries)
- $\mathcal{H}^1(\Gamma)$ is the 1-dimensional Hausdorff measure (length of $\Gamma$)
- $\alpha, \beta > 0$ are balancing parameters

**Step 1 (Euler-Lagrange equation):** For fixed $\Gamma$, minimizing $E$ with respect to $u$ yields:

$$-\Delta u + \alpha(u - f) = 0 \quad \text{in } \Omega \setminus \Gamma$$

with Neumann boundary condition $\frac{\partial u}{\partial n} = 0$ on $\Gamma$.

**Step 2 (Piecewise constant limit):** As $\alpha \to \infty$, $u \to f$ away from $\Gamma$. As the smoothness term dominates, $u$ becomes piecewise constant:

$$E_{PC}(\{R_i\}, \Gamma) = \sum_i \alpha \int_{R_i}(c_i - f)^2 dx + \beta \mathcal{H}^1(\Gamma)$$

where $c_i$ is the mean intensity on region $R_i$.

**Step 3 (Optimal $c_i$):** For fixed regions, the optimal constants are:

$$c_i^* = \frac{\int_{R_i} f \, dx}{|R_i|} = \text{mean}(f|_{R_i})$$

**Step 4 (Chan-Vese special case):** For two regions ($K=2$):

$$E_{CV}(c_1, c_2, C) = \alpha_1\int_{\text{inside}(C)}(f - c_1)^2 dx + \alpha_2\int_{\text{outside}(C)}(f - c_2)^2 dx + \beta|C|$$

### 1.3 Level Set Method Derivation

**Step 1:** Represent the boundary $\Gamma$ implicitly via a level set function $\phi: \Omega \to \mathbb{R}$:

$$\Gamma = \{x \in \Omega : \phi(x) = 0\}$$
$$\text{inside} = \{x : \phi(x) > 0\}, \quad \text{outside} = \{x : \phi(x) < 0\}$$

**Step 2:** The Heaviside function $H(\phi)$ and Dirac delta $\delta(\phi)$:

$$H(\phi) = \begin{cases} 1 & \phi > 0 \\ 0 & \phi < 0 \end{cases}, \quad \delta(\phi) = \frac{dH}{d\phi}$$

**Step 3:** Rewrite the Chan-Vese energy using level sets:

$$E(\phi, c_1, c_2) = \alpha_1\int_\Omega (f-c_1)^2 H(\phi) \, dx + \alpha_2\int_\Omega (f-c_2)^2(1-H(\phi)) \, dx + \beta\int_\Omega |\nabla H(\phi)| \, dx$$

**Step 4:** Compute the first variation with respect to $\phi$:

$$\frac{\partial E}{\partial \phi} = \delta(\phi)\left[-\alpha_1(f-c_1)^2 + \alpha_2(f-c_2)^2 + \beta \cdot \text{div}\left(\frac{\nabla\phi}{|\nabla\phi|}\right)\right]$$

**Step 5:** Evolve $\phi$ via gradient descent (Hamilton-Jacobi equation):

$$\frac{\partial \phi}{\partial t} = \delta(\phi)\left[\alpha_1(f-c_1)^2 - \alpha_2(f-c_2)^2 - \beta \cdot \kappa\right]$$

where $\kappa = \text{div}(\nabla\phi / |\nabla\phi|)$ is the curvature of the level set.

**Step 6:** Reinitialize $\phi$ periodically to maintain the signed distance property $|\nabla\phi| = 1$ by solving:

$$\frac{\partial\phi}{\partial\tau} = \text{sign}(\phi_0)(1 - |\nabla\phi|)$$

### 1.4 Proof: Mumford-Shah Minimizer Exists

**Theorem (De Giorgi, Carriero, Leaci 1989):** The Mumford-Shah functional admits a minimizer in the space $SBV(\Omega)$ (special functions of bounded variation).

**Proof sketch:**

**Step 1:** The energy is bounded below by 0, so minimizing sequences exist.

**Step 2:** By compactness of $SBV$ under appropriate topology (Ambrosio's compactness theorem), minimizing sequences have convergent subsequences.

**Step 3:** The functional is lower semicontinuous with respect to this topology. $\blacksquare$

---

## 2. Algorithm / Method

### 2.1 Pseudocode

```
Algorithm: RL Region Growing Agent
─────────────────────────────────────────
Input: Image I, seed pixel(s)
Output: Segmented region R

State: s = (current_region_R, boundary_pixels, 
            region_statistics, image_features)
Action: a ∈ {grow_to_pixel_i, stop} for each boundary pixel i

Initialize DQN Q_θ

for episode = 1 to M do
    R₀ ← {seed}
    B₀ ← neighbors(seed)
    s₀ ← (R₀, B₀, stats(R₀))
    for t = 0 to T_max do
        aₜ ← ε-greedy(Q_θ(sₜ, ·))
        if aₜ == stop: break
        R_{t+1} ← R_t ∪ {pixel(aₜ)}
        B_{t+1} ← update_boundary(B_t, pixel(aₜ))
        rₜ ← ΔIoU(R_{t+1}, R*) - λ·boundary_cost
        Store (sₜ, aₜ, rₜ, s_{t+1}) in replay buffer
    end for
    Update Q_θ via DQN (experience replay + target network)
end for
```

### 2.2 Complexity Analysis

- **Boundary update:** $O(k)$ where $k$ is connectivity (4 or 8)
- **Region statistics update:** $O(1)$ with running mean/variance
- **IoU computation:** $O(|R|)$ per step or $O(1)$ with incremental update
- **DQN forward pass:** $O(|\theta|)$
- **Total per episode:** $O(T \cdot (|\theta| + k))$ where $T \leq N$

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

- **State:** Current region mask + boundary pixel features + region statistics (mean, variance, shape descriptors)
- **Action:** Select a boundary pixel to include or decide to stop growing
- **Reward:** IoU improvement per pixel added minus a regularization cost proportional to boundary irregularity
- **Transition:** Deterministic region expansion

### 3.2 Why RL?

1. **Adaptive threshold:** Instead of a fixed homogeneity criterion, the agent learns scene-dependent growing strategies
2. **Shape awareness:** The agent can learn to prefer compact, regular regions (implicit Mumford-Shah boundary term)
3. **Multi-scale reasoning:** Agent learns when to grow aggressively (interior) vs. cautiously (near boundaries)
4. **Optimal stopping:** Learning when to stop is crucial and naturally modeled as RL

---

## 4. Dataset

| Dataset | Size | Type | Description |
|---------|------|------|-------------|
| BSDS500 | 500 images | Boundaries | Berkeley segmentation |
| PASCAL VOC | 11,530 | Semantic | Object segmentation |
| GrabCut | 50 images | Interactive | Foreground extraction |
| MSRA-B | 5,000 | Salient object | Binary segmentation |

---

## 5. Key Equations Summary

| Equation | Description |
|----------|-------------|
| $E(u,\Gamma) = \int|\nabla u|^2 dx + \alpha\int(u-f)^2 dx + \beta\mathcal{H}^1(\Gamma)$ | Mumford-Shah functional |
| $\frac{\partial\phi}{\partial t} = \delta(\phi)[\alpha_1(f-c_1)^2 - \alpha_2(f-c_2)^2 - \beta\kappa]$ | Level set evolution |
| $\kappa = \text{div}(\nabla\phi/|\nabla\phi|)$ | Mean curvature |
| $P(R,v) = \mathbb{1}[|I(v)-\mu_R| < \tau\sigma_R]$ | Growing predicate |
| $c_i^* = \text{mean}(f|_{R_i})$ | Optimal region constant |

---

## 6. References

1. Mumford, D., & Shah, J. (1989). Optimal approximations by piecewise smooth functions and associated variational problems. *Communications on Pure and Applied Mathematics*, 42(5), 577-685.
2. Chan, T. F., & Vese, L. A. (2001). Active contours without edges. *IEEE TIP*, 10(2), 266-277.
3. Osher, S., & Sethian, J. A. (1988). Fronts propagating with curvature-dependent speed. *Journal of Computational Physics*, 79(1), 12-49.
4. Adams, R., & Bischof, L. (1994). Seeded region growing. *IEEE TPAMI*, 16(6), 641-647.
