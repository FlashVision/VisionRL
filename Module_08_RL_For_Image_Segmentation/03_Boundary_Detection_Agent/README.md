![Module Logo](../logo.png)

# Boundary Detection Agent via Reinforcement Learning

## Overview

This module develops an RL agent that detects and refines object boundaries in images by controlling active contour (snake) parameters in real-time. The mathematical framework derives the energy functional for active contours, solves the Euler-Lagrange equation for optimal contour placement, extends to geodesic active contours, and formulates boundary refinement as a sequential RL problem.

## Prerequisites

- Variational calculus (Euler-Lagrange equations, first variation)
- Differential geometry (curvature, arc length, geodesics)
- Partial differential equations (gradient descent flows)
- Numerical methods (finite differences, implicit schemes)
- Reinforcement learning (continuous control, actor-critic)

---

## 1. Mathematical Foundations

### 1.1 Active Contour (Snake) Energy

**Definition:** A parametric active contour is a curve $\mathbf{v}(s) = (x(s), y(s))$ for $s \in [0, 1]$ that minimizes:

$$E_{snake} = \int_0^1 \left[E_{internal}(\mathbf{v}(s)) + E_{external}(\mathbf{v}(s))\right] ds$$

**Internal energy (regularization):**

$$E_{internal} = \frac{1}{2}\left(\alpha(s)|\mathbf{v}'(s)|^2 + \beta(s)|\mathbf{v}''(s)|^2\right)$$

where:
- $\alpha(s)|\mathbf{v}'(s)|^2$ is the elasticity term (penalizes stretching)
- $\beta(s)|\mathbf{v}''(s)|^2$ is the rigidity term (penalizes bending)

**External energy (image forces):**

$$E_{external} = -|\nabla I(\mathbf{v}(s))|^2$$

or using a Gaussian-smoothed gradient:

$$E_{external} = -|\nabla(G_\sigma * I)(\mathbf{v}(s))|^2$$

### 1.2 Euler-Lagrange Equation for Optimal Contour

**Step 1:** The total energy functional is:

$$E[\mathbf{v}] = \int_0^1 \left[\frac{\alpha}{2}|\mathbf{v}'|^2 + \frac{\beta}{2}|\mathbf{v}''|^2 + E_{ext}(\mathbf{v})\right] ds$$

**Step 2:** Take the first variation. For an admissible perturbation $\mathbf{h}(s)$ with $\mathbf{h}(0) = \mathbf{h}(1) = 0$:

$$\delta E = \lim_{\epsilon\to 0}\frac{d}{d\epsilon}E[\mathbf{v} + \epsilon\mathbf{h}]\bigg|_{\epsilon=0}$$

**Step 3:** Compute each term's variation:

$$\delta\int\frac{\alpha}{2}|\mathbf{v}'|^2 ds = \int \alpha \mathbf{v}' \cdot \mathbf{h}' \, ds = -\int \alpha \mathbf{v}'' \cdot \mathbf{h} \, ds$$

(integration by parts, boundary terms vanish)

$$\delta\int\frac{\beta}{2}|\mathbf{v}''|^2 ds = \int \beta \mathbf{v}'' \cdot \mathbf{h}'' \, ds = \int \beta \mathbf{v}'''' \cdot \mathbf{h} \, ds$$

(two integrations by parts)

$$\delta\int E_{ext}(\mathbf{v}) \, ds = \int \nabla E_{ext} \cdot \mathbf{h} \, ds$$

**Step 4:** Setting $\delta E = 0$ for all $\mathbf{h}$:

$$-\alpha \mathbf{v}''(s) + \beta \mathbf{v}''''(s) + \nabla E_{ext}(\mathbf{v}(s)) = 0$$

This is the Euler-Lagrange equation for the optimal snake.

**Step 5:** Introduce time evolution (gradient descent):

$$\frac{\partial \mathbf{v}}{\partial t} = \alpha \mathbf{v}'' - \beta \mathbf{v}'''' - \nabla E_{ext}$$

**Step 6:** Discretize: let $\mathbf{v}_i = \mathbf{v}(i/n)$ for $i = 0, \ldots, n-1$. The finite difference approximation:

$$\mathbf{v}_i'' \approx \mathbf{v}_{i-1} - 2\mathbf{v}_i + \mathbf{v}_{i+1}$$
$$\mathbf{v}_i'''' \approx \mathbf{v}_{i-2} - 4\mathbf{v}_{i-1} + 6\mathbf{v}_i - 4\mathbf{v}_{i+1} + \mathbf{v}_{i+2}$$

**Step 7:** Matrix form with implicit time stepping:

$$(\mathbf{I} + \Delta t \mathbf{A})\mathbf{v}^{t+1} = \mathbf{v}^t + \Delta t \cdot \mathbf{f}_{ext}(\mathbf{v}^t)$$

where $\mathbf{A}$ is the pentadiagonal matrix encoding the internal forces.

### 1.3 Geodesic Active Contour Derivation

**Step 1:** The classical snake has a geometric interpretation. Consider minimizing the weighted arc length:

$$L_g(C) = \int_0^1 g(|\nabla I(C(s))|) |C'(s)| \, ds$$

where $g: [0,\infty) \to (0,1]$ is a monotonically decreasing edge indicator function:

$$g(|\nabla I|) = \frac{1}{1 + |\nabla I|^2/\lambda^2}$$

**Step 2:** This is a geodesic computation in a Riemannian space with metric weighted by $g$. The curve minimizing $L_g$ is a geodesic in this metric.

**Step 3:** The Euler-Lagrange equation for the geodesic:

$$\frac{\partial C}{\partial t} = g(|\nabla I|)\kappa\mathbf{n} - (\nabla g \cdot \mathbf{n})\mathbf{n}$$

where $\kappa$ is the curvature and $\mathbf{n}$ is the unit inward normal.

**Step 4:** Using the level set formulation $C = \{\phi = 0\}$:

$$\frac{\partial \phi}{\partial t} = |\nabla\phi|\left[\text{div}\left(g\frac{\nabla\phi}{|\nabla\phi|}\right) + c \cdot g\right]$$

**Step 5:** Expand:

$$\frac{\partial \phi}{\partial t} = g|\nabla\phi|\,\kappa + \nabla g \cdot \nabla\phi + c\,g|\nabla\phi|$$

where $c$ is a balloon force constant that prevents the contour from shrinking to a point.

**Step 6:** The geodesic model unifies internal energy (curvature) and external energy (edge indicator $g$) in a single geometric framework.

### 1.4 Proof: Snake Energy Minimum is a Saddle Point of Constrained System

**Theorem:** The snake equilibrium satisfying the Euler-Lagrange equation is a minimum of the total energy subject to the contour remaining on the image plane.

**Proof:** The second variation of the internal energy is:

$$\delta^2 E_{int} = \int_0^1 (\alpha|\mathbf{h}'|^2 + \beta|\mathbf{h}''|^2) \, ds \geq 0$$

This is a positive semi-definite quadratic form for $\alpha, \beta \geq 0$. Combined with the convexity of the external potential near edges, the critical point is a local minimum. $\blacksquare$

---

## 2. Algorithm / Method

### 2.1 Pseudocode

```
Algorithm: RL Boundary Detection Agent
─────────────────────────────────────────
Input: Image I, initial contour C₀
Output: Refined boundary contour C*

State: s = (contour_points, local_gradients, curvature_profile,
            internal_energy, distance_to_edges)
Action: a = (Δα, Δβ, Δσ, balloon_force, step_size) ∈ ℝ⁵

Initialize actor π_θ, critic V_φ

for episode = 1 to M do
    C₀ ← initialize_contour(I)
    s₀ ← extract_contour_features(C₀, I)
    for t = 0 to T_max do
        aₜ ~ π_θ(·|sₜ)
        α_t, β_t, σ_t ← update_params(aₜ)
        C_{t+1} ← evolve_snake(Cₜ, α_t, β_t, σ_t, Δt)
        rₜ ← boundary_F1(C_{t+1}) - boundary_F1(Cₜ) - c·|aₜ|
        s_{t+1} ← extract_contour_features(C_{t+1}, I)
    end for
    Update π_θ, V_φ via PPO
end for
```

### 2.2 Complexity Analysis

- **Snake evolution (one step):** $O(n)$ for $n$ contour points (tridiagonal solve)
- **Gradient computation:** $O(n \cdot k^2)$ with local window of size $k$
- **Level set evolution:** $O(N)$ per time step for narrow band method
- **Feature extraction:** $O(n \cdot d)$ for $d$-dimensional local descriptors
- **Total per episode:** $O(T \cdot n \cdot k^2)$

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

- **State:** Contour shape descriptors, local edge strengths along contour, curvature profile, convergence indicator
- **Action:** Continuous adjustments to snake parameters $(\alpha, \beta, \sigma, c)$ applied globally or locally
- **Reward:** Improvement in boundary F1-score + smoothness bonus - energy cost
- **Transition:** One or more iterations of snake evolution under selected parameters

### 3.2 Why RL?

1. **Parameter sensitivity:** Snake performance is highly sensitive to $\alpha, \beta, \sigma$; RL learns adaptive parameter schedules
2. **Initialization dependence:** The agent can learn corrective actions for poor initializations
3. **Local minima escape:** RL can learn to temporarily increase energy to escape local minima (exploration)
4. **Real-time control:** Unlike offline optimization, RL enables interactive boundary refinement

---

## 4. Dataset

| Dataset | Size | Type | Description |
|---------|------|------|-------------|
| BSDS500 | 500 images | Boundary | Human-annotated boundaries |
| Weizmann | 200 images | Contour | Horse silhouettes |
| MPEG-7 CE Shape 1 | 1,400 | Shape | Binary shape boundaries |
| SBD | 11,355 | Boundary | Semantic boundaries |

---

## 5. Key Equations Summary

| Equation | Description |
|----------|-------------|
| $E_{snake} = \int[E_{int} + E_{ext}] ds$ | Snake total energy |
| $-\alpha\mathbf{v}'' + \beta\mathbf{v}'''' + \nabla E_{ext} = 0$ | Euler-Lagrange equation |
| $L_g(C) = \int g(\|\nabla I\|)|C'| ds$ | Geodesic active contour length |
| $\frac{\partial\phi}{\partial t} = g\|\nabla\phi\|\kappa + \nabla g\cdot\nabla\phi$ | Level set evolution |
| $g(\|\nabla I\|) = 1/(1 + \|\nabla I\|^2/\lambda^2)$ | Edge indicator function |

---

## 6. References

1. Kass, M., Witkin, A., & Terzopoulos, D. (1988). Snakes: Active contour models. *IJCV*, 1(4), 321-331.
2. Caselles, V., Kimmel, R., & Sapiro, G. (1997). Geodesic active contours. *IJCV*, 22(1), 61-79.
3. Xu, C., & Prince, J. L. (1998). Snakes, shapes, and gradient vector flow. *IEEE TIP*, 7(3), 359-369.
4. Marcos, D., Tuia, D., Kellenberger, B., et al. (2018). Learning deep structured active contours end-to-end. *CVPR*.
