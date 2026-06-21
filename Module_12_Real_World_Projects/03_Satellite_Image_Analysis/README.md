![Module Logo](../logo.png)

# Satellite Image Analysis via Reinforcement Learning

## Overview

This module applies reinforcement learning to satellite and remote sensing image analysis, combining SLAM-inspired state estimation with visual processing for geospatial intelligence. The mathematical framework covers Extended Kalman Filter (EKF) state estimation for tracking, Kalman filter prediction/update derivation, path planning as MDP with connections to A* search, and visual servoing control laws for autonomous satellite image acquisition.

## Prerequisites

- State estimation (Kalman filtering, Bayesian state estimation)
- Control theory (feedback control, stability, servoing)
- Graph algorithms (shortest paths, A*, Dijkstra)
- Linear systems (state-space models, observability)
- Reinforcement learning (model-based RL, planning)

---

## 1. Mathematical Foundations

### 1.1 EKF State Estimation for Visual Tracking

**Definition:** The Extended Kalman Filter estimates the state $\mathbf{x}_t$ of a nonlinear system:

$$\mathbf{x}_t = f(\mathbf{x}_{t-1}, \mathbf{u}_t) + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(0, Q)$$
$$\mathbf{z}_t = h(\mathbf{x}_t) + \mathbf{v}_t, \quad \mathbf{v}_t \sim \mathcal{N}(0, R)$$

where $f$ is the motion model, $h$ is the observation model, and $Q, R$ are noise covariances.

**Step 1 (Linearization):** Compute Jacobians at the current estimate:

$$F_t = \frac{\partial f}{\partial \mathbf{x}}\bigg|_{\hat{\mathbf{x}}_{t-1}}, \quad H_t = \frac{\partial h}{\partial \mathbf{x}}\bigg|_{\hat{\mathbf{x}}_{t|t-1}}$$

**Step 2 (Prediction):**

$$\hat{\mathbf{x}}_{t|t-1} = f(\hat{\mathbf{x}}_{t-1}, \mathbf{u}_t)$$
$$P_{t|t-1} = F_t P_{t-1} F_t^T + Q$$

**Step 3 (Innovation):**

$$\mathbf{y}_t = \mathbf{z}_t - h(\hat{\mathbf{x}}_{t|t-1})$$
$$S_t = H_t P_{t|t-1} H_t^T + R$$

**Step 4 (Kalman gain):**

$$K_t = P_{t|t-1} H_t^T S_t^{-1}$$

**Step 5 (Update):**

$$\hat{\mathbf{x}}_t = \hat{\mathbf{x}}_{t|t-1} + K_t \mathbf{y}_t$$
$$P_t = (I - K_t H_t) P_{t|t-1}$$

### 1.2 Kalman Filter Derivation: Optimality Proof

**Theorem:** The Kalman filter provides the minimum mean squared error (MMSE) estimate for linear Gaussian systems.

**Proof (for linear case $f = F\mathbf{x} + B\mathbf{u}$, $h = H\mathbf{x}$):**

**Step 1:** The posterior at time $t$ given all observations $\mathbf{z}_{1:t}$ is Gaussian:

$$p(\mathbf{x}_t | \mathbf{z}_{1:t}) = \mathcal{N}(\hat{\mathbf{x}}_t, P_t)$$

(Proved by induction: Gaussian prior + linear observation + Gaussian noise = Gaussian posterior.)

**Step 2:** The MMSE estimate is the posterior mean: $\hat{\mathbf{x}}_t^{MMSE} = E[\mathbf{x}_t | \mathbf{z}_{1:t}]$.

**Step 3:** For the update step, compute the joint distribution of $(\mathbf{x}_t, \mathbf{z}_t)$:

$$\begin{pmatrix}\mathbf{x}_t \\ \mathbf{z}_t\end{pmatrix} \sim \mathcal{N}\left(\begin{pmatrix}\hat{\mathbf{x}}_{t|t-1} \\ H\hat{\mathbf{x}}_{t|t-1}\end{pmatrix}, \begin{pmatrix}P_{t|t-1} & P_{t|t-1}H^T \\ HP_{t|t-1} & S_t\end{pmatrix}\right)$$

**Step 4:** The conditional mean (MMSE estimate):

$$E[\mathbf{x}_t|\mathbf{z}_t] = \hat{\mathbf{x}}_{t|t-1} + P_{t|t-1}H^T S_t^{-1}(\mathbf{z}_t - H\hat{\mathbf{x}}_{t|t-1})$$

This is exactly the Kalman filter update. $\blacksquare$

**Step 5:** The conditional covariance:

$$\text{Cov}[\mathbf{x}_t|\mathbf{z}_t] = P_{t|t-1} - P_{t|t-1}H^T S_t^{-1} H P_{t|t-1} = (I - K_t H)P_{t|t-1}$$

### 1.3 Path Planning as MDP: A* as Value Iteration

**Step 1 (Grid MDP):** Satellite coverage planning on a grid world:
- State: $(x, y)$ position on the grid
- Action: $\{N, S, E, W, NE, NW, SE, SW\}$ (8-connected)
- Reward: $R(s) = \text{information\_value}(s) - \text{movement\_cost}(s, a)$
- Goal: Maximize total information gathered along path

**Step 2 (Bellman equation for shortest path):**

$$V^*(s) = \min_a\left[c(s, a) + V^*(s')\right] \quad \text{for } s \neq s_{goal}$$

$$V^*(s_{goal}) = 0$$

**Step 3 (A* as heuristic-guided value iteration):** A* maintains:

$$f(n) = g(n) + h(n)$$

where $g(n)$ is the cost-to-come (actual) and $h(n)$ is the cost-to-go heuristic.

**Step 4 (A* optimality condition):** If $h(n)$ is admissible ($h(n) \leq V^*(n)$ for all $n$), then A* finds the optimal path.

**Proof:** A* expands nodes in order of $f(n)$. At goal, $f(goal) = g(goal) + 0 = g(goal)$ (actual cost). For any unexpanded node $n'$ on an alternative path: $f(n') = g(n') + h(n') \leq g(n') + V^*(n') = \text{total cost via } n'$. Since $f(goal) \leq f(n')$ (it was expanded first), the found path is optimal. $\blacksquare$

**Step 5 (Connection to value iteration):** A* is equivalent to performing value iteration only on states reachable from the start, guided by the heuristic to avoid exploring irrelevant states.

### 1.4 Visual Servoing Control Law

**Step 1 (Image-Based Visual Servoing):** Define visual features $\mathbf{s}(t) \in \mathbb{R}^k$ extracted from the image (e.g., point coordinates, moments).

**Step 2 (Feature dynamics):** The relationship between feature change and camera velocity $\mathbf{v}_c = (v, \omega) \in \mathbb{R}^6$:

$$\dot{\mathbf{s}} = L_s \mathbf{v}_c$$

where $L_s \in \mathbb{R}^{k\times 6}$ is the interaction matrix (image Jacobian).

**Step 3 (For a point feature $(u, v)$ at depth $Z$):**

$$L_s = \begin{pmatrix}-1/Z & 0 & u/Z & uv & -(1+u^2) & v \\ 0 & -1/Z & v/Z & 1+v^2 & -uv & -u\end{pmatrix}$$

**Step 4 (Control law):** To drive features to desired values $\mathbf{s}^*$:

Define error: $\mathbf{e} = \mathbf{s} - \mathbf{s}^*$

Proportional control: $\mathbf{v}_c = -\lambda\hat{L}_s^+ \mathbf{e}$

where $\hat{L}_s^+$ is the pseudoinverse of the estimated interaction matrix.

**Step 5 (Stability proof):** Choose Lyapunov function $V = \frac{1}{2}\|\mathbf{e}\|^2$:

$$\dot{V} = \mathbf{e}^T\dot{\mathbf{e}} = \mathbf{e}^T L_s\mathbf{v}_c = -\lambda\mathbf{e}^T L_s\hat{L}_s^+\mathbf{e}$$

If $L_s\hat{L}_s^+ \succ 0$ (positive definite), then $\dot{V} < 0$, guaranteeing asymptotic convergence. $\blacksquare$

---

## 2. Algorithm / Method

### 2.1 Pseudocode

```
Algorithm: RL Satellite Image Analysis Agent
──────────────────────────────────────────────
Input: Satellite image stream, mission objectives
Output: Optimal observation path + processed imagery

State: s = (current_position, observed_map, target_locations,
            fuel_remaining, image_quality_estimates)
Action: a = (next_observation_point, sensor_params, 
             processing_pipeline_selection)

Initialize: Policy π_θ, EKF state estimator, coverage map

for mission_step = 1 to T_mission do
    // State estimation
    z_t ← acquire_observation(current_position)
    x̂_t ← EKF_update(x̂_{t-1}, z_t)
    
    // RL decision
    a_t ~ π_θ(·|s_t)
    next_pos, sensor_config ← decode(a_t)
    
    // Execute and compute reward
    Move to next_pos (with fuel cost)
    Acquire image with sensor_config
    r_t ← information_gain - fuel_cost - time_cost
    
    // Update coverage and state
    Update observed_map
    s_{t+1} ← (next_pos, updated_map, ...)
end for
```

### 2.2 Complexity Analysis

- **EKF prediction:** $O(n^2)$ where $n$ is state dimension
- **EKF update:** $O(n^2 m + m^3)$ for $m$ measurements
- **A* path planning:** $O(|V|\log|V|)$ for $|V|$ grid cells
- **Visual servoing:** $O(k \cdot 6)$ for $k$ visual features
- **RL policy evaluation:** $O(|\theta|)$ per decision

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

- **State:** Satellite position + coverage map + EKF belief state + resource levels
- **Action:** Next observation target + sensor configuration + processing choice
- **Reward:** Information gathered (entropy reduction) - fuel cost - revisit penalty
- **Transition:** Orbital mechanics (deterministic) + observation quality (stochastic)
- **Horizon:** Mission duration (finite)

### 3.2 Why RL?

1. **Sequential planning:** Observation order affects information gain (covering adjacent regions provides context)
2. **Resource constraints:** Fuel and time budgets create complex trade-offs best handled by learned policies
3. **Adaptive replanning:** Weather, cloud cover, and targets of opportunity require dynamic decisions
4. **Multi-objective:** Coverage, timeliness, and image quality must be balanced

---

## 4. Dataset

| Dataset | Resolution | Size | Description |
|---------|-----------|------|-------------|
| SpaceNet | 0.3m-4m | 11,000+ | Building/road segmentation |
| DOTA | Variable | 2,806 | Object detection in aerial |
| xView | 0.3m | 1,415 km² | 60 object classes |
| RESISC45 | Variable | 31,500 | Scene classification |

---

## 5. Key Equations Summary

| Equation | Description |
|----------|-------------|
| $K_t = P_{t|t-1}H^TS_t^{-1}$ | Kalman gain |
| $\hat{x}_t = \hat{x}_{t|t-1} + K_t(z_t - H\hat{x}_{t|t-1})$ | KF update |
| $f(n) = g(n) + h(n)$ | A* evaluation function |
| $\mathbf{v}_c = -\lambda\hat{L}_s^+\mathbf{e}$ | Visual servoing control |
| $\dot{\mathbf{s}} = L_s\mathbf{v}_c$ | Feature dynamics |

---

## 6. References

1. Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press.
2. Chaumette, F., & Hutchinson, S. (2006). Visual servo control. Part I: Basic approaches. *IEEE RAM*, 13(4), 82-90.
3. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. *IEEE TSS*, 4(2), 100-107.
4. Shermeyer, J., & Van Etten, A. (2019). The effects of super-resolution on object detection performance in satellite imagery. *CVPR Workshops*.
