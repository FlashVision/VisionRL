![Module Logo](../logo.png)

# Sim-to-Real Transfer for Vision

## Overview

This module addresses the sim-to-real transfer problem for visual RL agents, where policies trained in simulated environments must work in the real world despite the domain gap. The mathematical framework covers domain randomization theory as uniform minimax optimization, derives the Ben-David domain adaptation bound, and formulates system identification as an RL problem for closing the reality gap.

## Prerequisites

- Statistical learning theory (VC dimension, Rademacher complexity)
- Domain adaptation (covariate shift, distribution matching)
- Robust optimization (minimax, distributionally robust)
- System identification (parameter estimation, model calibration)
- Reinforcement learning (transfer learning, domain randomization)

---

## 1. Mathematical Foundations

### 1.1 Domain Randomization Theory (Uniform Minimax)

**Definition:** Domain randomization trains a policy $\pi$ to perform well across a distribution of environment parameters $\xi \sim p(\xi)$:

$$\pi^* = \arg\max_\pi E_{\xi \sim p(\xi)}\left[J(\pi, \xi)\right]$$

where $J(\pi, \xi)$ is the expected return in environment with parameters $\xi$.

**Step 1 (Minimax interpretation):** If we don't know the real-world parameters $\xi^*$, we can take a robust approach:

$$\pi^* = \arg\max_\pi \min_{\xi \in \Xi} J(\pi, \xi)$$

This finds the policy that performs best in the worst-case environment.

**Step 2 (Uniform randomization as approximation):** When $p(\xi) = \text{Uniform}(\Xi)$ and $\Xi$ is bounded:

$$E_{\xi\sim\text{Unif}(\Xi)}[J(\pi, \xi)] \geq \min_{\xi\in\Xi}J(\pi,\xi) + \frac{\text{Var}_\xi[J(\pi,\xi)]}{2(\max J - \min J)}$$

Maximizing the uniform expectation approximately maximizes the worst case (up to variance).

**Step 3 (Sufficient coverage condition):** If the real-world parameters $\xi^* \in \Xi$ (the randomization range contains reality):

$$J(\pi^*, \xi^*) \geq E_\xi[J(\pi^*, \xi)] - \sqrt{\text{Var}_\xi[J(\pi^*, \xi)]}$$

by Chebyshev's inequality. As variance decreases (robust policy), performance at $\xi^*$ approaches the average.

**Step 4 (Randomization dimensions for vision):**
- Texture: color, material, surface normal perturbations
- Lighting: position, intensity, color temperature
- Camera: focal length, distortion, noise level
- Geometry: object scale, deformation, pose

### 1.2 Domain Adaptation Bound (Ben-David et al.)

**Theorem (Ben-David et al. 2010):** Let $\hat{h}$ be a hypothesis learned on source domain $S$. Its error on target domain $T$ is bounded by:

$$\epsilon_T(\hat{h}) \leq \epsilon_S(\hat{h}) + d_{\mathcal{H}\Delta\mathcal{H}}(S, T) + \lambda^*$$

where:
- $\epsilon_S(\hat{h})$: source error
- $d_{\mathcal{H}\Delta\mathcal{H}}(S, T)$: $\mathcal{H}\Delta\mathcal{H}$-divergence between domains
- $\lambda^* = \min_{h^*}[\epsilon_S(h^*) + \epsilon_T(h^*)]$: optimal combined error

**Proof:**

**Step 1:** Define the $\mathcal{H}\Delta\mathcal{H}$-divergence:

$$d_{\mathcal{H}\Delta\mathcal{H}}(S, T) = 2\sup_{h, h' \in \mathcal{H}}\left|P_S[h(x) \neq h'(x)] - P_T[h(x) \neq h'(x)]\right|$$

This measures how well a classifier can distinguish between the two domains.

**Step 2:** For any two hypotheses $h, h'$:

$$|\epsilon_S(h, h') - \epsilon_T(h, h')| \leq \frac{1}{2}d_{\mathcal{H}\Delta\mathcal{H}}(S, T)$$

where $\epsilon_D(h, h') = P_{x\sim D}[h(x) \neq h'(x)]$.

**Step 3:** Let $h^*$ be the optimal joint hypothesis. By triangle inequality for disagreement:

$$\epsilon_T(\hat{h}) \leq \epsilon_T(\hat{h}, h^*) + \epsilon_T(h^*)$$

**Step 4:** Bound the first term using domain divergence:

$$\epsilon_T(\hat{h}, h^*) \leq \epsilon_S(\hat{h}, h^*) + \frac{1}{2}d_{\mathcal{H}\Delta\mathcal{H}}(S, T)$$

**Step 5:** Apply triangle inequality again:

$$\epsilon_S(\hat{h}, h^*) \leq \epsilon_S(\hat{h}) + \epsilon_S(h^*)$$

**Step 6:** Combining:

$$\epsilon_T(\hat{h}) \leq \epsilon_S(\hat{h}) + \epsilon_S(h^*) + \epsilon_T(h^*) + \frac{1}{2}d_{\mathcal{H}\Delta\mathcal{H}}(S, T)$$

$$= \epsilon_S(\hat{h}) + d_{\mathcal{H}\Delta\mathcal{H}}(S, T) + \lambda^* \quad \blacksquare$$

**Step 7 (Implications for sim-to-real):**
- Minimize $\epsilon_S(\hat{h})$: train well in simulation
- Minimize $d_{\mathcal{H}\Delta\mathcal{H}}(S, T)$: make simulation realistic (domain randomization reduces this)
- $\lambda^*$ is irreducible if source and target are fundamentally different

### 1.3 Empirical $\mathcal{H}$-Divergence Estimation

**Step 1:** Train a domain classifier $D_\omega$ to distinguish source from target samples:

$$\hat{d}_{\mathcal{H}}(S, T) = 2\left(1 - 2\min_\omega \left[\frac{1}{n_S}\sum_{x\in S}\mathcal{L}(D_\omega(x), 0) + \frac{1}{n_T}\sum_{x\in T}\mathcal{L}(D_\omega(x), 1)\right]\right)$$

**Step 2:** If the domain classifier achieves accuracy $\alpha$: $\hat{d} = 2(2\alpha - 1)$.

**Step 3:** Perfect classification ($\alpha = 1$): $\hat{d} = 2$ (maximum divergence). Random ($\alpha = 0.5$): $\hat{d} = 0$ (domains indistinguishable).

### 1.4 System Identification as RL

**Step 1:** Define simulator parameters $\xi = (\xi_1, \ldots, \xi_K)$ (friction, mass, texture, lighting, etc.).

**Step 2 (Objective):** Find $\xi^*$ that minimizes the reality gap:

$$\xi^* = \arg\min_\xi d(\text{sim}(\xi), \text{real})$$

where $d$ measures the discrepancy between simulated and real observations/trajectories.

**Step 3 (RL formulation for SimOpt):**
- State: current simulator parameters + comparison metrics
- Action: adjustment to simulator parameters $\Delta\xi$
- Reward: $-d(\text{sim}(\xi + \Delta\xi), \text{real})$ (negative reality gap)

**Step 4 (Bayesian system identification):** Maintain posterior over $\xi$:

$$p(\xi | D_{real}) \propto p(D_{real} | \xi) p(\xi)$$

$$p(D_{real} | \xi) = \prod_{(s,a,s') \in D_{real}} P_{sim}(s'|s,a;\xi)$$

**Step 5 (Active system identification):** Choose real-world experiments that maximize information about $\xi$:

$$a^* = \arg\max_a I(\xi; s'|s, a)$$

---

## 2. Algorithm / Method

### 2.1 Pseudocode

```
Algorithm: Domain Randomization + RL Sim-to-Real
─────────────────────────────────────────────────
Input: Simulator Sim(ξ), randomization ranges Ξ
Output: Policy π that transfers to real world

Phase 1: Domain Randomization Training
    for episode = 1 to M do
        ξ ~ Uniform(Ξ)         // Randomize parameters
        env ← Sim(ξ)           // Create randomized environment
        Rollout τ ~ π_θ in env
        Update π_θ via PPO
    end for

Phase 2 (Optional): System Identification
    Collect real-world data D_real
    for iteration = 1 to K do
        ξ_est ← argmin_ξ d(Sim(ξ), D_real)
        Refine policy in Sim(ξ_est)
    end for

Phase 3 (Optional): Fine-tuning
    for episode = 1 to M_real do
        Rollout in real world (limited budget)
        Fine-tune π_θ with small learning rate
    end for
```

### 2.2 Complexity Analysis

- **Domain randomization overhead:** $O(|\xi| \cdot \text{render\_cost})$ per episode
- **Standard RL training:** $O(T \cdot |\theta|)$ per episode
- **System identification (Bayesian):** $O(|D_{real}| \cdot |\xi|^2)$ per update
- **Domain classifier training:** $O((n_S + n_T) \cdot |\omega|)$
- **Total:** Dominated by RL training cost; randomization adds constant factor

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

$$\mathcal{M}_{sim} = (\mathcal{S}, \mathcal{A}, P_\xi, R, \gamma)$$

- **State:** Visual observations (rendered images from simulator)
- **Action:** Robot/agent actions in the environment
- **Dynamics:** $P_\xi(s'|s,a)$ parameterized by simulator settings $\xi$
- **Transfer goal:** Policy $\pi^*_{sim}$ works on $P_{real}$
- **Reality gap:** $\|P_\xi - P_{real}\| \leq \epsilon$ for some $\xi^* \in \Xi$

### 3.2 Why RL for Sim-to-Real?

1. **Safe exploration:** Simulation allows unlimited exploration without real-world risks
2. **Data efficiency:** Real-world data is expensive; simulation provides infinite training data
3. **Robustness:** Domain randomization naturally produces robust policies
4. **Automatic adaptation:** RL can learn to identify and adapt to domain differences

---

## 4. Dataset / Environments

| Environment | Domain Gap | Type | Description |
|-------------|-----------|------|-------------|
| MuJoCo → Real Robot | Physics | Control | Sim dynamics mismatch |
| Unity → Real World | Visual | Vision | Rendering vs. reality |
| CARLA → Real Driving | Both | Autonomous | Full driving simulation |
| AI2-THOR → Real Home | Both | Navigation | Indoor navigation |

---

## 5. Key Equations Summary

| Equation | Description |
|----------|-------------|
| $\epsilon_T(\hat{h}) \leq \epsilon_S(\hat{h}) + d_{\mathcal{H}\Delta\mathcal{H}}(S,T) + \lambda^*$ | Domain adaptation bound |
| $\pi^* = \arg\max_\pi\min_{\xi\in\Xi}J(\pi,\xi)$ | Robust policy (minimax) |
| $d_{\mathcal{H}\Delta\mathcal{H}} = 2\sup_{h,h'}\|P_S[h\neq h']-P_T[h\neq h']\|$ | Domain divergence |
| $\xi^* = \arg\min_\xi d(\text{sim}(\xi),\text{real})$ | System identification |
| $p(\xi|D_{real}) \propto p(D_{real}|\xi)p(\xi)$ | Bayesian SysID |

---

## 6. References

1. Tobin, J., et al. (2017). Domain randomization for transferring deep neural networks from simulation to the real world. *IROS*.
2. Ben-David, S., et al. (2010). A theory of learning from different domains. *Machine Learning*, 79, 151-175.
3. Peng, X. B., et al. (2018). Sim-to-real transfer of robotic control with dynamics randomization. *ICRA*.
4. Ganin, Y., et al. (2016). Domain-adversarial training of neural networks. *JMLR*, 17, 1-35.
5. Ramos, F., et al. (2019). BayesSim: Adaptive domain randomization via probabilistic inference for robotics simulators. *RSS*.
