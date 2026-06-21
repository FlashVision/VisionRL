![Module Logo](../logo.png)

# Hierarchical Reinforcement Learning for Vision

## Overview

This module applies Hierarchical Reinforcement Learning (HRL) to complex vision tasks, enabling agents to reason at multiple temporal abstractions. The mathematical framework derives the options framework for temporal abstraction, the Semi-Markov Decision Process (SMDP) formulation, Bellman equations over options, and the Feudal Networks architecture with manager-worker decomposition for visual processing.

## Prerequisites

- Markov Decision Processes (value functions, Bellman equations)
- Semi-Markov processes (variable-length transitions)
- Temporal abstraction (macro-actions, hierarchical planning)
- Goal-conditioned RL (universal value functions)
- Deep learning (recurrent networks, hierarchical representations)

---

## 1. Mathematical Foundations

### 1.1 Options Framework

**Definition (Sutton, Precup, Singh 1999):** An option $\omega$ is a triple:

$$\omega = (\mathcal{I}_\omega, \pi_\omega, \beta_\omega)$$

where:
- $\mathcal{I}_\omega \subseteq \mathcal{S}$: Initiation set (states where $\omega$ can start)
- $\pi_\omega: \mathcal{S} \times \mathcal{A} \to [0,1]$: Intra-option policy (how to execute)
- $\beta_\omega: \mathcal{S} \to [0,1]$: Termination function (probability of ending)

**Step 1:** An option $\omega$ initiated in state $s \in \mathcal{I}_\omega$ executes by:
1. Selecting actions $a \sim \pi_\omega(\cdot|s)$
2. Transitioning to $s'$ via environment dynamics
3. Terminating with probability $\beta_\omega(s')$

**Step 2:** The policy over options $\mu: \mathcal{S} \times \Omega \to [0,1]$ selects which option to execute:

$$\mu(\omega|s) = P(\text{execute option } \omega | \text{state } s)$$

**Step 3:** The effective policy under options:

$$\pi(a|s) = \sum_{\omega: s \in \mathcal{I}_\omega} P(\omega \text{ active in } s) \cdot \pi_\omega(a|s)$$

### 1.2 Semi-MDP (SMDP) Derivation

**Step 1:** When options have variable duration, the process becomes a Semi-MDP. The transition under option $\omega$ starting in state $s$:

$$P(s', k | s, \omega) = P(\text{reach } s' \text{ in } k \text{ steps under } \pi_\omega \text{ and terminate})$$

**Step 2 (Multi-step transition):** Expanding recursively:

$$P(s', k | s, \omega) = \sum_{s''} P(s''|s, \pi_\omega(s)) \cdot [(1-\beta_\omega(s'))\cdot P(s', k-1|s'', \omega) \cdot \mathbb{1}_{k>1} + \beta_\omega(s')\cdot\mathbb{1}_{s''=s'}\cdot\mathbb{1}_{k=1}]$$

**Step 3 (Expected cumulative reward):** The reward accumulated during option $\omega$ from state $s$:

$$r(s, \omega) = E\left[\sum_{t=0}^{k-1}\gamma^t R(s_t, a_t) | s_0 = s, \omega\right]$$

**Step 4 (SMDP transition probability):**

$$\tilde{P}(s'|s, \omega) = \sum_{k=1}^\infty P(s', k | s, \omega)$$

**Step 5 (Effective discount):**

$$\tilde{\gamma}(s, \omega, s') = E[\gamma^k | s, \omega, s'] = \frac{\sum_k \gamma^k P(s',k|s,\omega)}{\tilde{P}(s'|s,\omega)}$$

### 1.3 Bellman Equation for Options

**Step 1:** The value function over options policy $\mu$:

$$V^\mu(s) = \sum_\omega \mu(\omega|s) Q^\mu(s, \omega)$$

**Step 2:** The Q-function for option $\omega$:

$$Q^\mu(s, \omega) = r(s, \omega) + \sum_{s'}\tilde{P}(s'|s,\omega)\tilde{\gamma}(s,\omega,s') V^\mu(s')$$

**Step 3 (Intra-option value function):** The value of being in state $s$ while executing option $\omega$:

$$V_\omega(s) = \sum_a \pi_\omega(a|s) Q_\omega(s, a)$$

**Step 4 (Intra-option Q-function):**

$$Q_\omega(s, a) = R(s, a) + \gamma\sum_{s'} P(s'|s, a) U_\omega(s')$$

where $U_\omega(s')$ is the value upon arrival at $s'$:

$$U_\omega(s') = (1-\beta_\omega(s'))V_\omega(s') + \beta_\omega(s')V^\mu(s')$$

**Step 5 (Option-value iteration):** Iterate until convergence:

$$Q^\mu(s, \omega) \leftarrow r(s, \omega) + \sum_{s'}\tilde{P}(s'|s,\omega)\tilde{\gamma}\max_{\omega':\,s'\in\mathcal{I}_{\omega'}} Q^\mu(s', \omega')$$

This converges to the optimal Q-function over options.

### 1.4 Feudal Networks (Manager-Worker Architecture)

**Step 1 (Manager):** Operates at lower temporal resolution, setting goals:

$$g_t = f_{manager}(s_t) \in \mathbb{R}^d$$

The manager selects a direction in latent space every $c$ timesteps.

**Step 2 (Worker):** Operates at every timestep, conditioned on the goal:

$$\pi_{worker}(a|s, g) = \text{softmax}(\mathbf{W}\phi(s) + \mathbf{U}g)$$

**Step 3 (Manager's transition policy):** The manager models transitions in a learned state space:

$$s_t^{manager} = f_{percept}(s_t) \in \mathbb{R}^d$$

$$g_t = \frac{s_{t+c}^{manager} - s_t^{manager}}{\|s_{t+c}^{manager} - s_t^{manager}\|}$$

The goal is the normalized direction of intended state change.

**Step 4 (Intrinsic reward for worker):**

$$r_t^{intrinsic} = \frac{1}{c}\sum_{i=1}^c \cos(s_{t}^{manager} - s_{t-i}^{manager}, g_{t-i})$$

This rewards the worker for moving in the direction the manager specified.

**Step 5 (Manager's objective):** The manager maximizes extrinsic reward at its coarser timescale:

$$J_{manager} = E\left[\sum_{t=0,c,2c,\ldots}\gamma^{t/c} R_t^{ext}\right]$$

**Step 6 (Advantages of hierarchy):**
- Manager explores in latent goal space (lower-dimensional)
- Worker solves simpler subproblems (reach goal states)
- Temporal abstraction: manager's effective horizon is $T/c$

---

## 2. Algorithm / Method

### 2.1 Pseudocode

```
Algorithm: Hierarchical RL for Vision (Option-Critic)
──────────────────────────────────────────────────────
Input: Visual environment, option set Ω
Output: Learned options and policy over options

Initialize: Option policies {π_ω}, termination functions {β_ω},
            policy over options μ, critic Q(s, ω)

for episode = 1 to M do
    s₀ ← env.reset()
    ω₀ ~ μ(·|s₀)  // Select initial option
    
    for t = 0 to T do
        // Execute intra-option policy
        aₜ ~ π_ω(·|sₜ)
        s_{t+1}, rₜ ← env.step(aₜ)
        
        // Check termination
        if Bernoulli(β_ω(s_{t+1})):
            ω_{t+1} ~ μ(·|s_{t+1})  // Select new option
        else:
            ω_{t+1} ← ω_t  // Continue current option
        end if
        
        // Update (Option-Critic gradients)
        Update Q(s, ω) via TD: Q ← r + γ U_ω(s')
        Update π_ω via intra-option policy gradient
        Update β_ω via termination gradient:
            ∇β = (Q(s', ω) - V(s')) · ∇β_ω(s')
        Update μ via policy gradient over options
    end for
end for
```

### 2.2 Complexity Analysis

- **Manager (per decision):** $O(|\theta_M|)$ every $c$ steps
- **Worker (per step):** $O(|\theta_W|)$
- **Total per episode:** $O(T \cdot |\theta_W| + (T/c) \cdot |\theta_M|)$
- **Option termination check:** $O(|\theta_\beta|)$ per step
- **Advantage:** Effective planning horizon is $T/c$ instead of $T$

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

$$\mathcal{M}_{SMDP} = (\mathcal{S}, \Omega, \tilde{P}, \tilde{R}, \tilde{\gamma})$$

- **State:** Visual features at current resolution/abstraction level
- **Options:** High-level visual strategies (scan, zoom, classify, segment)
- **Manager policy:** Selects which visual strategy to execute
- **Worker policy:** Executes the strategy at the primitive action level
- **Intrinsic reward:** Progress toward manager-specified visual goals

### 3.2 Why Hierarchical RL?

1. **Long-horizon visual tasks:** Complex image analysis requires many steps; hierarchy reduces effective horizon
2. **Transfer:** Learned options (visual primitives) transfer across tasks
3. **Exploration:** Manager explores in abstract goal space, more efficient than random primitive actions
4. **Interpretability:** Hierarchy provides natural decomposition of visual reasoning

---

## 4. Dataset

| Dataset | Task | Hierarchy | Description |
|---------|------|-----------|-------------|
| Visual Room Navigation | Navigate | Room→Object→Action | Multi-level planning |
| Montezuma's Revenge | Game | Strategy→Skill→Action | Hard exploration |
| CLEVR | Reasoning | Parse→Relate→Answer | Multi-hop QA |
| Habitat | Navigation | Goal→Path→Step | 3D embodied AI |

---

## 5. Key Equations Summary

| Equation | Description |
|----------|-------------|
| $\omega = (\mathcal{I}_\omega, \pi_\omega, \beta_\omega)$ | Option definition |
| $U_\omega(s') = (1-\beta_\omega)V_\omega(s') + \beta_\omega V^\mu(s')$ | Arrival value |
| $Q^\mu(s,\omega) = r(s,\omega) + \sum_{s'}\tilde{P}\tilde{\gamma}V^\mu(s')$ | Option Q-function |
| $r_t^{int} = \frac{1}{c}\sum_i\cos(s_t^M-s_{t-i}^M, g_{t-i})$ | Feudal intrinsic reward |
| $\nabla_{\beta_\omega} = (Q(s',\omega)-V(s'))\nabla\beta_\omega(s')$ | Termination gradient |

---

## 6. References

1. Sutton, R. S., Precup, D., & Singh, S. (1999). Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning. *Artificial Intelligence*, 112, 181-211.
2. Bacon, P.-L., Harb, J., & Precup, D. (2017). The option-critic architecture. *AAAI*.
3. Vezhnevets, A. S., et al. (2017). FeUdal Networks for hierarchical reinforcement learning. *ICML*.
4. Nachum, O., et al. (2018). Data-efficient hierarchical reinforcement learning. *NeurIPS*.
