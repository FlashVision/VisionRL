![Module Logo](../logo.png)

# Markov Decision Process (MDP)

## Overview

The Markov Decision Process is the mathematical framework for sequential decision-making under uncertainty. This document gives the formal MDP definition, proves the existence of an optimal policy via the Bellman optimality operator, provides a complete proof of policy iteration convergence, and develops the interpretation of the discount factor from both mathematical and economic perspectives.

## Prerequisites

- Probability theory (Module 03.1)
- Linear algebra (matrix norms, spectral radius)
- Real analysis (fixed-point theorems, contraction mappings)

---

## 1. Mathematical Foundations

### 1.1 Core Definition — Formal MDP

An MDP is a 5-tuple $(\mathcal{S}, \mathcal{A}, T, R, \gamma)$:

- **State space:** $\mathcal{S}$ — finite or countable set of states
- **Action space:** $\mathcal{A}$ — finite set of actions (may be state-dependent: $\mathcal{A}(s)$)
- **Transition function:** $T: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0, 1]$ where $T(s' \mid s, a) = P(s_{t+1} = s' \mid s_t = s, a_t = a)$
- **Reward function:** $R: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ (expected immediate reward)
- **Discount factor:** $\gamma \in [0, 1)$

A **policy** $\pi: \mathcal{S} \to \Delta(\mathcal{A})$ maps states to probability distributions over actions. A deterministic policy satisfies $\pi(a \mid s) \in \{0, 1\}$.

The **value function** under policy $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi\!\left[\sum_{t=0}^{\infty} \gamma^t r_t \;\middle|\; s_0 = s\right]$$

The **action-value function:**

$$Q^\pi(s, a) = \mathbb{E}_\pi\!\left[\sum_{t=0}^{\infty} \gamma^t r_t \;\middle|\; s_0 = s, a_0 = a\right]$$

### 1.2 Proof of Existence of an Optimal Policy

**Theorem:** For any finite MDP with $\gamma < 1$, there exists a deterministic stationary policy $\pi^*$ that is simultaneously optimal for all states:

$$V^{\pi^*}(s) \geq V^\pi(s) \quad \forall s \in \mathcal{S}, \; \forall \pi$$

**Proof:**

**Step 1:** Define the Bellman optimality operator $\mathcal{T}: \mathbb{R}^{|\mathcal{S}|} \to \mathbb{R}^{|\mathcal{S}|}$:

$$(\mathcal{T}V)(s) = \max_{a \in \mathcal{A}} \left[R(s, a) + \gamma \sum_{s'} T(s' \mid s, a) V(s')\right]$$

**Step 2:** Show $\mathcal{T}$ is a $\gamma$-contraction in the $\ell_\infty$ norm. For any $V_1, V_2$:

$$|(\mathcal{T}V_1)(s) - (\mathcal{T}V_2)(s)| = \left|\max_a[R + \gamma \sum_{s'} T \cdot V_1(s')] - \max_a[R + \gamma \sum_{s'} T \cdot V_2(s')]\right|$$

**Step 3:** Using the inequality $|\max_a f(a) - \max_a g(a)| \leq \max_a |f(a) - g(a)|$:

$$\leq \max_a \left|\gamma \sum_{s'} T(s' \mid s, a)(V_1(s') - V_2(s'))\right| \leq \gamma \max_a \sum_{s'} T(s' \mid s, a)|V_1(s') - V_2(s')|$$

$$\leq \gamma \|V_1 - V_2\|_\infty \max_a \underbrace{\sum_{s'} T(s' \mid s, a)}_{= 1} = \gamma \|V_1 - V_2\|_\infty$$

**Step 4:** Therefore $\|\mathcal{T}V_1 - \mathcal{T}V_2\|_\infty \leq \gamma \|V_1 - V_2\|_\infty$, confirming $\mathcal{T}$ is a $\gamma$-contraction.

**Step 5:** By the Banach fixed-point theorem, $\mathcal{T}$ has a unique fixed point $V^*$:

$$V^*(s) = \max_a \left[R(s, a) + \gamma \sum_{s'} T(s' \mid s, a) V^*(s')\right]$$

**Step 6:** Define the greedy policy:

$$\pi^*(s) = \arg\max_a \left[R(s, a) + \gamma \sum_{s'} T(s' \mid s, a) V^*(s')\right]$$

**Step 7:** Show $V^{\pi^*} = V^*$. Since $\pi^*$ is greedy w.r.t. $V^*$:

$$V^*(s) = R(s, \pi^*(s)) + \gamma \sum_{s'} T(s' \mid s, \pi^*(s)) V^*(s')$$

This is the Bellman equation for policy $\pi^*$, whose unique solution is $V^{\pi^*}$. Hence $V^{\pi^*} = V^*$.

**Step 8:** Show optimality. For any policy $\pi$ and any state $s$:

$$V^\pi(s) = R(s, \pi(s)) + \gamma \sum_{s'} T(s' \mid s, \pi(s)) V^\pi(s') \leq \max_a[\cdots] = (\mathcal{T}V^\pi)(s)$$

Iterating: $V^\pi \leq \mathcal{T}V^\pi \leq \mathcal{T}^2 V^\pi \leq \cdots \to V^*$. Therefore $V^\pi \leq V^* = V^{\pi^*}$.

**Result:** $\pi^*$ exists, is deterministic and stationary, and achieves $V^* = V^{\pi^*} \geq V^\pi$ for all $\pi$.
$\blacksquare$

### 1.3 Proof of Policy Iteration Convergence

**Algorithm — Policy Iteration:**

1. **Initialize:** Choose arbitrary policy $\pi_0$.
2. **Policy Evaluation:** Solve $V^{\pi_k}(s) = R(s, \pi_k(s)) + \gamma \sum_{s'} T(s' \mid s, \pi_k(s)) V^{\pi_k}(s')$ for all $s$.
3. **Policy Improvement:** $\pi_{k+1}(s) = \arg\max_a [R(s,a) + \gamma \sum_{s'} T(s' \mid s, a) V^{\pi_k}(s')]$.
4. If $\pi_{k+1} = \pi_k$, stop. Else, go to step 2.

**Theorem:** Policy iteration converges to $\pi^*$ in at most $|\mathcal{A}|^{|\mathcal{S}|}$ iterations (the number of deterministic policies).

**Proof:**

**Step 1 — Policy Improvement Theorem:** We show $V^{\pi_{k+1}} \geq V^{\pi_k}$ pointwise.

For any state $s$:

$$V^{\pi_k}(s) = R(s, \pi_k(s)) + \gamma \sum_{s'} T V^{\pi_k}(s') \leq \max_a [R(s,a) + \gamma \sum_{s'} T V^{\pi_k}(s')]$$

$$= R(s, \pi_{k+1}(s)) + \gamma \sum_{s'} T(s' \mid s, \pi_{k+1}(s)) V^{\pi_k}(s')$$

**Step 2:** Iterate the inequality:

$$V^{\pi_k}(s) \leq R(s, \pi_{k+1}) + \gamma \sum_{s'} T \cdot V^{\pi_k}(s')$$

$$\leq R(s, \pi_{k+1}) + \gamma \sum_{s'} T \left[R(s', \pi_{k+1}) + \gamma \sum_{s''} T \cdot V^{\pi_k}(s'')\right]$$

$$\leq \cdots \leq \mathbb{E}_{\pi_{k+1}}\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s\right] = V^{\pi_{k+1}}(s)$$

**Step 3:** The sequence $V^{\pi_0} \leq V^{\pi_1} \leq V^{\pi_2} \leq \cdots$ is monotonically non-decreasing.

**Step 4:** Since there are finitely many deterministic policies ($|\mathcal{A}|^{|\mathcal{S}|}$), and each improvement is strict (unless $\pi_k$ is already optimal), the algorithm must terminate in finite steps at $\pi^*$.

**Result:** Policy iteration converges to the optimal policy in a finite number of iterations.
$\blacksquare$

### 1.4 Discount Factor Interpretation

**Mathematical role:** $\gamma < 1$ ensures the infinite sum $\sum_{t=0}^\infty \gamma^t r_t$ converges:

$$|V^\pi(s)| \leq \sum_{t=0}^{\infty} \gamma^t |r_t| \leq \frac{R_{\max}}{1 - \gamma}$$

**Economic interpretation:** $\gamma$ represents the time value of rewards. A reward of $r$ received $t$ steps in the future has present value $\gamma^t r$. The effective planning horizon is approximately:

$$T_{\text{eff}} = \frac{1}{1 - \gamma}$$

For $\gamma = 0.99$: $T_{\text{eff}} = 100$ steps. For $\gamma = 0.999$: $T_{\text{eff}} = 1000$ steps.

**Survival interpretation:** $\gamma$ can be interpreted as the probability of the process continuing at each step. Then $\gamma^t$ is the probability of surviving to step $t$, and:

$$V^\pi(s) = \mathbb{E}\left[\sum_{t=0}^{T} r_t \mid s_0 = s\right]$$

where $T \sim \text{Geometric}(1 - \gamma)$ is a random stopping time.

---

## 2. Algorithm / Method

### 2.1 Pseudocode: Policy Iteration

```
Algorithm: Policy_Iteration
Input: MDP (S, A, T, R, γ)
Output: Optimal policy π*, value V*

1. INITIALIZE: π(s) = arbitrary action for all s
2. REPEAT:
   a. POLICY EVALUATION: Solve linear system
      V^π = R^π + γ T^π V^π
      V^π = (I - γ T^π)^{-1} R^π
   b. POLICY IMPROVEMENT:
      For each s: π_new(s) = argmax_a [R(s,a) + γ Σ_{s'} T(s'|s,a)V^π(s')]
   c. If π_new = π: STOP
   d. π ← π_new
3. Return π, V^π
```

### 2.2 Complexity Analysis

- **Policy evaluation (exact):** $O(|\mathcal{S}|^3)$ per iteration (solving linear system)
- **Policy improvement:** $O(|\mathcal{S}||\mathcal{A}||\mathcal{S}|)$ per iteration
- **Number of iterations:** At most $|\mathcal{A}|^{|\mathcal{S}|}$ (typically much fewer in practice)

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation for Image Processing

- **State:** $s_t = \mathbf{I}_t$ — the current image
- **Action:** $a_t$ — an image processing operation
- **Reward:** $r_t = \mathcal{Q}(\mathbf{I}_{t+1}) - \mathcal{Q}(\mathbf{I}_t)$ — quality improvement
- **Transition:** $T(s_{t+1} \mid s_t, a_t)$ — deterministic or stochastic effect of the operation

### 3.2 Why RL?

The MDP framework formalizes the sequential nature of image processing: each operation changes the image, and the agent must plan ahead. The existence of an optimal policy (proven above) guarantees that there is a best processing pipeline for each image — RL algorithms find it.

---

## 4. Dataset

- **Name:** Synthetic MDPs (GridWorld) and image-based environments
- **Size:** Configurable
- **Auto-download:**

```python
import numpy as np
n_states, n_actions = 16, 4
T = np.random.dirichlet(np.ones(n_states), size=(n_states, n_actions))
R = np.random.randn(n_states, n_actions)
```

---

## 5. Key Equations Summary

| Equation | Meaning |
|----------|---------|
| $V^\pi(s) = \mathbb{E}_\pi[\sum_t \gamma^t r_t \mid s_0 = s]$ | Value function definition |
| $V^*(s) = \max_a[R(s,a) + \gamma\sum_{s'}T(s'\mid s,a)V^*(s')]$ | Bellman optimality equation |
| $\|\mathcal{T}V_1 - \mathcal{T}V_2\|_\infty \leq \gamma\|V_1 - V_2\|_\infty$ | Contraction property |
| $V^{\pi_{k+1}}(s) \geq V^{\pi_k}(s)$ | Policy improvement theorem |
| $T_{\text{eff}} = 1/(1-\gamma)$ | Effective planning horizon |

---

## 6. References

- Puterman, M. L. *Markov Decision Processes: Discrete Stochastic Dynamic Programming*, Wiley, 1994.
- Sutton, R. S. & Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018.
- Bertsekas, D. P. *Dynamic Programming and Optimal Control*, 4th ed., Athena Scientific, 2017.
- Howard, R. A. *Dynamic Programming and Markov Processes*, MIT Press, 1960.
