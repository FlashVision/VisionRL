![Module Logo](../logo.png)

# Dynamic Programming

## Overview

Dynamic programming (DP) solves MDPs exactly when the model (transition and reward functions) is known. This document proves the convergence of value iteration via the contraction mapping theorem, proves that policy iteration converges in a finite number of steps, derives the linear programming formulation of MDPs, and establishes the computational complexity of each approach.

## Prerequisites

- MDP framework (Module 03.2)
- Bellman equations (Module 03.3)
- Linear algebra (matrix inversion)
- Linear programming basics

---

## 1. Mathematical Foundations

### 1.1 Core Definition — Dynamic Programming Principle

Bellman's principle of optimality: An optimal policy has the property that whatever the initial state and initial decision are, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision.

Mathematically: if $\pi^*$ is optimal and $\pi^*$ takes action $a$ in state $s$ reaching $s'$, then $\pi^*$ restricted to states reachable from $s'$ is optimal for the sub-MDP starting at $s'$.

### 1.2 Value Iteration Convergence Proof

**Algorithm:** $V_{k+1}(s) = \max_a[R(s,a) + \gamma\sum_{s'}T(s'|s,a)V_k(s')]$, equivalently $V_{k+1} = \mathcal{T}V_k$.

**Theorem:** Value iteration converges to $V^*$ from any initialization $V_0$.

**Proof:**

**Step 1:** We proved in Module 03.3 that $\mathcal{T}$ is a $\gamma$-contraction in $\ell_\infty$:

$$\|\mathcal{T}V_1 - \mathcal{T}V_2\|_\infty \leq \gamma\|V_1 - V_2\|_\infty$$

**Step 2:** The space $(\mathbb{R}^{|\mathcal{S}|}, \|\cdot\|_\infty)$ is a complete metric space.

**Step 3:** By the Banach fixed-point theorem, $\mathcal{T}^k V_0 \to V^*$ for any $V_0$.

**Step 4:** The convergence rate is geometric:

$$\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty \leq \frac{\gamma^k R_{\max}}{1-\gamma}$$

**Step 5:** To achieve $\varepsilon$-optimality:

$$\gamma^k \leq \varepsilon(1-\gamma)/R_{\max} \implies k \geq \frac{1}{1-\gamma}\log\frac{R_{\max}}{\varepsilon(1-\gamma)}$$

(using $\log(1/\gamma) \geq 1 - \gamma$ for $\gamma \in (0,1)$).
$\blacksquare$

### 1.3 Policy Iteration — Finite Convergence Proof

**Theorem:** Policy iteration converges to $\pi^*$ in at most $|\mathcal{A}|^{|\mathcal{S}|}$ iterations. In practice, convergence is much faster (often $O(|\mathcal{S}||\mathcal{A}|/(1-\gamma))$).

**Proof:**

**Step 1 — Policy evaluation is well-defined.** For a fixed policy $\pi$, the Bellman equation becomes linear:

$$V^\pi = \mathbf{R}^\pi + \gamma \mathbf{T}^\pi V^\pi$$

where $\mathbf{R}^\pi \in \mathbb{R}^{|\mathcal{S}|}$ with $R^\pi(s) = R(s, \pi(s))$ and $\mathbf{T}^\pi \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{S}|}$ with $T^\pi(s' | s) = T(s' | s, \pi(s))$.

Solution: $V^\pi = (\mathbf{I} - \gamma\mathbf{T}^\pi)^{-1}\mathbf{R}^\pi$.

The inverse exists because $\gamma\mathbf{T}^\pi$ has spectral radius at most $\gamma < 1$ (since $\mathbf{T}^\pi$ is a stochastic matrix with spectral radius 1).

**Step 2 — Strict improvement.** From the policy improvement theorem (Module 03.2), $V^{\pi_{k+1}} \geq V^{\pi_k}$ pointwise, with strict inequality unless $\pi_k$ is already optimal.

**Step 3 — Finite termination.** Since there are finitely many deterministic policies ($|\mathcal{A}|^{|\mathcal{S}|}$) and each policy is visited at most once (strict improvement), the algorithm terminates.

**Step 4 — Termination at optimality.** When $\pi_{k+1} = \pi_k$ (no improvement found):

$$V^{\pi_k}(s) = \max_a[R(s,a) + \gamma\sum_{s'} T(s'|s,a)V^{\pi_k}(s')]$$

This is the Bellman optimality equation, so $V^{\pi_k} = V^*$ and $\pi_k = \pi^*$.
$\blacksquare$

### 1.4 Linear Programming Formulation of MDPs

**Theorem:** The optimal value function $V^*$ is the solution to the following linear program:

$$\min_V \sum_{s \in \mathcal{S}} \alpha(s) V(s)$$

$$\text{s.t.} \quad V(s) \geq R(s, a) + \gamma \sum_{s'} T(s' \mid s, a) V(s') \quad \forall s, a$$

where $\alpha(s) > 0$ for all $s$ (any positive state weighting).

**Proof:**

**Step 1:** The constraints encode $V(s) \geq (\mathcal{T}_a V)(s)$ for all actions $a$, which means $V \geq \mathcal{T}V$ (componentwise).

**Step 2:** $V^*$ satisfies $V^* = \mathcal{T}V^*$, so $V^*$ is feasible.

**Step 3:** Any feasible $V$ satisfies $V \geq \mathcal{T}V$. Applying $\mathcal{T}$ (which is monotone):

$$V \geq \mathcal{T}V \geq \mathcal{T}^2 V \geq \cdots \to V^*$$

So $V \geq V^*$ componentwise.

**Step 4:** Since $V^*$ is the smallest feasible solution and the objective $\sum \alpha(s)V(s)$ with $\alpha > 0$ is minimized at the smallest feasible point, $V^*$ is optimal.
$\blacksquare$

**Dual LP:** The dual gives the state-action occupation measure:

$$\max \sum_{s,a} d(s,a) R(s,a)$$

$$\text{s.t.} \quad \sum_a d(s,a) = \alpha(s) + \gamma\sum_{s',a'} d(s',a') T(s|s',a') \quad \forall s$$

$$d(s,a) \geq 0 \quad \forall s, a$$

The optimal policy is recovered from the dual variables: $\pi^*(a|s) = d^*(s,a) / \sum_{a'} d^*(s,a')$.

### 1.5 Comparison of DP Methods

| Method | Per-iteration cost | Iterations | Total |
|--------|-------------------|------------|-------|
| Value Iteration | $O(|\mathcal{S}|^2|\mathcal{A}|)$ | $O(\frac{1}{1-\gamma}\log\frac{1}{\varepsilon})$ | $O(\frac{|\mathcal{S}|^2|\mathcal{A}|}{1-\gamma}\log\frac{1}{\varepsilon})$ |
| Policy Iteration | $O(|\mathcal{S}|^3 + |\mathcal{S}|^2|\mathcal{A}|)$ | $O(|\mathcal{A}|^{|\mathcal{S}|})$ worst | Typically faster |
| LP | $O((|\mathcal{S}||\mathcal{A}|)^{3.5})$ | 1 (LP solver) | Polynomial |

---

## 2. Algorithm / Method

### 2.1 Pseudocode: Value Iteration

```
Algorithm: Value_Iteration
Input: MDP (S, A, T, R, γ), tolerance ε
Output: V*, π*

1. V(s) = 0 ∀s
2. REPEAT:
     δ = 0
     For each s:
       v_old = V(s)
       V(s) = max_a [R(s,a) + γ Σ_{s'} T(s'|s,a)V(s')]
       δ = max(δ, |V(s) - v_old|)
     UNTIL δ < ε(1-γ)/(2γ)
3. π*(s) = argmax_a [R(s,a) + γ Σ_{s'} T(s'|s,a)V(s')]
4. Return V, π*
```

### 2.2 Complexity Analysis

- **Value iteration:** $O(|\mathcal{S}|^2|\mathcal{A}|)$ per iteration
- **Policy iteration (evaluation):** $O(|\mathcal{S}|^3)$ per iteration (matrix inverse)
- **LP formulation:** $|\mathcal{S}|$ variables, $|\mathcal{S}||\mathcal{A}|$ constraints

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

- **State:** $s_t \in \mathcal{S}$ (finite)
- **Action:** $a_t \in \mathcal{A}$ (finite)
- **Reward:** $R(s, a)$ — known
- **Transition:** $T(s' \mid s, a)$ — known

### 3.2 Why RL?

DP requires a known model. When $T$ and $R$ are unknown (the typical case for image processing), we must use model-free RL (MC, TD, Q-learning). DP provides the theoretical benchmark: any RL algorithm should approach the DP solution as data increases.

---

## 4. Dataset

- **Name:** GridWorld environments
- **Size:** Configurable grid sizes
- **Auto-download:**

```python
import numpy as np
grid_size = 5
n_states = grid_size ** 2
n_actions = 4  # up, down, left, right
T = np.zeros((n_states, n_actions, n_states))
R = np.zeros((n_states, n_actions))
```

---

## 5. Key Equations Summary

| Equation | Meaning |
|----------|---------|
| $V_{k+1} = \mathcal{T}V_k$ | Value iteration update |
| $\|V_k - V^*\| \leq \gamma^k R_{\max}/(1-\gamma)$ | Convergence rate |
| $V^\pi = (I - \gamma T^\pi)^{-1}R^\pi$ | Exact policy evaluation |
| $\min_V \sum \alpha(s)V(s)$ s.t. $V \geq \mathcal{T}_a V$ | LP formulation |
| $\pi^*(a\mid s) = d^*(s,a)/\sum_{a'}d^*(s,a')$ | Policy from dual LP |

---

## 6. References

- Bellman, R. *Dynamic Programming*, Princeton University Press, 1957.
- Puterman, M. L. *Markov Decision Processes*, Wiley, 1994.
- Bertsekas, D. P. *Dynamic Programming and Optimal Control*, 4th ed., Athena Scientific, 2017.
- d'Epenoux, F. "A Probabilistic Production and Inventory Problem," *Management Science*, 1963.
