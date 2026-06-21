![Module Logo](../logo.png)

# Value Functions

## Overview

Value functions are the core objects in reinforcement learning, quantifying the expected long-term return from states and state-action pairs. This document proves the relationship between state-value and action-value functions, derives the advantage function and its properties, develops the soft value function for maximum entropy RL, and establishes the Legendre–Fenchel duality connection between value functions and policies.

## Prerequisites

- Bellman equations (Module 03.3)
- Probability theory (Module 03.1)
- Convex analysis basics (conjugate functions)

---

## 1. Mathematical Foundations

### 1.1 Core Definitions

**State-value function:**

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t \;\middle|\; s_0 = s\right]$$

**Action-value function:**

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t \;\middle|\; s_0 = s, \, a_0 = a\right]$$

### 1.2 Proof of the $V$–$Q$ Relationship

**Theorem:**

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \, Q^\pi(s, a) = \mathbb{E}_{a \sim \pi(\cdot|s)}[Q^\pi(s, a)]$$

$$Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} T(s' \mid s, a) V^\pi(s')$$

**Proof of the first identity:**

**Step 1:** From the definition:

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty}\gamma^t r_t \;\middle|\; s_0 = s\right]$$

**Step 2:** Condition on the first action using the tower property:

$$= \mathbb{E}_{a_0 \sim \pi(\cdot|s)}\left[\mathbb{E}_\pi\left[\sum_{t=0}^{\infty}\gamma^t r_t \;\middle|\; s_0 = s, a_0\right]\right]$$

**Step 3:** The inner expectation is exactly $Q^\pi(s, a_0)$ by definition:

$$= \mathbb{E}_{a_0 \sim \pi(\cdot|s)}[Q^\pi(s, a_0)] = \sum_a \pi(a \mid s) Q^\pi(s, a)$$
$\blacksquare$

**Proof of the second identity:**

**Step 1:** From the $Q$ definition:

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[r_0 + \sum_{t=1}^{\infty}\gamma^t r_t \;\middle|\; s_0 = s, a_0 = a\right]$$

**Step 2:** $\mathbb{E}[r_0 \mid s_0 = s, a_0 = a] = R(s, a)$.

**Step 3:** For the remaining sum, condition on $s_1$:

$$\gamma \mathbb{E}\left[\sum_{t=1}^{\infty}\gamma^{t-1}r_t \;\middle|\; s_0 = s, a_0 = a\right] = \gamma \sum_{s'} T(s' \mid s, a) V^\pi(s')$$

by the Markov property and definition of $V^\pi$.
$\blacksquare$

### 1.3 Advantage Function Derivation

**Definition:**

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

**Properties:**

**Property 1 — Zero expected advantage:**

$$\mathbb{E}_{a \sim \pi}[A^\pi(s, a)] = \mathbb{E}_{a \sim \pi}[Q^\pi(s, a)] - V^\pi(s) = V^\pi(s) - V^\pi(s) = 0$$

**Property 2 — Sign interpretation:** $A^\pi(s, a) > 0$ means action $a$ is better than the average action under $\pi$ in state $s$; $A^\pi(s, a) < 0$ means it is worse.

**Property 3 — Policy improvement direction:** The policy gradient theorem (Module 06.2) shows that the gradient of $J(\pi)$ involves $A^\pi$:

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi\left[\nabla_\theta \log \pi_\theta(a \mid s) \, A^\pi(s, a)\right]$$

Actions with positive advantage are reinforced; those with negative advantage are discouraged. The advantage function provides a natural baseline that reduces variance compared to using $Q^\pi$ directly.

### 1.4 Soft Value Function Derivation (Maximum Entropy RL)

In maximum entropy RL, the agent maximizes expected return plus entropy:

$$J_{\text{soft}}(\pi) = \sum_{t=0}^{\infty} \gamma^t \, \mathbb{E}\left[r_t + \alpha \mathcal{H}(\pi(\cdot \mid s_t))\right]$$

where $\alpha > 0$ is the temperature parameter and $\mathcal{H}(\pi(\cdot \mid s)) = -\sum_a \pi(a \mid s) \log \pi(a \mid s)$.

**Step 1 — Soft Bellman equation.** The soft value function satisfies:

$$V^{\pi}_{\text{soft}}(s) = \sum_a \pi(a \mid s)\left[Q^\pi_{\text{soft}}(s, a) - \alpha \log \pi(a \mid s)\right]$$

**Step 2 — Optimal soft policy.** Maximizing the RHS over $\pi$ with the constraint $\sum_a \pi(a \mid s) = 1$:

Form the Lagrangian:

$$\mathcal{L} = \sum_a \pi_a [Q_a - \alpha \log \pi_a] - \lambda(\sum_a \pi_a - 1)$$

**Step 3:** Take derivative with respect to $\pi_a$:

$$\frac{\partial \mathcal{L}}{\partial \pi_a} = Q_a - \alpha \log \pi_a - \alpha - \lambda = 0$$

$$\log \pi_a = \frac{Q_a - \alpha - \lambda}{\alpha} \implies \pi_a^* = \frac{\exp(Q_a / \alpha)}{Z(s)}$$

where $Z(s) = \sum_a \exp(Q_a / \alpha)$ is the partition function (determined by the constraint).

**Step 4 — Soft value function.** Substituting the optimal policy:

$$V^*_{\text{soft}}(s) = \alpha \log \sum_a \exp\!\left(\frac{Q^*_{\text{soft}}(s, a)}{\alpha}\right) = \alpha \log Z(s)$$

This is the **log-sum-exp** (soft maximum) operator.

**Result:**

$$\boxed{V^*_{\text{soft}}(s) = \alpha \log \sum_a \exp(Q^*(s,a)/\alpha), \quad \pi^*(a \mid s) = \frac{\exp(Q^*(s,a)/\alpha)}{\sum_{a'}\exp(Q^*(s,a')/\alpha)}}$$
$\blacksquare$

**Intuition:** As $\alpha \to 0$, the soft-max approaches the hard max (standard RL). As $\alpha \to \infty$, the policy becomes uniform (maximum entropy). The temperature $\alpha$ controls the exploration–exploitation tradeoff.

### 1.5 Legendre–Fenchel Duality Connection

The soft value function is connected to the entropy-regularized optimization via convex conjugation.

**Step 1:** The convex conjugate (Legendre–Fenchel transform) of $f(\pi) = \sum_a \pi_a \log \pi_a$ (negative entropy) is:

$$f^*(\mathbf{q}) = \sup_{\pi \in \Delta} \left[\sum_a q_a \pi_a - \sum_a \pi_a \log \pi_a\right] = \log \sum_a \exp(q_a)$$

**Step 2:** This is exactly the log-sum-exp function! Therefore:

$$V^*_{\text{soft}}(s) = \alpha \cdot f^*(Q^*(s, \cdot) / \alpha)$$

**Step 3:** The optimal policy $\pi^* = \nabla f^*(\mathbf{q})$ (gradient of the conjugate), which for log-sum-exp gives the softmax:

$$\pi^*(a \mid s) = \frac{\partial}{\partial q_a} \log \sum_{a'} e^{q_{a'}} = \text{softmax}(Q^*/\alpha)$$

**Interpretation:** The duality between entropy regularization and softmax policies is a manifestation of Legendre–Fenchel duality. This provides a principled way to derive exploration policies from value functions.
$\blacksquare$

---

## 2. Algorithm / Method

### 2.1 Pseudocode: Value Function Computation

```
Algorithm: Iterative_Policy_Evaluation
Input: Policy π, MDP (S, A, T, R, γ), tolerance ε
Output: V^π

1. INITIALIZE: V(s) = 0 for all s
2. REPEAT:
     δ = 0
     For each s:
       v = V(s)
       V(s) = Σ_a π(a|s) [R(s,a) + γ Σ_{s'} T(s'|s,a) V(s')]
       δ = max(δ, |v - V(s)|)
     UNTIL δ < ε
3. Compute Q(s,a) = R(s,a) + γ Σ_{s'} T(s'|s,a) V(s')
4. Compute A(s,a) = Q(s,a) - V(s)
5. Return V, Q, A
```

### 2.2 Complexity Analysis

- **Per iteration:** $O(|\mathcal{S}|^2|\mathcal{A}|)$
- **Convergence:** Same as value iteration — $O(\frac{1}{1-\gamma}\log\frac{1}{\varepsilon})$ iterations
- **Space:** $O(|\mathcal{S}||\mathcal{A}|)$ for $Q$, $O(|\mathcal{S}|)$ for $V$

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

- **State:** $s_t$ — current environment observation
- **Action:** $a_t$ — agent's decision
- **Reward:** $r_t$ — immediate scalar feedback
- **Transition:** $s_{t+1} \sim T(\cdot \mid s_t, a_t)$

### 3.2 Why RL?

Value functions are the core objects that RL algorithms estimate. Understanding the $V$–$Q$–$A$ relationships is essential for implementing actor-critic methods (Module 06.3), where the advantage function reduces policy gradient variance, and for SAC/soft RL methods that use the soft value function for principled exploration.

---

## 4. Dataset

- **Name:** Synthetic MDPs
- **Size:** Configurable
- **Auto-download:**

```python
import numpy as np
n_states, n_actions = 10, 3
gamma = 0.99
V = np.zeros(n_states)
Q = np.zeros((n_states, n_actions))
```

---

## 5. Key Equations Summary

| Equation | Meaning |
|----------|---------|
| $V^\pi(s) = \sum_a \pi(a\mid s)Q^\pi(s,a)$ | $V$–$Q$ relationship |
| $Q^\pi(s,a) = R(s,a) + \gamma\sum_{s'}TV^\pi(s')$ | $Q$–$V$ relationship |
| $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ | Advantage function |
| $\mathbb{E}_\pi[A^\pi(s,a)] = 0$ | Zero-mean advantage |
| $V^*_{\text{soft}} = \alpha\log\sum_a e^{Q^*/\alpha}$ | Soft value function |

---

## 6. References

- Sutton, R. S. & Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018, Ch. 3.
- Haarnoja, T. et al. "Soft Actor-Critic," *ICML*, 2018.
- Nachman, O. et al. "Bridging the Gap Between Value and Policy Based RL," *NeurIPS*, 2017.
- Rockafellar, R. T. *Convex Analysis*, Princeton University Press, 1970.
