![Module Logo](../logo.png)

# Q-Learning

## Overview

Q-learning is the most important off-policy temporal difference algorithm, learning the optimal action-value function $Q^*$ directly without requiring a model or on-policy data. This document provides a convergence proof via stochastic approximation theory, derives sample complexity bounds, analyzes the maximization bias problem, and proves the correction provided by Double Q-learning.

## Prerequisites

- TD learning (Module 04.4)
- Bellman optimality equations (Module 03.3)
- Stochastic approximation theory

---

## 1. Mathematical Foundations

### 1.1 Core Definition — Q-Learning Update

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha_t\left[r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)\right]$$

Key difference from SARSA: Q-learning uses $\max_{a'} Q(s_{t+1}, a')$ (off-policy, greedy w.r.t. $Q$) instead of $Q(s_{t+1}, a_{t+1})$ (on-policy, follows $\pi$).

### 1.2 Q-Learning Convergence Proof

**Theorem (Watkins & Dayan, 1992; Tsitsiklis, 1994):** Under the following conditions, Q-learning converges to $Q^*$ with probability 1:

1. All state-action pairs are visited infinitely often.
2. Step sizes satisfy: $\sum_t \alpha_t(s,a) = \infty$ and $\sum_t \alpha_t^2(s,a) < \infty$ for all $(s,a)$.

**Proof:**

**Step 1:** Write Q-learning as a stochastic approximation of the Bellman optimality operator. Define:

$$(\mathcal{T}Q)(s, a) = R(s, a) + \gamma \sum_{s'} T(s' \mid s, a) \max_{a'} Q(s', a')$$

**Step 2:** The Q-learning update can be written as:

$$Q_{t+1}(s_t, a_t) = (1 - \alpha_t)Q_t(s_t, a_t) + \alpha_t\left[(\mathcal{T}Q_t)(s_t, a_t) + w_t\right]$$

where $w_t = [r_t + \gamma\max_{a'} Q_t(s_{t+1}, a')] - (\mathcal{T}Q_t)(s_t, a_t)$ is zero-mean noise:

$$\mathbb{E}[w_t \mid s_t, a_t, Q_t] = 0$$

**Step 3:** Define the error $\Delta_t(s, a) = Q_t(s, a) - Q^*(s, a)$. Since $Q^* = \mathcal{T}Q^*$:

$$\Delta_{t+1}(s_t, a_t) = (1 - \alpha_t)\Delta_t(s_t, a_t) + \alpha_t[(\mathcal{T}Q_t)(s_t, a_t) - Q^*(s_t, a_t)] + \alpha_t w_t$$

**Step 4:** The contraction property of $\mathcal{T}$ gives:

$$|(\mathcal{T}Q_t)(s,a) - Q^*(s,a)| = |(\mathcal{T}Q_t)(s,a) - (\mathcal{T}Q^*)(s,a)| \leq \gamma\|Q_t - Q^*\|_\infty$$

**Step 5:** This bounds the deterministic part of the update:

$$|\mathbb{E}[\Delta_{t+1}(s_t, a_t) \mid Q_t]| \leq (1 - \alpha_t)|\Delta_t(s_t, a_t)| + \alpha_t\gamma\|\Delta_t\|_\infty$$

**Step 6:** Apply the stochastic approximation theorem for contractive operators (Jaakkola et al., 1994). The conditions are:

(a) $\mathcal{T}$ is a contraction with modulus $\gamma < 1$. $\checkmark$

(b) The noise $w_t$ has bounded conditional variance: $\mathbb{E}[w_t^2 \mid \mathcal{F}_t] \leq C(1 + \|Q_t\|_\infty^2)$. $\checkmark$ (since rewards are bounded).

(c) Step size conditions hold. $\checkmark$ (by assumption).

(d) All state-action pairs visited infinitely often. $\checkmark$ (by assumption).

**Step 7:** By the theorem, $Q_t \to Q^*$ a.s.

**Result:**

$$\boxed{Q_t(s, a) \xrightarrow{\text{a.s.}} Q^*(s, a) \quad \forall (s, a)}$$
$\blacksquare$

### 1.3 Sample Complexity Bounds

**Theorem (Even-Dar & Mansour, 2003):** With polynomial learning rate $\alpha_t = 1/(t + 1)^w$ where $w \in (1/2, 1)$, Q-learning with $\varepsilon$-greedy exploration finds an $\varepsilon$-optimal policy using:

$$\tilde{O}\!\left(\frac{|\mathcal{S}||\mathcal{A}|}{\varepsilon^2(1-\gamma)^5}\right) \text{ samples}$$

The $(1-\gamma)^{-5}$ dependence highlights the curse of long horizons.

**PAC bound (Strehl et al., 2006):** Model-based methods like R-MAX achieve:

$$\tilde{O}\!\left(\frac{|\mathcal{S}|^2|\mathcal{A}|}{\varepsilon^2(1-\gamma)^3}\right)$$

which is better in $1/(1-\gamma)$ but worse in $|\mathcal{S}|$.

### 1.4 Maximization Bias Analysis

**Problem:** Q-learning uses $\max_{a'} Q(s', a')$ as an estimate of $\max_{a'} Q^*(s', a')$. This introduces positive bias:

$$\mathbb{E}\left[\max_{a'} Q(s', a')\right] \geq \max_{a'} \mathbb{E}[Q(s', a')]$$

by Jensen's inequality (max is convex).

**Formal analysis:**

**Step 1:** Let $Q(s', a_i) = Q^*(s', a_i) + \epsilon_i$ where $\epsilon_i$ are zero-mean estimation errors.

**Step 2:** $\max_i (Q^* + \epsilon_i) \geq \max_i Q^*_i + \min_i \epsilon_i$... but more precisely:

$$\mathbb{E}\left[\max_i(Q^*_i + \epsilon_i)\right] = Q^*_{\text{best}} + \mathbb{E}\left[\max_i(\epsilon_i + \Delta_i)\right]$$

where $\Delta_i = Q^*_i - Q^*_{\text{best}} \leq 0$.

**Step 3:** If all actions are equally valued ($Q^*_i = Q^*$ for all $i$), and $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$:

$$\mathbb{E}[\max_i \epsilon_i] = \sigma \cdot \mathbb{E}[\max_i Z_i] \approx \sigma\sqrt{2\log K}$$

where $K$ is the number of actions. This is the order statistics result for Gaussian maxima.

**Step 4:** The bias $\sigma\sqrt{2\log K}$ can be significant when $Q$ estimates are noisy (early training) and the action space is large.

### 1.5 Double Q-Learning Correction Proof

**Algorithm:** Maintain two independent Q-functions $Q^A$ and $Q^B$. Update alternately:

$$Q^A(s, a) \leftarrow Q^A(s, a) + \alpha\left[r + \gamma Q^B(s', \arg\max_{a'} Q^A(s', a')) - Q^A(s, a)\right]$$

**Theorem:** Double Q-learning eliminates the maximization bias.

**Proof:**

**Step 1:** The selection $a^* = \arg\max_{a'} Q^A(s', a')$ is determined by $Q^A$.

**Step 2:** The evaluation $Q^B(s', a^*)$ is computed using the independent $Q^B$.

**Step 3:** Since $Q^A$ and $Q^B$ are learned from independent samples:

$$\mathbb{E}[Q^B(s', a^*)] = \mathbb{E}_{a^* \sim Q^A}[Q^B(s', a^*)]$$

**Step 4:** If $Q^B$ is an unbiased estimator of $Q^*$, then:

$$\mathbb{E}[Q^B(s', a^*)] = \mathbb{E}_{a^*}[Q^*(s', a^*)] \leq \max_{a'} Q^*(s', a')$$

The inequality follows because $a^*$ is not necessarily the true optimal action (it was selected using the noisy $Q^A$), so $Q^*(s', a^*)$ is on average less than the true maximum.

**Step 5:** In expectation: $\mathbb{E}[Q^B(s', \arg\max Q^A)] \leq \max_{a'} Q^*(s', a')$, eliminating the upward bias.

**Result:** Double Q-learning provides an unbiased (or slightly underestimating) target, correcting the maximization bias.
$\blacksquare$

---

## 2. Algorithm / Method

### 2.1 Pseudocode: Q-Learning

```
Algorithm: Q_Learning
Input: step size α, discount γ, exploration rate ε, episodes N
Output: Q(s,a)

1. Q(s,a) = 0 ∀s,a
2. For episode = 1 to N:
     s = initial state
     REPEAT:
       a = ε-greedy(Q, s)
       Take action a, observe r, s'
       Q(s,a) = Q(s,a) + α[r + γ·max_{a'} Q(s',a') - Q(s,a)]
       s = s'
     UNTIL s is terminal
3. Return Q
```

### 2.2 Complexity Analysis

- **Per step:** $O(|\mathcal{A}|)$ — compute max over actions
- **Sample complexity:** $\tilde{O}(|\mathcal{S}||\mathcal{A}|/(\varepsilon^2(1-\gamma)^5))$
- **Space:** $O(|\mathcal{S}||\mathcal{A}|)$ for the Q-table

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

- **State:** $s_t$ — environment state
- **Action:** $a_t$ — selected by exploration policy (e.g., $\varepsilon$-greedy)
- **Reward:** $r_t$ — immediate reward
- **Transition:** Unknown (model-free, off-policy)

### 3.2 Why RL?

Q-learning is off-policy: the behavior policy (exploratory) differs from the target policy ($\arg\max Q$). This enables experience replay (Module 06.1) and learning from demonstrations. It is the foundation for DQN and all value-based deep RL.

---

## 4. Dataset

- **Name:** GridWorld / FrozenLake / Taxi environments
- **Size:** Generated through interaction
- **Auto-download:**

```python
import gymnasium as gym
env = gym.make('FrozenLake-v1', is_slippery=True)
obs, info = env.reset()
```

---

## 5. Key Equations Summary

| Equation | Meaning |
|----------|---------|
| $Q(s,a) \leftarrow Q + \alpha[r + \gamma\max_{a'}Q(s',a') - Q(s,a)]$ | Q-learning update |
| $Q_t \xrightarrow{\text{a.s.}} Q^*$ | Convergence guarantee |
| $\mathbb{E}[\max_i \epsilon_i] \approx \sigma\sqrt{2\log K}$ | Maximization bias |
| $Q^A(s,a) \leftarrow Q^A + \alpha[r + \gamma Q^B(s', \arg\max Q^A) - Q^A]$ | Double Q-learning |
| $\tilde{O}(|\mathcal{S}||\mathcal{A}|/\varepsilon^2(1-\gamma)^5)$ | Sample complexity |

---

## 6. References

- Watkins, C. J. C. H. & Dayan, P. "Q-Learning," *Machine Learning*, 8:279–292, 1992.
- Hasselt, H. van. "Double Q-learning," *NeurIPS*, 2010.
- Even-Dar, E. & Mansour, Y. "Learning Rates for Q-Learning," *JMLR*, 5:1–25, 2003.
- Jaakkola, T., Jordan, M. I., & Singh, S. P. "On the Convergence of Stochastic Iterative Dynamic Programming Algorithms," *Neural Computation*, 1994.
