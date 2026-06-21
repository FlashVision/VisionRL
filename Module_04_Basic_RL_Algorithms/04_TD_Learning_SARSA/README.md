![Module Logo](../logo.png)

# TD Learning and SARSA

## Overview

Temporal Difference (TD) learning combines the model-free property of Monte Carlo with the bootstrapping property of dynamic programming. This document provides a convergence proof sketch for TD(0), proves the equivalence of forward and backward views in TD($\lambda$), derives eligibility traces from first principles, and analyzes SARSA as an on-policy TD control algorithm.

## Prerequisites

- Monte Carlo methods (Module 04.3)
- Bellman equations (Module 03.3)
- Stochastic approximation theory basics

---

## 1. Mathematical Foundations

### 1.1 Core Definition — TD(0) Update

The TD(0) update rule for policy evaluation:

$$V(s_t) \leftarrow V(s_t) + \alpha\left[r_t + \gamma V(s_{t+1}) - V(s_t)\right]$$

The term $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the **TD error**.

### 1.2 TD(0) Convergence Proof Sketch

**Theorem (Dayan, 1992; Tsitsiklis, 1994):** Under appropriate conditions on the step size $\alpha_t$, TD(0) converges to $V^\pi$ almost surely.

**Proof sketch:**

**Step 1:** TD(0) is a stochastic approximation algorithm of the form:

$$V_{t+1}(s) = V_t(s) + \alpha_t(s)\left[(\mathcal{T}^\pi V_t)(s) - V_t(s) + w_t\right]$$

where $w_t = [r_t + \gamma V_t(s_{t+1})] - (\mathcal{T}^\pi V_t)(s_t)$ is zero-mean noise (conditional on $s_t$).

**Step 2:** The expected update direction is:

$$\mathbb{E}[\delta_t \mid s_t = s] = \mathbb{E}[r_t + \gamma V(s_{t+1}) \mid s_t = s] - V(s) = (\mathcal{T}^\pi V)(s) - V(s)$$

This points toward the Bellman fixed point $V^\pi$.

**Step 3:** Convergence requires the Robbins–Monro conditions on step sizes:

$$\sum_{t=0}^{\infty} \alpha_t(s) = \infty \quad \text{and} \quad \sum_{t=0}^{\infty} \alpha_t^2(s) < \infty \quad \forall s$$

The first condition ensures the updates can reach $V^\pi$ from any initial point; the second ensures the noise is eventually averaged out.

**Step 4:** Since $\mathcal{T}^\pi$ is a $\gamma$-contraction (as proven in Module 03.3), the ODE method (Borkar & Meyn, 2000) establishes that the stochastic iterates track the ODE:

$$\dot{V}(s) = (\mathcal{T}^\pi V)(s) - V(s)$$

whose unique stable equilibrium is $V^\pi$.

**Step 5:** By the stochastic approximation theorem (with the contraction property ensuring stability), $V_t \to V^\pi$ almost surely.
$\blacksquare$

### 1.3 TD($\lambda$) — Forward and Backward Views

**Forward view — $\lambda$-return:**

Define the $n$-step return:

$$G_t^{(n)} = \sum_{k=0}^{n-1}\gamma^k r_{t+k} + \gamma^n V(s_{t+n})$$

The **$\lambda$-return** is an exponentially weighted average of all $n$-step returns:

$$G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)}$$

**Proof of valid weighting:** The weights $(1-\lambda)\lambda^{n-1}$ sum to 1:

$$\sum_{n=1}^{\infty}(1-\lambda)\lambda^{n-1} = (1-\lambda)\frac{1}{1-\lambda} = 1 \quad \checkmark$$

**Backward view — Eligibility traces:**

Define the eligibility trace for state $s$:

$$e_0(s) = 0$$

$$e_t(s) = \gamma\lambda \, e_{t-1}(s) + \mathbb{1}[s_t = s]$$

The TD($\lambda$) update:

$$V(s) \leftarrow V(s) + \alpha \delta_t e_t(s) \quad \forall s$$

### 1.4 Proof of Forward-Backward Equivalence

**Theorem:** Over a complete episode, the total update to $V(s)$ using the backward view (eligibility traces) equals the total update using the forward view ($\lambda$-returns).

**Proof:**

**Step 1:** The total backward-view update for state $s$ over an episode is:

$$\Delta V_{\text{back}}(s) = \alpha \sum_{t=0}^{T-1} \delta_t e_t(s)$$

**Step 2:** Expand the eligibility trace. $e_t(s) = \sum_{k=0}^{t}(\gamma\lambda)^{t-k}\mathbb{1}[s_k = s]$.

**Step 3:** Substitute and interchange sums. Let $\mathcal{T}_s = \{k : s_k = s\}$:

$$\sum_{t=0}^{T-1}\delta_t e_t(s) = \sum_{t=0}^{T-1}\delta_t\sum_{k=0}^{t}(\gamma\lambda)^{t-k}\mathbb{1}[s_k = s] = \sum_{k \in \mathcal{T}_s}\sum_{t=k}^{T-1}(\gamma\lambda)^{t-k}\delta_t$$

**Step 4:** For the inner sum, define $m = t - k$:

$$\sum_{t=k}^{T-1}(\gamma\lambda)^{t-k}\delta_t = \sum_{m=0}^{T-1-k}(\gamma\lambda)^m \delta_{k+m}$$

**Step 5:** Expand $\delta_{k+m} = r_{k+m} + \gamma V(s_{k+m+1}) - V(s_{k+m})$. After telescoping:

$$\sum_{m=0}^{T-1-k}(\gamma\lambda)^m\delta_{k+m} = -V(s_k) + (1-\lambda)\sum_{n=1}^{T-k-1}\lambda^{n-1}G_k^{(n)} + \lambda^{T-k-1}G_k^{(T-k)}$$

**Step 6:** This equals $G_k^\lambda - V(s_k)$, the forward-view TD error.

**Step 7:** Therefore:

$$\Delta V_{\text{back}}(s) = \alpha\sum_{k \in \mathcal{T}_s}(G_k^\lambda - V(s_k)) = \Delta V_{\text{forward}}(s)$$

**Result:** The forward and backward views produce identical updates over complete episodes.
$\blacksquare$

**Intuition:** Eligibility traces provide a computationally efficient mechanism ($O(|\mathcal{S}|)$ per step) to implement what would otherwise require storing and replaying entire episodes (the forward view). The trace $e_t(s)$ records "how responsible" state $s$ is for the current TD error — recently visited states get more credit.

### 1.5 SARSA — On-Policy TD Control

SARSA updates the action-value function using the quintuple $(s_t, a_t, r_t, s_{t+1}, a_{t+1})$:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\left[r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)\right]$$

**Convergence:** Under the Robbins–Monro conditions on $\alpha_t$ and GLIE (Greedy in the Limit with Infinite Exploration) conditions on the policy:

$$Q_t(s, a) \xrightarrow{\text{a.s.}} Q^\pi(s, a) \text{ (policy evaluation)} \quad \text{or} \quad Q_t \to Q^* \text{ (with GLIE)}$$

---

## 2. Algorithm / Method

### 2.1 Pseudocode: SARSA with Eligibility Traces

```
Algorithm: SARSA(λ)
Input: step size α, discount γ, trace decay λ, episodes N
Output: Q(s,a)

1. Q(s,a) = 0 ∀s,a
2. For episode = 1 to N:
     e(s,a) = 0 ∀s,a
     s = initial state, a = ε-greedy(Q, s)
     REPEAT:
       Take action a, observe r, s'
       a' = ε-greedy(Q, s')
       δ = r + γQ(s',a') - Q(s,a)
       e(s,a) = e(s,a) + 1
       For all (s,a):
         Q(s,a) = Q(s,a) + α·δ·e(s,a)
         e(s,a) = γ·λ·e(s,a)
       s = s', a = a'
     UNTIL s is terminal
3. Return Q
```

### 2.2 Complexity Analysis

- **TD(0):** $O(1)$ per step (update single state)
- **TD($\lambda$) with traces:** $O(|\mathcal{S}|)$ per step (update all traces)
- **SARSA:** $O(|\mathcal{S}||\mathcal{A}|)$ per step with traces

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

- **State:** $s_t$ — environment observation
- **Action:** $a_t$ — selected by $\varepsilon$-greedy policy
- **Reward:** $r_t$ — immediate feedback
- **Transition:** Unknown (model-free)

### 3.2 Why RL?

TD learning is the most practical RL method: it updates online (per step, not per episode), works with incomplete episodes, and has lower variance than MC. Eligibility traces provide a smooth spectrum between MC ($\lambda=1$) and TD(0) ($\lambda=0$), allowing practitioners to tune the bias-variance tradeoff.

---

## 4. Dataset

- **Name:** GridWorld / CliffWalking environments
- **Size:** Configurable
- **Auto-download:**

```python
import gymnasium as gym
env = gym.make('CliffWalking-v0')
obs, info = env.reset()
```

---

## 5. Key Equations Summary

| Equation | Meaning |
|----------|---------|
| $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ | TD error |
| $G_t^\lambda = (1-\lambda)\sum_{n=1}^\infty \lambda^{n-1}G_t^{(n)}$ | $\lambda$-return |
| $e_t(s) = \gamma\lambda e_{t-1}(s) + \mathbb{1}[s_t = s]$ | Eligibility trace |
| $\sum_t \alpha_t = \infty, \; \sum_t \alpha_t^2 < \infty$ | Robbins–Monro conditions |
| $Q(s,a) \leftarrow Q + \alpha[r + \gamma Q(s',a') - Q]$ | SARSA update |

---

## 6. References

- Sutton, R. S. "Learning to Predict by the Methods of Temporal Differences," *Machine Learning*, 3(1):9–44, 1988.
- Tsitsiklis, J. N. "Asynchronous Stochastic Approximation and Q-Learning," *Machine Learning*, 16(3):185–202, 1994.
- Singh, S. et al. "Convergence Results for Single-Step On-Policy RL Algorithms," *Machine Learning*, 2000.
- Sutton, R. S. & Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018, Ch. 6–7.
