![Module Logo](../logo.png)

# Multi-Armed Bandits

## Overview

The multi-armed bandit (MAB) problem is the simplest form of reinforcement learning — a single-state MDP where the agent must balance exploration and exploitation. This document derives the UCB1 algorithm rigorously from Hoeffding's inequality, proves the $O(\sqrt{KT\log T})$ regret bound, develops the Bayesian Thompson Sampling framework, and establishes information-theoretic lower bounds on achievable regret.

## Prerequisites

- Probability theory (concentration inequalities, Module 03.1)
- Bayesian inference basics
- Exploration vs. exploitation concepts (Module 03.5)

---

## 1. Mathematical Foundations

### 1.1 Core Definition

A stochastic $K$-armed bandit consists of:
- $K$ arms with unknown reward distributions $\nu_k$, each with mean $\mu_k$
- The optimal arm: $k^* = \arg\max_k \mu_k$, with $\mu^* = \mu_{k^*}$
- Suboptimality gaps: $\Delta_k = \mu^* - \mu_k \geq 0$

**Pseudo-regret:**

$$\bar{R}_T = \mathbb{E}\left[\sum_{t=1}^{T} (\mu^* - \mu_{A_t})\right] = \sum_{k=1}^{K} \Delta_k \, \mathbb{E}[N_k(T)]$$

### 1.2 UCB1 Derivation from Hoeffding's Inequality

**Step 1:** For arm $k$ pulled $N_k$ times with i.i.d. rewards $X_1, \ldots, X_{N_k} \in [0,1]$, Hoeffding's inequality gives:

$$P\!\left(\hat{\mu}_k - \mu_k \geq \epsilon\right) \leq e^{-2N_k \epsilon^2}$$

$$P\!\left(\mu_k - \hat{\mu}_k \geq \epsilon\right) \leq e^{-2N_k \epsilon^2}$$

**Step 2:** Set the confidence radius $c_{k,t} = \sqrt{\frac{2\log t}{N_k(t)}}$. By union bound over time:

$$P\!\left(\mu_k \notin \left[\hat{\mu}_k - c_{k,t}, \; \hat{\mu}_k + c_{k,t}\right]\right) \leq 2t^{-4}$$

**Step 3:** The UCB index is the upper end of the confidence interval:

$$\text{UCB}_k(t) = \hat{\mu}_k(t) + \sqrt{\frac{2\log t}{N_k(t)}}$$

**Step 4:** UCB1 selects $A_t = \arg\max_k \text{UCB}_k(t)$.

**Optimism principle:** With high probability, $\mu_k \leq \text{UCB}_k(t)$. If a suboptimal arm $k$ is selected, either: (a) $\text{UCB}_k(t) \geq \text{UCB}_{k^*}(t) \geq \mu^*$, which requires the confidence interval to be wide (small $N_k$), meaning $k$ hasn't been explored enough. As $N_k$ grows, the interval shrinks and $\text{UCB}_k$ drops below $\mu^*$, so arm $k$ is abandoned.

### 1.3 Proof of UCB1 Regret Bound

**Theorem:** The pseudo-regret of UCB1 satisfies:

$$\bar{R}_T \leq \sum_{k:\Delta_k > 0} \left(\frac{8\log T}{\Delta_k} + (1 + \frac{\pi^2}{3})\Delta_k\right)$$

**Proof:**

**Step 1:** Arm $k$ is selected at time $t$ only if $\text{UCB}_k(t) \geq \text{UCB}_{k^*}(t)$. This requires at least one of:

- (i) $\hat{\mu}_{k^*}(t) + c_{k^*,t} \leq \mu^*$ — the optimal arm is underestimated
- (ii) $\hat{\mu}_k(t) + c_{k,t} \geq \mu^*$ — the suboptimal arm is overestimated
- (iii) $N_k(t) \leq \frac{8\log T}{\Delta_k^2}$ — arm $k$ hasn't been pulled enough

**Step 2:** Events (i) and (ii) each occur with probability at most $t^{-4}$ by Hoeffding.

**Step 3:** Therefore:

$$\mathbb{E}[N_k(T)] \leq \frac{8\log T}{\Delta_k^2} + \sum_{t=1}^{T} 2t^{-4} \leq \frac{8\log T}{\Delta_k^2} + 1 + \frac{\pi^2}{3}$$

**Step 4:** Multiply by $\Delta_k$:

$$\bar{R}_T = \sum_k \Delta_k \mathbb{E}[N_k] \leq \sum_{k:\Delta_k>0}\left(\frac{8\log T}{\Delta_k} + (1+\frac{\pi^2}{3})\Delta_k\right)$$

**Step 5:** Gap-independent bound. For the worst case, set $\Delta_k = \Delta$ for all suboptimal arms:

$$\bar{R}_T \leq \frac{8(K-1)\log T}{\Delta} + \text{lower order}$$

Optimizing over $\Delta$: $\bar{R}_T = O(\sqrt{KT\log T})$.

**Result:**

$$\boxed{\bar{R}_T = O\!\left(\sqrt{KT\log T}\right)}$$
$\blacksquare$

### 1.4 Bayesian Thompson Sampling

**Setup:** Place a prior on each arm's mean: $\mu_k \sim P_0$ (e.g., $\text{Beta}(1,1)$ for Bernoulli rewards).

**Algorithm:**
1. For each arm $k$, sample $\tilde{\mu}_k \sim P(\mu_k \mid \text{data}_k)$.
2. Play $A_t = \arg\max_k \tilde{\mu}_k$.
3. Update posterior with observed reward.

**Conjugate posterior for Bernoulli rewards:**

After observing $s_k$ successes and $f_k$ failures:

$$\mu_k \mid \text{data} \sim \text{Beta}(1 + s_k, 1 + f_k)$$

**Posterior mean:** $\frac{1 + s_k}{2 + s_k + f_k}$ — a smoothed version of the sample mean $\frac{s_k}{s_k + f_k}$.

**Regret bound (Agrawal & Goyal):** Thompson Sampling achieves:

$$\bar{R}_T \leq O\!\left(\sum_{k:\Delta_k>0}\frac{\log T}{\Delta_k}\right)$$

This matches the Lai–Robbins information-theoretic lower bound asymptotically, proving Thompson Sampling is asymptotically optimal.

### 1.5 Information-Theoretic Lower Bound

**Theorem (Lai–Robbins, 1985):** For any consistent policy (one where $\mathbb{E}[N_k(T)] = o(T^a)$ for all $a > 0$ and all suboptimal $k$):

$$\liminf_{T \to \infty} \frac{\bar{R}_T}{\log T} \geq \sum_{k:\Delta_k > 0} \frac{\Delta_k}{D_{\text{KL}}(\nu_k \| \nu^*)}$$

where $D_{\text{KL}}(\nu_k \| \nu^*)$ is the KL divergence between the distributions of arm $k$ and the optimal arm.

**Intuition:** To distinguish arm $k$ from the optimal arm, the agent needs at least $\Omega(1/D_{\text{KL}})$ samples. Since each sample contributes $\Delta_k$ regret, the total regret from arm $k$ is at least $\Omega(\Delta_k / D_{\text{KL}})$.

---

## 2. Algorithm / Method

### 2.1 Pseudocode: UCB1

```
Algorithm: UCB1
Input: K arms, T rounds
Output: Total reward

1. For k = 1 to K: play arm k, observe reward, set N_k = 1
2. For t = K+1 to T:
     For each k: UCB_k = μ̂_k + sqrt(2·log(t)/N_k)
     a_t = argmax_k UCB_k
     Observe r_t, update μ̂_{a_t} and N_{a_t}
3. Return Σ r_t
```

### 2.2 Complexity Analysis

- **Time per step:** $O(K)$
- **Space:** $O(K)$
- **Regret:** $O(\sqrt{KT\log T})$ (UCB1), $O(\sqrt{KT})$ (Thompson Sampling, minimax)

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

- **State:** Single state (stateless) or context $s_t$ (contextual bandits)
- **Action:** $a_t \in \{1, \ldots, K\}$ — arm/filter selection
- **Reward:** $r_t \sim \nu_{a_t}$ — quality metric improvement
- **Transition:** None (single-step)

### 3.2 Why RL?

In image processing, bandits model the problem of selecting the best filter or enhancement from $K$ options without knowing which is optimal for a given image. Contextual bandits extend this: the "context" is the image, and the agent learns which filter works best for each type of image.

---

## 4. Dataset

- **Name:** Synthetic bandit instances
- **Size:** Configurable
- **Auto-download:**

```python
import numpy as np
K, T = 10, 10000
true_means = np.random.beta(2, 5, size=K)
rewards = np.array([np.random.binomial(1, p, T) for p in true_means])
```

---

## 5. Key Equations Summary

| Equation | Meaning |
|----------|---------|
| $\bar{R}_T = \sum_k \Delta_k \mathbb{E}[N_k(T)]$ | Regret decomposition |
| $\text{UCB}_k = \hat{\mu}_k + \sqrt{2\log t / N_k}$ | UCB1 index |
| $\bar{R}_T = O(\sqrt{KT\log T})$ | UCB1 regret bound |
| $\mu_k \mid \text{data} \sim \text{Beta}(1+s, 1+f)$ | Thompson sampling posterior |
| $\bar{R}_T / \log T \geq \sum_k \Delta_k/D_{\text{KL}}$ | Lai–Robbins lower bound |

---

## 6. References

- Auer, P., Cesa-Bianchi, N., & Fischer, P. "Finite-time Analysis of the Multiarmed Bandit Problem," *Machine Learning*, 2002.
- Agrawal, S. & Goyal, N. "Analysis of Thompson Sampling for the Multi-armed Bandit Problem," *COLT*, 2012.
- Lai, T. L. & Robbins, H. "Asymptotically Efficient Adaptive Allocation Rules," 1985.
- Lattimore, T. & Szepesvári, C. *Bandit Algorithms*, Cambridge University Press, 2020.
