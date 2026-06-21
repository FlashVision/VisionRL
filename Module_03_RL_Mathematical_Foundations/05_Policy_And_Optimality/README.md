![Module Logo](../logo.png)

# Exploration vs. Exploitation and Optimal Policies

## Overview

The exploration–exploitation tradeoff is the fundamental dilemma in reinforcement learning: should the agent exploit the best-known action or explore to discover potentially better ones? This document derives the Upper Confidence Bound (UCB) algorithm from Hoeffding's inequality (with full proof), develops the Bayesian Thompson Sampling analysis, provides epsilon-greedy regret analysis, and connects these ideas to optimal policy theory.

## Prerequisites

- Probability theory (Module 03.1)
- Value functions (Module 03.4)
- Basic statistics (confidence intervals, Bayesian inference)

---

## 1. Mathematical Foundations

### 1.1 Core Definition — Multi-Armed Bandit Problem

A $K$-armed bandit is defined by $K$ reward distributions $\nu_1, \ldots, \nu_K$ with means $\mu_1 \geq \mu_2 \geq \cdots \geq \mu_K$. At each step $t$, the agent selects an arm $a_t \in \{1, \ldots, K\}$ and receives reward $r_t \sim \nu_{a_t}$.

**Regret** measures the cost of not always playing the optimal arm:

$$\text{Regret}(T) = T\mu_1 - \mathbb{E}\left[\sum_{t=1}^{T} r_t\right] = \sum_{k=1}^{K} \Delta_k \, \mathbb{E}[N_k(T)]$$

where $\Delta_k = \mu_1 - \mu_k$ is the suboptimality gap and $N_k(T)$ is the number of times arm $k$ is pulled in $T$ rounds.

### 1.2 Proof of Hoeffding's Inequality

**Theorem (Hoeffding):** Let $X_1, \ldots, X_n$ be independent random variables with $X_i \in [a_i, b_i]$. Then for $\bar{X} = \frac{1}{n}\sum_i X_i$:

$$P(\bar{X} - \mathbb{E}[\bar{X}] \geq t) \leq \exp\!\left(-\frac{2n^2t^2}{\sum_{i=1}^{n}(b_i - a_i)^2}\right)$$

**Proof:**

**Step 1 — Chernoff bound technique.** For any $s > 0$:

$$P(\bar{X} - \mathbb{E}[\bar{X}] \geq t) = P\!\left(e^{s(\bar{X} - \mathbb{E}[\bar{X}])} \geq e^{st}\right) \leq e^{-st} \mathbb{E}\!\left[e^{s(\bar{X} - \mathbb{E}[\bar{X}])}\right]$$

by Markov's inequality.

**Step 2 — Factor by independence.** Let $Y_i = X_i - \mathbb{E}[X_i]$. Then $\sum Y_i = n(\bar{X} - \mathbb{E}[\bar{X}])$:

$$\mathbb{E}\!\left[e^{s\sum Y_i / n}\right] = \prod_{i=1}^{n} \mathbb{E}\!\left[e^{sY_i/n}\right]$$

**Step 3 — Hoeffding's lemma.** For a bounded random variable $Y \in [a, b]$ with $\mathbb{E}[Y] = 0$:

$$\mathbb{E}[e^{sY}] \leq \exp\!\left(\frac{s^2(b-a)^2}{8}\right)$$

*Proof of lemma:* By convexity of $e^{sy}$, for $y \in [a, b]$:

$$e^{sy} \leq \frac{b-y}{b-a}e^{sa} + \frac{y-a}{b-a}e^{sb}$$

Taking expectations ($\mathbb{E}[Y] = 0$):

$$\mathbb{E}[e^{sY}] \leq \frac{b}{b-a}e^{sa} + \frac{-a}{b-a}e^{sb}$$

Let $p = -a/(b-a)$, $u = s(b-a)$. Then $\mathbb{E}[e^{sY}] \leq e^{-pu}(1 - p + pe^u) \leq e^{u^2/8}$ (by the inequality $\log(1-p+pe^u) \leq pu + u^2/8$).

**Step 4 — Combine.** Each term satisfies $\mathbb{E}[e^{sY_i/n}] \leq \exp(s^2(b_i - a_i)^2/(8n^2))$:

$$P(\bar{X} - \mathbb{E}[\bar{X}] \geq t) \leq \exp\!\left(-st + \frac{s^2}{8n^2}\sum(b_i - a_i)^2\right)$$

**Step 5 — Optimize over $s$.** Let $c = \sum(b_i - a_i)^2 / (8n^2)$. Minimize $-st + s^2 c$ by setting $s = t/(2c)$:

$$P(\bar{X} - \mathbb{E}[\bar{X}] \geq t) \leq \exp\!\left(-\frac{t^2}{4c}\right) = \exp\!\left(-\frac{2n^2t^2}{\sum(b_i - a_i)^2}\right)$$

**Result:**

$$\boxed{P(\bar{X} - \mathbb{E}[\bar{X}] \geq t) \leq \exp\!\left(-\frac{2n^2t^2}{\sum_{i=1}^n(b_i-a_i)^2}\right)}$$
$\blacksquare$

### 1.3 UCB Algorithm Derivation

**Step 1:** For rewards in $[0, 1]$, Hoeffding gives:

$$P\!\left(\hat{\mu}_k - \mu_k \geq \sqrt{\frac{\log(1/\delta)}{2N_k}}\right) \leq \delta$$

where $\hat{\mu}_k$ is the sample mean of arm $k$ after $N_k$ pulls.

**Step 2:** Similarly, $P\!\left(\mu_k - \hat{\mu}_k \geq \sqrt{\frac{\log(1/\delta)}{2N_k}}\right) \leq \delta$.

**Step 3:** Set $\delta = 1/t^2$ (decreasing with time) to get a confidence bound that holds uniformly:

$$\mu_k \leq \hat{\mu}_k + \sqrt{\frac{\log t}{N_k(t)}} \quad \text{with high probability}$$

**Step 4:** The UCB1 policy selects:

$$a_t = \arg\max_k \left[\hat{\mu}_k(t) + \sqrt{\frac{2\log t}{N_k(t)}}\right]$$

The first term exploits (prefers high estimated reward); the second term explores (prefers under-sampled arms).

**Result (UCB Regret Bound):**

**Theorem:** The regret of UCB1 satisfies:

$$\text{Regret}(T) \leq \sum_{k:\Delta_k>0} \frac{8\log T}{\Delta_k} + (1 + \frac{\pi^2}{3})\sum_{k=1}^K \Delta_k$$

$$= O\!\left(\sqrt{KT\log T}\right)$$

(using $\sum_k 1/\Delta_k \leq K/\Delta_{\min}$ and optimizing).
$\blacksquare$

### 1.4 Thompson Sampling — Bayesian Analysis

**Algorithm:** Maintain a posterior distribution $P(\mu_k \mid \text{data})$ for each arm. At each step:

1. Sample $\tilde{\mu}_k \sim P(\mu_k \mid \text{data})$ for each arm.
2. Play $a_t = \arg\max_k \tilde{\mu}_k$.

**For Bernoulli bandits:** Use the Beta-Bernoulli conjugate model:

**Step 1:** Prior: $\mu_k \sim \text{Beta}(\alpha_0, \beta_0)$ (typically $\alpha_0 = \beta_0 = 1$, uniform).

**Step 2:** After $s$ successes and $f$ failures on arm $k$: $\mu_k \mid \text{data} \sim \text{Beta}(\alpha_0 + s, \beta_0 + f)$.

**Step 3:** The posterior mean is $\frac{\alpha_0 + s}{\alpha_0 + \beta_0 + s + f}$, which approaches the sample mean as $s + f \to \infty$.

**Regret bound (Agrawal & Goyal, 2012):**

$$\mathbb{E}[\text{Regret}(T)] \leq O\!\left(\sum_{k:\Delta_k > 0}\frac{\log T}{\Delta_k}\right)$$

matching the Lai–Robbins lower bound asymptotically.

### 1.5 Epsilon-Greedy Analysis

**Policy:**

$$a_t = \begin{cases} \arg\max_k \hat{\mu}_k & \text{w.p. } 1 - \varepsilon \\ \text{uniform random arm} & \text{w.p. } \varepsilon \end{cases}$$

**Regret analysis for constant $\varepsilon$:**

**Step 1:** In each step, the expected regret contribution is at most $\varepsilon \cdot \Delta_{\max} \cdot \frac{K-1}{K}$ from random exploration, plus $(1-\varepsilon) \cdot P(\text{wrong greedy})$ from exploitation errors.

**Step 2:** For constant $\varepsilon$: $\text{Regret}(T) = O(\varepsilon T + (1-\varepsilon) \cdot \text{estimation error})$. This is linear in $T$ — suboptimal.

**Step 3:** With decaying $\varepsilon_t = \min(1, cK/(d^2 t))$ where $d = \min_{k:\Delta_k>0}\Delta_k$:

$$\text{Regret}(T) = O(K\log T / d)$$

matching UCB up to constants, but requiring knowledge of $d$.

---

## 2. Algorithm / Method

### 2.1 Pseudocode: UCB1

```
Algorithm: UCB1
Input: Number of arms K, time horizon T
Output: Sequence of arms played

1. INITIALIZE: Play each arm once (t = 1 to K)
2. For t = K+1 to T:
     For each arm k:
       UCB_k = μ̂_k + sqrt(2·log(t) / N_k)
     Play a_t = argmax_k UCB_k
     Update μ̂_{a_t} and N_{a_t}
3. Return total reward
```

### 2.2 Complexity Analysis

- **Time per step:** $O(K)$ — compute UCB for each arm
- **Space:** $O(K)$ — store counts and means
- **Regret:** $O(\sqrt{KT\log T})$

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

- **State:** In bandits, there is only one state (stateless). In contextual bandits, $s_t$ is the context (e.g., image features).
- **Action:** $a_t \in \{1, \ldots, K\}$ — arm selection
- **Reward:** $r_t \sim \nu_{a_t}$ — stochastic reward
- **Transition:** $s_{t+1} = s_t$ (single state) or $s_{t+1}$ from environment

### 3.2 Why RL?

Bandit algorithms are the simplest form of RL (one-step MDPs). The exploration–exploitation tradeoff in bandits directly generalizes to full RL: UCB-style bonuses become exploration bonuses in state spaces, Thompson sampling becomes posterior sampling RL (PSRL), and $\varepsilon$-greedy is the most common RL exploration strategy.

---

## 4. Dataset

- **Name:** Synthetic bandit instances
- **Size:** Configurable ($K$ arms, $T$ rounds)
- **Auto-download:**

```python
import numpy as np
K = 10
true_means = np.random.uniform(0, 1, size=K)
def pull_arm(k):
    return np.random.binomial(1, true_means[k])
```

---

## 5. Key Equations Summary

| Equation | Meaning |
|----------|---------|
| $\text{Regret}(T) = T\mu^* - \sum_t \mathbb{E}[r_t]$ | Cumulative regret |
| $P(\bar{X} - \mu \geq t) \leq e^{-2nt^2}$ | Hoeffding's inequality |
| $a_t = \arg\max_k[\hat{\mu}_k + \sqrt{2\log t/N_k}]$ | UCB1 selection rule |
| $\mu_k \mid \text{data} \sim \text{Beta}(\alpha+s, \beta+f)$ | Thompson sampling posterior |
| $\text{Regret}_{\text{UCB}} = O(\sqrt{KT\log T})$ | UCB regret bound |

---

## 6. References

- Auer, P., Cesa-Bianchi, N., & Fischer, P. "Finite-time Analysis of the Multiarmed Bandit Problem," *Machine Learning*, 47(2):235–256, 2002.
- Agrawal, S. & Goyal, N. "Analysis of Thompson Sampling for the Multi-armed Bandit Problem," *COLT*, 2012.
- Lai, T. L. & Robbins, H. "Asymptotically Efficient Adaptive Allocation Rules," *Advances in Applied Mathematics*, 6(1):4–22, 1985.
- Lattimore, T. & Szepesvári, C. *Bandit Algorithms*, Cambridge University Press, 2020.
