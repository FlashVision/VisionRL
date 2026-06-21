![Module Logo](../logo.png)

# 6.5 Experience Replay & Advanced DQN Tricks

## Overview

This notebook covers the most important improvements to the base DQN algorithm: Prioritized Experience Replay (PER) with importance sampling corrections, Double DQN to fix Q-value overestimation, Dueling DQN that separates value and advantage streams, and N-step returns for faster credit assignment. We derive each technique mathematically and run ablation studies on the ImageEnhancementEnv.

---

## Mathematical Foundation

### 1. Prioritized Experience Replay (PER)

**Problem with uniform replay:** In a buffer of 100K transitions, most are "boring" (small TD error — already well-predicted). The rare, surprising transitions (large TD error) are the most informative but sampled as rarely as boring ones.

**PER idea:** Sample transitions with probability proportional to their TD error.

**Priority of transition $i$:**

$$p_i = |\delta_i| + \varepsilon$$

where $\delta_i = r_i + \gamma\max_{a'}Q(s'_i,a';\theta^-) - Q(s_i,a_i;\theta)$ is the TD error and $\varepsilon > 0$ is a small constant preventing zero-priority transitions from never being sampled.

**Sampling probability (with temperature $\alpha$):**

$$P(i) = \frac{p_i^\alpha}{\sum_{j=1}^{N}p_j^\alpha}$$

where $\alpha$ controls how much prioritization is used:
- $\alpha = 0$: Uniform sampling (standard replay)
- $\alpha = 1$: Full prioritization (greedy priority)
- Typical: $\alpha = 0.6$

**Problem — Biased gradient estimates:**

PER changes the sampling distribution from uniform to priority-based. The DQN loss under uniform sampling is:

$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}\delta_i^2 = \mathbb{E}_{i \sim \text{Uniform}}[\delta_i^2]$$

Under PER, we're computing $\mathbb{E}_{i \sim P}[\delta_i^2] \neq \mathbb{E}_{i \sim \text{Uniform}}[\delta_i^2]$.

**Solution — Importance Sampling (IS) correction:**

$$w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta$$

The corrected loss:

$$\mathcal{L}_{\text{PER}} = \mathbb{E}_{i \sim P}\left[w_i \cdot \delta_i^2\right] = \sum_{i} P(i) \cdot \left(\frac{1}{N \cdot P(i)}\right)^\beta \cdot \delta_i^2$$

When $\beta = 1$: $w_i = \frac{1}{N \cdot P(i)}$, which exactly corrects the bias:

$$\sum_i P(i) \cdot \frac{1}{N \cdot P(i)} \cdot \delta_i^2 = \frac{1}{N}\sum_i \delta_i^2$$

**Annealing schedule:** Start with $\beta_0 < 1$ (allowing bias early when exploration is more important) and linearly anneal to $\beta = 1$:

$$\beta_t = \beta_0 + (1 - \beta_0)\frac{t}{T}$$

**Normalization:** For stability, IS weights are normalized by the maximum weight:

$$\hat{w}_i = \frac{w_i}{\max_j w_j} = \left(\frac{\min_j P(j)}{P(i)}\right)^\beta$$

This ensures $\hat{w}_i \leq 1$, preventing large gradient magnitudes.

**Data structure:** PER uses a **sum tree** (segment tree) for $O(\log N)$ sampling and priority updates, rather than $O(N)$ for naive implementation.

### 2. Double DQN

**Problem — Overestimation bias in DQN:**

The standard DQN target uses:

$$y = r + \gamma\max_{a'}Q(s', a'; \theta^-)$$

The $\max$ operator introduces a positive bias:

$$\mathbb{E}[\max_{a'}Q(s',a')] \geq \max_{a'}\mathbb{E}[Q(s',a')]$$

*This is Jensen's inequality applied to the convex $\max$ function.* Even with unbiased Q-value estimates, taking the max over noisy estimates overestimates the true maximum.

**Quantifying the bias:** For $n$ actions with Q-values $Q_i \sim \mathcal{N}(Q^*, \sigma^2)$:

$$\mathbb{E}\left[\max_i Q_i\right] \approx Q^* + \sigma\sqrt{2\log n}$$

The bias grows with the number of actions and the noise level.

**Double DQN solution (Van Hasselt et al., 2016):**

Decouple action **selection** from action **evaluation** using two networks:

$$y^{\text{DDQN}} = r + \gamma Q\left(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-\right)$$

- **Online network** $\theta$ selects the best action: $a^* = \arg\max_{a'} Q(s', a'; \theta)$
- **Target network** $\theta^-$ evaluates that action: $Q(s', a^*; \theta^-)$

**Why this reduces overestimation:** If the online network overestimates Q for action $a'$, it might select $a'$. But the target network (with different noise) is unlikely to *also* overestimate the same action. The noise in the two networks is decorrelated, so overestimation in one is not reinforced by the other.

**Formal analysis:** Let $Q^\theta = Q^* + \varepsilon^\theta$ and $Q^{\theta^-} = Q^* + \varepsilon^{\theta^-}$ where $\varepsilon$ terms are zero-mean noise. Then:

$$\mathbb{E}[Q^{\theta^-}(s', \arg\max_{a'}Q^\theta(s',a'))] \leq \mathbb{E}[\max_{a'}Q^\theta(s',a')]$$

The inequality holds because the evaluation noise $\varepsilon^{\theta^-}$ is independent of the selection noise $\varepsilon^\theta$.

### 3. Dueling DQN

**Intuition:** For many states, the value of being in state $s$ is more important than which specific action to take. A high-value state (e.g., a nearly-clean image) is good regardless of action.

**Architecture:** Split the Q-network into two streams:

$$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a'; \theta, \alpha)$$

where:
- $V(s; \theta, \beta)$: **Value stream** — how good is state $s$?
- $A(s, a; \theta, \alpha)$: **Advantage stream** — how much better is action $a$ than average?
- The mean subtraction ensures identifiability: $\sum_a A(s,a) = 0$

**Why subtract the mean?** Without it, $V$ and $A$ are not uniquely determined — we could add a constant to $V$ and subtract it from all $A$ values. The constraint $\sum_a A = 0$ makes the decomposition unique:

$$V(s) = Q(s, a^*) - A(s, a^*) = \max_a Q(s,a) \quad \text{(approximately)}$$

**Network architecture:**

```
Input s → CNN encoder → shared features
                            ├─→ FC → FC → V(s)     [scalar]
                            └─→ FC → FC → A(s,a)   [|A| values]
                                          ↓
                            Q(s,a) = V(s) + A(s,a) - mean(A)
```

**Gradient flow benefit:** The value stream gets gradient from *every* action's loss, not just the selected action. This leads to faster and more stable learning of state values.

### 4. N-Step Returns

**Standard 1-step target:**

$$G_t^{(1)} = r_t + \gamma\max_{a'}Q(s_{t+1}, a'; \theta^-)$$

**N-step target:**

$$G_t^{(n)} = \sum_{k=0}^{n-1}\gamma^k r_{t+k} + \gamma^n\max_{a'}Q(s_{t+n}, a'; \theta^-)$$

**Bias-variance trade-off:**
- $n = 1$: More bias (heavy bootstrapping), lower variance
- $n = \infty$: Zero bias (pure Monte Carlo), highest variance
- Typical: $n = 3$ or $n = 5$

**Why N-step helps:** Consider a reward that occurs $k$ steps after an action. With 1-step returns, this reward must propagate through $k$ Bellman backups (one per training iteration) to reach the action that caused it. With $n$-step returns ($n \geq k$), the reward directly affects the loss for that action — **accelerating credit assignment**.

**N-step with experience replay (complication):** N-step returns are computed under the behavior policy $\pi_b$ (the policy during data collection). If we store $(s_t, a_t, G_t^{(n)}, s_{t+n})$ in the replay buffer and train off-policy, the intermediate actions $a_{t+1}, \ldots, a_{t+n-1}$ were chosen by $\pi_b$, not the current policy $\pi$. This introduces off-policy bias. Solutions:

1. **Use small $n$** (reduces bias)
2. **Importance sampling correction** (complex, high variance)
3. **Retrace($\lambda$)** (truncated IS, theoretical guarantees)
4. **Accept the bias** (works well in practice for small $n$)

### 5. Combining All Tricks — Rainbow DQN

Rainbow (Hessel et al., 2018) combines six improvements:

| Component | Contribution |
|-----------|-------------|
| Double DQN | Reduces overestimation |
| PER | Focuses on informative transitions |
| Dueling | Separates V and A |
| N-step returns | Faster credit assignment |
| Distributional DQN | Models full return distribution |
| Noisy Nets | Parameter-space exploration |

**Ablation finding:** Each component contributes, but PER and N-step returns provide the largest individual improvements.

### 6. Noisy Networks (Brief)

Replace linear layers with noisy variants:

$$y = (\mu^w + \sigma^w \odot \varepsilon^w)x + (\mu^b + \sigma^b \odot \varepsilon^b)$$

where $\varepsilon^w, \varepsilon^b$ are noise vectors sampled each forward pass. This replaces ε-greedy with **learned, state-dependent exploration**.

---

## Step-by-Step Breakdown

1. **Implement Prioritized Replay Buffer:** Build a sum-tree data structure. Implement `add(transition, priority)`, `sample(batch_size)` returning transitions + IS weights, and `update_priorities(indices, new_priorities)`.

2. **Implement Double DQN:** Modify the DQN target computation: select action with online network, evaluate with target network. Compare Q-value estimates with standard DQN — plot the overestimation.

3. **Implement Dueling Architecture:** Split the network after the conv layers into value and advantage streams. Implement mean subtraction for identifiability. Verify that $V(s) + A(s,a) - \text{mean}(A)$ outputs valid Q-values.

4. **Implement N-step Returns:** Modify the replay buffer to store N-step transitions. Implement the N-step return computation. Compare $n \in \{1, 3, 5, 10\}$.

5. **Ablation study setup:** Train five agents on the same ImageEnhancementEnv: (a) base DQN, (b) + Double, (c) + Dueling, (d) + PER, (e) + N-step. Use identical hyperparameters except the ablated component.

6. **Combined agent:** Implement all four tricks together. Compare with base DQN — show improvement in final PSNR, convergence speed, and training stability.

7. **Visualization:** Plot Q-value distributions (overestimation analysis), PER sampling frequencies (which transitions are replayed most), and per-technique learning curves.

---

## Dataset Used

**ImageEnhancementEnv — Technique Comparison (CIFAR-10)**
- **Source:** `torchvision.datasets.CIFAR10` (auto-downloads)
- **Environment:** Same ImageEnhancementEnv as DQN (Module 6.1)
- **Purpose:** Controlled comparison of DQN improvements on identical task
- **Evaluation:** 100 held-out degraded images, measuring PSNR improvement

```python
from torchvision import datasets

cifar = datasets.CIFAR10(root='./data', train=True, download=True)
cifar_test = datasets.CIFAR10(root='./data', train=False, download=True)

# Use test set images for evaluation
eval_images = [cifar_test[i][0] for i in range(100)]
```

---

## Key Equations Summary

| Concept | Equation |
|---------|----------|
| PER sampling probability | $P(i) = p_i^\alpha / \sum_j p_j^\alpha$ |
| IS weight | $w_i = (N \cdot P(i))^{-\beta}$ |
| Double DQN target | $y = r + \gamma Q(s', \arg\max_{a'}Q(s',a';\theta); \theta^-)$ |
| Dueling Q-value | $Q(s,a) = V(s) + A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'}A(s,a')$ |
| N-step return | $G_t^{(n)} = \sum_{k=0}^{n-1}\gamma^k r_{t+k} + \gamma^n \max_{a'}Q(s_{t+n},a';\theta^-)$ |
| Overestimation bound | $\mathbb{E}[\max_i Q_i] \approx Q^* + \sigma\sqrt{2\log n}$ |

---

## Connection to RL for Image Processing

These tricks are essential for practical image-based RL:

- **PER for image enhancement:** Some image degradations are harder than others. PER ensures the agent focuses on transitions where enhancement was surprisingly good or bad — accelerating learning on difficult cases.
- **Double DQN for stable enhancement:** Q-value overestimation can cause the agent to take overly aggressive enhancement actions (over-sharpening, over-saturating). Double DQN provides more accurate Q-estimates.
- **Dueling for state-aware decisions:** Some images are already near-perfect (high $V(s)$), requiring no action. Dueling DQN explicitly models this, learning that the value of a good state is high regardless of which action is taken.
- **N-step for credit assignment:** If sharpening an image helps a later denoising step produce better results, N-step returns allow this credit to flow directly, rather than requiring many 1-step backups.
- **Rainbow for production systems:** Real image processing agents (Module 07-10) benefit from combining all tricks for maximum performance.

---

## Prerequisites & Next Steps

**Prerequisites:**
- **6.1 DQN** (the base algorithm that all tricks improve)
- **6.2-6.3** (policy gradient and advantage — for understanding Dueling)
- Data structures: Binary trees (for sum tree in PER)

**Next Steps:**
- **Module 07** → Apply these advanced techniques to real image enhancement tasks
- **Module 09** → Use Dueling DQN for bounding box refinement
- **Module 11** → Combine with multi-agent and hierarchical RL for complex vision tasks
