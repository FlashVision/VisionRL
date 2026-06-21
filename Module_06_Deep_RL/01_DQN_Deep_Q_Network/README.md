![Module Logo](../logo.png)

# DQN — Deep Q-Network

## Overview

DQN (Mnih et al., 2015) combines Q-learning with deep neural networks to handle high-dimensional state spaces like images. This document derives the DQN loss function from the Bellman equation, proves why experience replay breaks temporal correlation, provides a stability analysis of the target network, and compares Huber loss to MSE with full derivations.

## Prerequisites

- Q-learning (Module 04.5)
- Neural networks and CNNs (Module 05.1–05.2)
- Bellman equations (Module 03.3)

---

## 1. Mathematical Foundations

### 1.1 Core Definition — DQN

Approximate $Q^*(s, a)$ with a neural network $Q(s, a; \theta)$. The network takes image state $s$ as input and outputs Q-values for all actions.

### 1.2 DQN Loss Derivation

**Step 1:** The Bellman optimality equation states:

$$Q^*(s, a) = \mathbb{E}_{s'}\left[r + \gamma \max_{a'} Q^*(s', a') \;\middle|\; s, a\right]$$

**Step 2:** Define the target:

$$y_i = r_i + \gamma \max_{a'} Q(s'_i, a'; \theta^-)$$

where $\theta^-$ are frozen target network parameters.

**Step 3:** The DQN loss minimizes the squared temporal difference error over a mini-batch $\mathcal{B}$ sampled from replay buffer $\mathcal{D}$:

$$\mathcal{L}(\theta) = \frac{1}{|\mathcal{B}|}\sum_{(s,a,r,s',d) \in \mathcal{B}} \left(y - Q(s, a; \theta)\right)^2$$

where $y = r + \gamma(1-d)\max_{a'} Q(s', a'; \theta^-)$ and $d$ is the terminal flag.

**Step 4:** The gradient (semi-gradient, not through the target):

$$\nabla_\theta \mathcal{L} = -\frac{2}{|\mathcal{B}|}\sum_{\mathcal{B}} \left(y - Q(s, a; \theta)\right) \nabla_\theta Q(s, a; \theta)$$

**Step 5:** This is a semi-gradient method because $y$ depends on $\theta^-$ (frozen), not $\theta$. Differentiating through $y$ would give the full gradient but would destabilize training.

**Result:**

$$\boxed{\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\!\left[\left(r + \gamma\max_{a'}Q(s',a';\theta^-) - Q(s,a;\theta)\right)^2\right]}$$
$\blacksquare$

### 1.3 Experience Replay — Proof of Correlation Breaking

**Problem:** Training on consecutive transitions $\{(s_t, a_t, r_t, s_{t+1})\}$ violates the i.i.d. assumption of SGD because $s_t$ and $s_{t+1}$ are correlated.

**Theorem:** Uniform random sampling from a replay buffer produces uncorrelated samples.

**Proof:**

**Step 1:** Let the replay buffer $\mathcal{D} = \{e_1, e_2, \ldots, e_N\}$ contain $N$ transitions collected over time.

**Step 2:** Sample a mini-batch $\mathcal{B} = \{e_{i_1}, e_{i_2}, \ldots, e_{i_B}\}$ where indices $i_k$ are drawn uniformly at random from $\{1, \ldots, N\}$ without replacement.

**Step 3:** For any two sampled transitions $e_{i_j}$ and $e_{i_k}$ with $j \neq k$:

$$P(i_j = m, i_k = n) = \frac{1}{N(N-1)} \quad \forall m \neq n$$

**Step 4:** The marginal distribution: $P(i_j = m) = 1/N$ for all $m$.

**Step 5:** Check independence:

$$P(i_j = m, i_k = n) = \frac{1}{N(N-1)} \neq \frac{1}{N^2} = P(i_j = m)P(i_k = n)$$

The samples are not independent (sampling without replacement), but the dependence vanishes as $N \to \infty$: $\frac{1}{N(N-1)} \approx \frac{1}{N^2}$.

**Step 6:** Crucially, temporal correlation is broken: even if $e_m$ and $e_{m+1}$ are correlated in the buffer (consecutive frames), the probability that both appear in the mini-batch is only $B^2 / N^2 \ll 1$ for $B \ll N$.

**Step 7:** The effective correlation between mini-batch samples is:

$$\text{Corr}(e_{i_j}, e_{i_k}) = \frac{1}{N}\sum_{m=1}^N \text{Corr}(e_m, e_m) + \frac{N-1}{N}\bar{\text{Corr}}_{\text{off-diag}} \approx \bar{\text{Corr}}_{\text{off-diag}} \approx 0$$

for a large, diverse buffer.
$\blacksquare$

### 1.4 Target Network Stability Analysis

**Step 1:** Without a target network, the loss is:

$$\mathcal{L}(\theta) = \mathbb{E}\left[(r + \gamma\max_{a'}Q(s',a';\theta) - Q(s,a;\theta))^2\right]$$

Both the prediction and target depend on $\theta$, creating a moving target.

**Step 2 — Soft update (Polyak averaging):**

$$\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$$

**Step 3:** The effective target parameters at step $t$ are the exponential moving average:

$$\theta_t^- = \tau\sum_{k=0}^{t}(1-\tau)^{t-k}\theta_k$$

**Step 4:** The target changes at rate $\tau$ per step. The prediction changes at rate $\alpha$ (learning rate) per step. Stability requires $\tau \ll \alpha$ so the target changes slower than the prediction.

**Step 5 — Convergence condition (informal):** The system converges when the update-to-target-drift ratio satisfies:

$$\frac{\alpha \|\nabla_\theta\mathcal{L}\|}{\tau \|\theta - \theta^-\|} > 1$$

The learning must make progress faster than the target moves.

### 1.5 Huber Loss vs. MSE Analysis

**MSE loss:** $\ell_{\text{MSE}}(\delta) = \delta^2$, gradient $= 2\delta$.

**Huber loss:**

$$\ell_{\text{Huber}}(\delta) = \begin{cases} \frac{1}{2}\delta^2 & \text{if } |\delta| \leq \kappa \\ \kappa(|\delta| - \frac{1}{2}\kappa) & \text{if } |\delta| > \kappa \end{cases}$$

**Gradient:**

$$\frac{\partial \ell_{\text{Huber}}}{\partial \delta} = \begin{cases} \delta & \text{if } |\delta| \leq \kappa \\ \kappa \cdot \text{sign}(\delta) & \text{if } |\delta| > \kappa \end{cases}$$

**Advantage of Huber loss:** For large TD errors (common early in training), MSE gives gradients proportional to $\delta$, which can be very large and cause instability. Huber loss clips gradients to $\pm\kappa$, providing robustness to outlier transitions while preserving the quadratic (efficient) behavior near the minimum.

**Continuity verification at $|\delta| = \kappa$:**
- Quadratic: $\frac{1}{2}\kappa^2$
- Linear: $\kappa(\kappa - \frac{\kappa}{2}) = \frac{1}{2}\kappa^2$ $\checkmark$

Both values and derivatives match, so Huber loss is $C^1$-smooth.

---

## 2. Algorithm / Method

### 2.1 Pseudocode: DQN Training

```
Algorithm: DQN
Input: Environment env, network Q_θ, target Q_{θ⁻}, buffer D, episodes N
Output: Trained Q-network

1. θ⁻ = θ (initialize target)
2. For episode = 1 to N:
     s = env.reset()
     For each step:
       a = ε-greedy(Q_θ, s)
       s', r, done = env.step(a)
       D.push(s, a, r, s', done)
       Sample batch B from D
       y = r + γ(1-done)·max_{a'} Q_{θ⁻}(s', a')
       L = mean((y - Q_θ(s,a))²)
       θ = θ - α·∇_θ L
       θ⁻ = τθ + (1-τ)θ⁻  (soft update)
       s = s'
3. Return Q_θ
```

### 2.2 Complexity Analysis

- **Per step:** $O(|\mathcal{B}| \cdot d_{\text{network}})$ — forward + backward pass per batch
- **Replay buffer:** $O(|\mathcal{D}| \cdot d_{\text{state}})$ — memory for stored transitions
- **Space:** $O(2 \cdot d_{\text{network}})$ — online + target network

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

- **State:** $s_t = \mathbf{I}_t \in \mathbb{R}^{C \times H \times W}$ — image observation
- **Action:** $a_t \in \{a_1, \ldots, a_K\}$ — discrete action (e.g., image enhancement operation)
- **Reward:** $r_t = \text{PSNR}(s_{t+1}, s_{\text{clean}}) - \text{PSNR}(s_t, s_{\text{clean}})$
- **Transition:** Application of the chosen enhancement action

### 3.2 Why RL?

DQN is the foundational deep RL algorithm for image-based tasks. It handles the exponentially large state space of images through function approximation, and experience replay + target networks provide the stability needed for convergence.

---

## 4. Dataset

- **Name:** CIFAR-10 (for custom ImageEnhancementEnv)
- **Size:** 50K images, synthetically degraded
- **Auto-download:**

```python
from torchvision import datasets
cifar = datasets.CIFAR10(root='./data', train=True, download=True)
```

---

## 5. Key Equations Summary

| Equation | Meaning |
|----------|---------|
| $\mathcal{L} = \mathbb{E}[(y - Q(s,a;\theta))^2]$ | DQN loss |
| $y = r + \gamma\max_{a'}Q(s',a';\theta^-)$ | TD target with target network |
| $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$ | Soft target update |
| $\ell_{\text{Huber}}(\delta) = \frac{1}{2}\delta^2$ for $|\delta|\leq\kappa$ | Huber loss (quadratic region) |

---

## 6. References

- Mnih, V. et al. "Human-Level Control through Deep Reinforcement Learning," *Nature*, 518:529–533, 2015.
- Lin, L.-J. "Self-Improving Reactive Agents Based on RL, Planning and Teaching," *Machine Learning*, 8:293–321, 1992.
- Hasselt, H. van, Guez, A., & Silver, D. "Deep Reinforcement Learning with Double Q-learning," *AAAI*, 2016.
- Huber, P. J. "Robust Estimation of a Location Parameter," *Annals of Mathematical Statistics*, 35(1):73–101, 1964.
