![Module Logo](../logo.png)

# 6.4 PPO — Proximal Policy Optimization

## Overview

This notebook derives PPO from its theoretical foundation in Trust Region Policy Optimization (TRPO), showing why constraining policy updates is necessary, how the clipped surrogate objective achieves this without second-order optimization, and why clipping creates a pessimistic bound that guarantees monotonic improvement. We also derive KL divergence from first principles and present the adaptive KL penalty variant.

---

## Mathematical Foundation

### 1. The Problem with Large Policy Updates

**Issue:** In REINFORCE/A2C, the policy gradient update $\theta \leftarrow \theta + \alpha\nabla J(\theta)$ can take arbitrarily large steps. A large step can:

1. **Collapse the policy** — all probability mass on one action
2. **Move to a bad region** — from which recovery is difficult (no replay buffer in on-policy methods)
3. **Invalidate the gradient estimate** — the gradient was computed under $\pi_{\text{old}}$, but the update moved to a very different $\pi_{\text{new}}$

**Key insight:** We need to control *how much* the policy changes per update.

### 2. TRPO — Trust Region Policy Optimization

**Theoretical foundation (Kakade & Langford, 2002; Schulman et al., 2015):**

The expected return under a new policy $\pi_{\text{new}}$ can be decomposed as:

$$J(\pi_{\text{new}}) = J(\pi_{\text{old}}) + \mathbb{E}_{s \sim d^{\pi_{\text{new}}}, a \sim \pi_{\text{new}}}\left[A^{\pi_{\text{old}}}(s,a)\right]$$

where $d^{\pi_{\text{new}}}$ is the state visitation distribution under the new policy.

**Problem:** We can't sample from $d^{\pi_{\text{new}}}$ (we haven't deployed it yet). So we approximate with $d^{\pi_{\text{old}}}$ and use importance sampling:

$$L(\theta) = \mathbb{E}_{s \sim d^{\pi_{\text{old}}}, a \sim \pi_{\text{old}}}\left[\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}A^{\pi_{\text{old}}}(s,a)\right]$$

$$= \mathbb{E}\left[r_t(\theta) \cdot A_t\right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ is the **probability ratio**.

**TRPO constraint:** The approximation $L(\theta) \approx J(\pi_{\text{new}}) - J(\pi_{\text{old}})$ is accurate only when $\pi_\theta \approx \pi_{\text{old}}$. TRPO enforces this with a KL constraint:

$$\max_\theta \; \mathbb{E}\left[r_t(\theta) A_t\right] \quad \text{s.t.} \quad \mathbb{E}\left[D_{\text{KL}}(\pi_{\text{old}}(\cdot|s) \| \pi_\theta(\cdot|s))\right] \leq \delta$$

**TRPO solution (conjugate gradient + line search):** This is a constrained optimization problem solved with:
1. Compute the policy gradient $g = \nabla_\theta L(\theta)$
2. Compute the Fisher Information Matrix $F = \nabla^2_\theta \mathbb{E}[D_{\text{KL}}]$ (second-order!)
3. Compute the natural gradient $\tilde{g} = F^{-1}g$ via conjugate gradient
4. Line search to find the largest step satisfying the KL constraint

**Problem with TRPO:** Computing the Fisher matrix and conjugate gradient is computationally expensive and hard to implement correctly.

### 3. KL Divergence — Full Derivation

**Definition:** For discrete distributions $p$ and $q$:

$$D_{\text{KL}}(p \| q) = \sum_{x} p(x)\log\frac{p(x)}{q(x)}$$

**Properties:**
- $D_{\text{KL}}(p \| q) \geq 0$ (Gibbs' inequality), with equality iff $p = q$
- **NOT symmetric:** $D_{\text{KL}}(p\|q) \neq D_{\text{KL}}(q\|p)$
- NOT a true metric (no triangle inequality)

**Proof of non-negativity (via Jensen's inequality):**

$$D_{\text{KL}}(p\|q) = -\sum_x p(x)\log\frac{q(x)}{p(x)} \geq -\log\sum_x p(x)\frac{q(x)}{p(x)} = -\log\sum_x q(x) = -\log 1 = 0$$

using Jensen's inequality on the concave $\log$ function. $\blacksquare$

**KL for softmax policies:** If $\pi_{\text{old}}(a|s) = \text{softmax}(z^{\text{old}})_a$ and $\pi_\theta(a|s) = \text{softmax}(z^\theta)_a$:

$$D_{\text{KL}}(\pi_{\text{old}} \| \pi_\theta) = \sum_a \pi_{\text{old}}(a|s)\left[\log\pi_{\text{old}}(a|s) - \log\pi_\theta(a|s)\right]$$

$$= \sum_a \pi_{\text{old}}(a|s)\left[(z_a^{\text{old}} - \log Z^{\text{old}}) - (z_a^\theta - \log Z^\theta)\right]$$

### 4. PPO Clipped Objective — The Key Innovation

**PPO replaces the KL constraint with clipping:**

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)A_t, \; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)A_t\right)\right]$$

where $\varepsilon$ is a small hyperparameter (typically $\varepsilon = 0.2$).

**How it works — case analysis:**

**Case 1: $A_t > 0$ (good action, want to increase probability):**

- $r_t$ increases as we increase $\pi_\theta(a_t|s_t)$
- $r_t A_t$ increases (good!)
- But $\text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)A_t$ caps the benefit at $(1+\varepsilon)A_t$
- $\min(\cdot, \cdot)$ takes the smaller → **clips excessive increase**

$$L^{\text{CLIP}} = \begin{cases} r_t A_t & \text{if } r_t \leq 1+\varepsilon \\ (1+\varepsilon)A_t & \text{if } r_t > 1+\varepsilon \end{cases}$$

**Case 2: $A_t < 0$ (bad action, want to decrease probability):**

- $r_t$ decreases as we decrease $\pi_\theta(a_t|s_t)$
- $r_t A_t$ increases (less negative × negative = less negative, which is improvement)
- But clipping caps at $(1-\varepsilon)A_t$
- $\min(\cdot, \cdot)$ takes the smaller (more negative) → **clips excessive decrease**

$$L^{\text{CLIP}} = \begin{cases} r_t A_t & \text{if } r_t \geq 1-\varepsilon \\ (1-\varepsilon)A_t & \text{if } r_t < 1-\varepsilon \end{cases}$$

**Why clipping creates a pessimistic bound:**

The clipped objective is a **lower bound** on the unclipped surrogate:

$$\mathcal{L}^{\text{CLIP}}(\theta) \leq \mathcal{L}^{\text{CPI}}(\theta) = \mathbb{E}[r_t(\theta)A_t]$$

By maximizing a lower bound, PPO makes conservative updates — it never overestimates the improvement. This is the **pessimistic bound** property.

### 5. Adaptive KL Penalty Variant

An alternative to clipping uses a dynamic KL penalty:

$$\mathcal{L}^{\text{KLPEN}}(\theta) = \mathbb{E}\left[r_t(\theta)A_t - \beta D_{\text{KL}}(\pi_{\text{old}}(\cdot|s_t) \| \pi_\theta(\cdot|s_t))\right]$$

The penalty coefficient $\beta$ is adapted:

$$\beta \leftarrow \begin{cases} \beta / 2 & \text{if } D_{\text{KL}} < d_{\text{target}} / 1.5 \\ \beta \times 2 & \text{if } D_{\text{KL}} > d_{\text{target}} \times 1.5 \\ \beta & \text{otherwise} \end{cases}$$

This automatically tunes $\beta$ to keep the KL divergence near $d_{\text{target}}$.

### 6. Complete PPO Objective

$$\mathcal{L}^{\text{PPO}}(\theta, \phi) = \mathbb{E}_t\left[\mathcal{L}_t^{\text{CLIP}}(\theta) - c_1\mathcal{L}_t^{\text{VF}}(\phi) + c_2 H(\pi_\theta(\cdot|s_t))\right]$$

where:
- $\mathcal{L}_t^{\text{VF}} = (V_\phi(s_t) - G_t^{\text{target}})^2$ (value function loss)
- $H(\pi_\theta)$ is the entropy bonus
- $c_1 = 0.5$, $c_2 = 0.01$ (typical values)

**Value function clipping (optional but common):**

$$\mathcal{L}^{\text{VF,CLIP}} = \max\left[(V_\phi(s_t) - G_t)^2, (\text{clip}(V_\phi(s_t), V_{\text{old}}(s_t) \pm \varepsilon_V) - G_t)^2\right]$$

### 7. PPO Training Loop

```
For iteration k = 1, 2, ...:
    Collect T timesteps with current policy π_θ_old
    Compute advantages Â_t using GAE(λ)
    For epoch e = 1, ..., E:
        For mini-batch B ⊂ {1,...,T}:
            Compute r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
            Compute L^CLIP, L^VF, entropy H
            Update θ via gradient ascent on L^PPO
    θ_old ← θ
```

Key detail: PPO performs **multiple epochs** of gradient updates on the same batch of data (typically $E = 3$-$10$). The clipping prevents the policy from changing too much during these reuses.

---

## Step-by-Step Breakdown

1. **Implement probability ratio:** Given old log-probs $\log\pi_{\text{old}}(a_t|s_t)$ (stored during collection) and current log-probs, compute $r_t(\theta) = \exp(\log\pi_\theta - \log\pi_{\text{old}})$.

2. **Implement clipped objective:** Code the $\min(r_t A_t, \text{clip}(r_t, 1\pm\varepsilon)A_t)$ formula. Visualize $\mathcal{L}^{\text{CLIP}}$ as a function of $r_t$ for positive and negative advantages.

3. **Implement GAE:** Reuse the recursive GAE computation from A2C (Module 6.3). Compare $\lambda \in \{0.9, 0.95, 0.99\}$.

4. **Build PPO agent:** Combine actor-critic network, GAE, clipped objective, value loss, entropy bonus. Implement the multi-epoch update loop.

5. **Train on ImageEnhancementEnv:** Multi-step image enhancement with PPO. The agent applies a sequence of operations to improve CIFAR-10 images. Plot: episode reward, policy entropy, KL divergence, clip fraction.

6. **Compare PPO vs A2C:** Same environment, same hyperparameters (where applicable). Show PPO's improved stability (less variance in learning curves).

7. **Ablation studies:** Remove clipping (pure surrogate), remove entropy bonus, vary $\varepsilon \in \{0.1, 0.2, 0.3\}$, vary number of epochs $E \in \{1, 3, 10\}$. Show how each component affects training stability.

---

## Dataset Used

**ImageEnhancementEnv — Multi-step Enhancement (CIFAR-10)**
- **Source:** `torchvision.datasets.CIFAR10` (auto-downloads)
- **Environment:** Multi-step enhancement (up to 10 steps per episode)
- **Actions:** 8 enhancement operations (same as A2C)
- **Reward:** Per-step ΔPSNR + terminal bonus for quality threshold
- **PPO-specific:** Parallel environments (vectorized) for efficient data collection

```python
from torchvision import datasets

cifar = datasets.CIFAR10(root='./data', train=True, download=True)

# PPO typically uses multiple parallel environments
NUM_ENVS = 8
envs = [ImageEnhancementEnv(cifar) for _ in range(NUM_ENVS)]
```

---

## Key Equations Summary

| Concept | Equation |
|---------|----------|
| Probability ratio | $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\text{old}}(a_t|s_t)$ |
| PPO clipped objective | $L^{\text{CLIP}} = \mathbb{E}[\min(r_t A_t, \text{clip}(r_t, 1\pm\varepsilon)A_t)]$ |
| TRPO constraint | $\max \mathbb{E}[r_t A_t]$ s.t. $\mathbb{E}[D_{\text{KL}}] \leq \delta$ |
| KL divergence | $D_{\text{KL}}(p\|q) = \sum p\log(p/q)$ |
| Adaptive KL penalty | $L = \mathbb{E}[r_t A_t] - \beta D_{\text{KL}}$ |
| Full PPO loss | $L^{\text{CLIP}} - c_1 L^{\text{VF}} + c_2 H(\pi)$ |

---

## Connection to RL for Image Processing

PPO is the **workhorse algorithm** for image-based RL in practice:

- **Stability:** PPO's clipped objective prevents catastrophic policy updates that are common in image processing environments where rewards can vary dramatically (one bad enhancement ruins an image).
- **Multi-step enhancement:** PPO excels at sequential decision-making — applying a sequence of image operations where each step's effect depends on previous choices.
- **RLHF for image generation (Module 10.4):** PPO is the standard algorithm for Reinforcement Learning from Human Feedback, used in text-to-image models like DALL-E and Stable Diffusion to align outputs with human preferences.
- **Sample efficiency:** The multi-epoch reuse of collected data makes PPO more sample-efficient than REINFORCE, which is critical when each environment step involves expensive image operations.

---

## Prerequisites & Next Steps

**Prerequisites:**
- **6.3 Actor-Critic (A2C)** (advantage estimation, GAE, actor-critic architecture)
- **6.2 REINFORCE** (policy gradient theorem, importance sampling)
- Optimization: Constrained optimization, Lagrange multipliers (for understanding TRPO)

**Next Steps:**
- **6.5 Experience Replay & Tricks** → Advanced DQN improvements (Double, Dueling, PER)
- **Module 07-10** → Apply PPO to real image processing tasks
- **Module 10.4** → PPO for RLHF in text-to-image generation
