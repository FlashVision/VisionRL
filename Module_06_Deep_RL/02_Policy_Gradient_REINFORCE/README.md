![Module Logo](../logo.png)

# Policy Gradient — REINFORCE

## Overview

Policy gradient methods directly parameterize and optimize the policy $\pi_\theta(a|s)$, enabling continuous action spaces and stochastic policies. This document derives the REINFORCE algorithm from the likelihood ratio trick, proves that baselines reduce variance without introducing bias, derives the natural policy gradient via the Fisher information matrix, and analyzes convergence properties.

## Prerequisites

- Value functions (Module 03.4)
- Neural network optimization (Module 05.1)
- Statistics (variance reduction techniques)

---

## 1. Mathematical Foundations

### 1.1 Core Definition — Policy Gradient Objective

The RL objective is to maximize the expected return:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right] = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

where $\tau = (s_0, a_0, r_0, s_1, \ldots)$ is a trajectory.

### 1.2 REINFORCE Derivation — Likelihood Ratio Trick

**Step 1:** Write the objective as an integral over trajectories:

$$J(\theta) = \int P(\tau; \theta) R(\tau) \, d\tau$$

where $P(\tau; \theta) = p(s_0)\prod_{t=0}^{T-1}\pi_\theta(a_t|s_t)T(s_{t+1}|s_t,a_t)$.

**Step 2:** Compute the gradient:

$$\nabla_\theta J = \int \nabla_\theta P(\tau; \theta) R(\tau) \, d\tau$$

**Step 3:** Apply the log-derivative trick: $\nabla_\theta P = P \cdot \nabla_\theta \log P$:

$$\nabla_\theta J = \int P(\tau; \theta) \nabla_\theta \log P(\tau; \theta) \, R(\tau) \, d\tau = \mathbb{E}_{\tau \sim \pi_\theta}[\nabla_\theta \log P(\tau; \theta) \, R(\tau)]$$

**Step 4:** Expand $\log P(\tau; \theta)$:

$$\log P(\tau; \theta) = \log p(s_0) + \sum_{t=0}^{T-1}\left[\log \pi_\theta(a_t|s_t) + \log T(s_{t+1}|s_t,a_t)\right]$$

**Step 5:** Since $\log p(s_0)$ and $\log T$ do not depend on $\theta$:

$$\nabla_\theta \log P(\tau; \theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)$$

**Step 6:** The REINFORCE gradient estimate:

$$\nabla_\theta J = \mathbb{E}_{\tau}\left[\sum_{t=0}^{T-1}\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)\right]$$

**Step 7 — Causality refinement.** Actions at time $t$ cannot affect rewards before time $t$:

$$\nabla_\theta J = \mathbb{E}_{\tau}\left[\sum_{t=0}^{T-1}\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

where $G_t = \sum_{k=t}^{T-1}\gamma^{k-t}r_k$ is the return from time $t$.

**Result (REINFORCE Gradient):**

$$\boxed{\nabla_\theta J = \mathbb{E}_\pi\left[\sum_{t=0}^{T-1}\nabla_\theta \log\pi_\theta(a_t|s_t) \cdot G_t\right]}$$
$\blacksquare$

### 1.3 Baseline Variance Reduction — Proof of Unbiasedness

**Theorem:** Subtracting any function $b(s_t)$ from the return does not change the expected gradient:

$$\mathbb{E}\left[\nabla_\theta\log\pi_\theta(a_t|s_t)(G_t - b(s_t))\right] = \mathbb{E}\left[\nabla_\theta\log\pi_\theta(a_t|s_t) \cdot G_t\right]$$

**Proof:**

**Step 1:** We need to show:

$$\mathbb{E}\left[\nabla_\theta\log\pi_\theta(a_t|s_t) \cdot b(s_t)\right] = 0$$

**Step 2:** Condition on $s_t$:

$$\mathbb{E}_{a_t \sim \pi_\theta(\cdot|s_t)}\left[\nabla_\theta\log\pi_\theta(a_t|s_t) \cdot b(s_t)\right] = b(s_t)\sum_a \nabla_\theta\pi_\theta(a|s_t) \cdot \frac{1}{\pi_\theta(a|s_t)} \cdot \pi_\theta(a|s_t)$$

Wait — let me redo this correctly:

$$= b(s_t)\sum_a \pi_\theta(a|s_t) \nabla_\theta\log\pi_\theta(a|s_t) = b(s_t)\sum_a \nabla_\theta\pi_\theta(a|s_t)$$

**Step 3:** Since $\sum_a \pi_\theta(a|s_t) = 1$ for all $\theta$:

$$\nabla_\theta \sum_a \pi_\theta(a|s_t) = \sum_a \nabla_\theta \pi_\theta(a|s_t) = \nabla_\theta 1 = 0$$

**Step 4:** Therefore $\mathbb{E}[\nabla_\theta\log\pi \cdot b(s_t)] = b(s_t) \cdot 0 = 0$.

**Result:** The baseline $b(s_t)$ does not introduce bias but reduces variance when $b(s_t) \approx V^\pi(s_t)$.
$\blacksquare$

**Optimal baseline derivation:** The variance-minimizing baseline is:

$$b^*(s) = \frac{\mathbb{E}\left[\|\nabla_\theta\log\pi\|^2 G_t\right]}{\mathbb{E}\left[\|\nabla_\theta\log\pi\|^2\right]}$$

which is a weighted average of returns, approximated in practice by $V^\pi(s)$.

### 1.4 Natural Policy Gradient Derivation

**Problem:** Standard gradient descent uses the Euclidean metric in parameter space, which doesn't respect the geometry of probability distributions.

**Step 1:** Define the Fisher Information Matrix (FIM):

$$\mathbf{F}(\theta) = \mathbb{E}_{s \sim d^\pi, a \sim \pi_\theta}\left[\nabla_\theta\log\pi_\theta(a|s)\nabla_\theta\log\pi_\theta(a|s)^T\right]$$

**Step 2:** The FIM is the Hessian of the KL divergence at $\theta' = \theta$:

$$D_{\text{KL}}(\pi_\theta \| \pi_{\theta'}) \approx \frac{1}{2}(\theta' - \theta)^T\mathbf{F}(\theta)(\theta' - \theta)$$

**Step 3:** Natural gradient descent minimizes $J$ subject to a KL constraint:

$$\max_{\Delta\theta} \; \nabla_\theta J^T \Delta\theta \quad \text{s.t.} \quad \frac{1}{2}\Delta\theta^T\mathbf{F}\Delta\theta \leq \epsilon$$

**Step 4:** By the method of Lagrange multipliers:

$$\mathcal{L} = \nabla J^T\Delta\theta - \lambda(\frac{1}{2}\Delta\theta^T\mathbf{F}\Delta\theta - \epsilon)$$

$$\nabla_{\Delta\theta}\mathcal{L} = \nabla J - \lambda\mathbf{F}\Delta\theta = 0 \implies \Delta\theta = \frac{1}{\lambda}\mathbf{F}^{-1}\nabla J$$

**Step 5:** The natural gradient is:

$$\tilde{\nabla}_\theta J = \mathbf{F}^{-1}\nabla_\theta J$$

**Result:**

$$\boxed{\Delta\theta_{\text{natural}} = \alpha \mathbf{F}(\theta)^{-1}\nabla_\theta J(\theta)}$$
$\blacksquare$

**Intuition:** The natural gradient takes equal-sized steps in distribution space (KL divergence) rather than parameter space (Euclidean distance). This is the foundation for TRPO (Module 06.4).

### 1.5 High Variance Problem

The variance of the REINFORCE estimator scales with the trajectory length:

$$\text{Var}(\hat{g}_{\text{REINFORCE}}) = O\!\left(\frac{T \cdot \text{Var}(G_t)}{N_{\text{episodes}}}\right)$$

where $\text{Var}(G_t) = O(R_{\max}^2/(1-\gamma)^2)$. This makes REINFORCE impractical for long-horizon tasks without variance reduction (baselines, actor-critic).

---

## 2. Algorithm / Method

### 2.1 Pseudocode: REINFORCE with Baseline

```
Algorithm: REINFORCE_with_Baseline
Input: Policy π_θ, value network V_φ, learning rates α_θ, α_φ
Output: Trained policy

1. For episode = 1 to N:
     Generate trajectory τ = (s_0,a_0,r_0,...,s_T) using π_θ
     For t = T-1 down to 0:
       G_t = r_t + γ·G_{t+1}  (G_T = 0)
     For t = 0 to T-1:
       δ_t = G_t - V_φ(s_t)
       θ = θ + α_θ · γ^t · δ_t · ∇_θ log π_θ(a_t|s_t)
       φ = φ - α_φ · ∇_φ (G_t - V_φ(s_t))²
2. Return π_θ
```

### 2.2 Complexity Analysis

- **Per episode:** $O(T \cdot d_\theta)$ for policy gradient, $O(T \cdot d_\phi)$ for value update
- **Sample efficiency:** Poor — requires many complete episodes
- **Variance:** High without baseline; moderate with $V^\pi$ baseline

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

- **State:** $s_t = \mathbf{I}_t$ — image observation
- **Action:** $a_t \sim \pi_\theta(\cdot|s_t)$ — stochastic policy output
- **Reward:** Task-specific signal
- **Transition:** Environment dynamics

### 3.2 Why RL?

Policy gradients handle continuous action spaces (unlike DQN) and naturally output stochastic policies (useful for exploration). The REINFORCE derivation is the foundation for all policy-based deep RL: A2C, PPO, SAC.

---

## 4. Dataset

- **Name:** Custom image processing environments
- **Size:** Generated through interaction
- **Auto-download:**

```python
from torchvision import datasets
cifar = datasets.CIFAR10('./data', download=True)
```

---

## 5. Key Equations Summary

| Equation | Meaning |
|----------|---------|
| $\nabla J = \mathbb{E}[\sum_t \nabla\log\pi \cdot G_t]$ | REINFORCE gradient |
| $\mathbb{E}[\nabla\log\pi \cdot b(s)] = 0$ | Baseline unbiasedness |
| $\tilde{\nabla}J = \mathbf{F}^{-1}\nabla J$ | Natural policy gradient |
| $\mathbf{F} = \mathbb{E}[\nabla\log\pi \cdot \nabla\log\pi^T]$ | Fisher information matrix |

---

## 6. References

- Williams, R. J. "Simple Statistical Gradient-Following Algorithms for Connectionist RL," *Machine Learning*, 8:229–256, 1992.
- Kakade, S. M. "A Natural Policy Gradient," *NeurIPS*, 2002.
- Sutton, R. S. et al. "Policy Gradient Methods for RL with Function Approximation," *NeurIPS*, 2000.
- Peters, J. & Schaal, S. "Natural Actor-Critic," *Neurocomputing*, 71(7–9):1180–1190, 2008.
