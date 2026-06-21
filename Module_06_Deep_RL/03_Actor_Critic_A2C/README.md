![Module Logo](../logo.png)

# Actor-Critic — A2C

## Overview

Actor-critic methods combine the strengths of policy gradient (actor) with value function estimation (critic), achieving lower variance than REINFORCE while maintaining the ability to handle continuous actions. This document derives the A2C algorithm by combining the policy gradient with a learned value baseline, provides the full derivation of Generalized Advantage Estimation (GAE-$\lambda$), and analyzes the bias-variance tradeoff controlled by $\lambda$.

## Prerequisites

- Policy gradient / REINFORCE (Module 06.2)
- TD learning (Module 04.4)
- Value functions and advantage (Module 03.4)

---

## 1. Mathematical Foundations

### 1.1 Core Definition — Actor-Critic Architecture

- **Actor:** Policy network $\pi_\theta(a|s)$ — selects actions
- **Critic:** Value network $V_\phi(s)$ — evaluates states

### 1.2 A2C Derivation

**Step 1:** From the policy gradient theorem (Module 06.2):

$$\nabla_\theta J = \mathbb{E}_\pi\left[\sum_t \nabla_\theta\log\pi_\theta(a_t|s_t) \cdot (G_t - b(s_t))\right]$$

**Step 2:** The optimal baseline is $b(s_t) = V^\pi(s_t)$. The quantity $G_t - V^\pi(s_t)$ is a sample estimate of the advantage:

$$A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)$$

**Step 3:** Replace $V^\pi$ with a learned critic $V_\phi$:

$$\nabla_\theta J \approx \mathbb{E}_\pi\left[\sum_t \nabla_\theta\log\pi_\theta(a_t|s_t) \cdot \hat{A}_t\right]$$

**Step 4:** The simplest advantage estimate (one-step TD):

$$\hat{A}_t^{(1)} = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) = \delta_t$$

This is just the TD error. It has low variance (single-step) but high bias (depends on $V_\phi$ accuracy).

**Step 5:** The $n$-step advantage estimate:

$$\hat{A}_t^{(n)} = \sum_{k=0}^{n-1}\gamma^k r_{t+k} + \gamma^n V_\phi(s_{t+n}) - V_\phi(s_t)$$

This interpolates between TD(0) bias ($n = 1$) and MC variance ($n = T$).

### 1.3 Full Derivation of GAE-$\lambda$

**Goal:** A single estimator that smoothly interpolates between bias and variance by weighting all $n$-step advantages.

**Step 1:** Define the $n$-step advantage:

$$\hat{A}_t^{(n)} = -V_\phi(s_t) + r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1}r_{t+n-1} + \gamma^n V_\phi(s_{t+n})$$

**Step 2:** Express in terms of TD errors $\delta_k = r_k + \gamma V_\phi(s_{k+1}) - V_\phi(s_k)$:

$$\hat{A}_t^{(1)} = \delta_t$$

$$\hat{A}_t^{(2)} = \delta_t + \gamma\delta_{t+1}$$

By induction:

$$\hat{A}_t^{(n)} = \sum_{k=0}^{n-1}\gamma^k \delta_{t+k}$$

**Proof:** We show $\hat{A}_t^{(n)} = \sum_{k=0}^{n-1}\gamma^k\delta_{t+k}$.

Expand $\sum_{k=0}^{n-1}\gamma^k\delta_{t+k} = \sum_{k=0}^{n-1}\gamma^k[r_{t+k} + \gamma V(s_{t+k+1}) - V(s_{t+k})]$.

The $V$ terms telescope: $-V(s_t) + \gamma V(s_{t+1}) - \gamma V(s_{t+1}) + \gamma^2 V(s_{t+2}) - \cdots + \gamma^n V(s_{t+n})$.

Remaining: $-V(s_t) + \sum_{k=0}^{n-1}\gamma^k r_{t+k} + \gamma^n V(s_{t+n}) = \hat{A}_t^{(n)}$. $\checkmark$

**Step 3:** Define GAE as the exponentially-weighted average of all $n$-step advantages:

$$\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\hat{A}_t^{(n)}$$

**Step 4:** Substitute $\hat{A}_t^{(n)} = \sum_{k=0}^{n-1}\gamma^k\delta_{t+k}$:

$$\hat{A}_t^{\text{GAE}} = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\sum_{k=0}^{n-1}\gamma^k\delta_{t+k}$$

**Step 5:** Exchange the order of summation. For each $\delta_{t+k}$, it appears in all $\hat{A}^{(n)}$ with $n \geq k+1$:

$$= (1-\lambda)\sum_{k=0}^{\infty}\gamma^k\delta_{t+k}\sum_{n=k+1}^{\infty}\lambda^{n-1}$$

**Step 6:** Evaluate the inner geometric series:

$$\sum_{n=k+1}^{\infty}\lambda^{n-1} = \frac{\lambda^k}{1-\lambda}$$

**Step 7:** Substitute:

$$\hat{A}_t^{\text{GAE}} = (1-\lambda)\sum_{k=0}^{\infty}\gamma^k\delta_{t+k}\frac{\lambda^k}{1-\lambda} = \sum_{k=0}^{\infty}(\gamma\lambda)^k\delta_{t+k}$$

**Result:**

$$\boxed{\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{k=0}^{\infty}(\gamma\lambda)^k\delta_{t+k}, \quad \delta_k = r_k + \gamma V_\phi(s_{k+1}) - V_\phi(s_k)}$$
$\blacksquare$

### 1.4 Bias-Variance Tradeoff of $\lambda$

**$\lambda = 0$:** $\hat{A}_t = \delta_t$ — pure TD (one step).
- Bias: High (depends on $V_\phi$ accuracy)
- Variance: Low (single reward, single $V$ evaluation)

**$\lambda = 1$:** $\hat{A}_t = \sum_{k=0}^{\infty}\gamma^k\delta_{t+k} = G_t - V_\phi(s_t)$ — MC return minus baseline.
- Bias: Zero (if $V_\phi = V^\pi$, or asymptotically as $T \to \infty$)
- Variance: High (full trajectory return)

**Intermediate $\lambda$:** Smooth interpolation.

**Formal bias-variance decomposition:**

$$\text{Bias}(\hat{A}_t^{\text{GAE}}) = O\left(\frac{\gamma\lambda}{1-\gamma\lambda}\right) \cdot \max_s\|V_\phi(s) - V^\pi(s)\|$$

$$\text{Var}(\hat{A}_t^{\text{GAE}}) = O\left(\frac{1}{(1-\gamma\lambda)^2}\right) \cdot \text{Var}(\delta)$$

Increasing $\lambda$ decreases bias but increases variance (longer effective horizon $(1-\gamma\lambda)^{-1}$).

### 1.5 Critic Loss

The critic minimizes the TD error:

$$\mathcal{L}_{\text{critic}}(\phi) = \frac{1}{2}\mathbb{E}\left[(G_t^\lambda - V_\phi(s_t))^2\right]$$

where $G_t^\lambda = \hat{A}_t^{\text{GAE}} + V_\phi(s_t)$ is the $\lambda$-return target.

---

## 2. Algorithm / Method

### 2.1 Pseudocode: A2C

```
Algorithm: A2C (Advantage Actor-Critic)
Input: Actor π_θ, Critic V_φ, GAE parameter λ
Output: Trained policy

1. For iteration = 1 to N:
     Collect T steps of experience using π_θ
     Compute TD errors: δ_t = r_t + γV_φ(s_{t+1}) - V_φ(s_t)
     Compute GAE: Â_t = Σ_k (γλ)^k δ_{t+k}
     Compute returns: G_t = Â_t + V_φ(s_t)

     Actor loss: L_actor = -mean(log π_θ(a_t|s_t) · Â_t)
     Critic loss: L_critic = mean((G_t - V_φ(s_t))²)
     Entropy bonus: L_entropy = -mean(H(π_θ(·|s_t)))

     L = L_actor + c₁·L_critic - c₂·L_entropy
     Update θ, φ by gradient descent on L
2. Return π_θ
```

### 2.2 Complexity Analysis

- **Per iteration:** $O(T \cdot (d_\theta + d_\phi))$ — forward/backward through both networks
- **GAE computation:** $O(T)$ — single backward pass through TD errors
- **Memory:** $O(T \cdot d_{\text{state}})$ for trajectory buffer

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

- **State:** $s_t$ — image or feature vector
- **Action:** $a_t \sim \pi_\theta(\cdot|s_t)$ — sampled from policy
- **Reward:** $r_t$ — task reward
- **Transition:** Environment step

### 3.2 Why RL?

A2C is the practical workhorse of deep RL: lower variance than REINFORCE (via the critic baseline), handles continuous actions (unlike DQN), and is straightforward to parallelize across multiple environments. GAE provides a principled, tunable bias-variance tradeoff.

---

## 4. Dataset

- **Name:** Custom environments with image observations
- **Size:** Generated online
- **Auto-download:**

```python
from torchvision import datasets
cifar = datasets.CIFAR10('./data', download=True)
```

---

## 5. Key Equations Summary

| Equation | Meaning |
|----------|---------|
| $\hat{A}_t = \sum_k(\gamma\lambda)^k\delta_{t+k}$ | GAE-$\lambda$ advantage estimate |
| $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ | TD error |
| $\nabla J = \mathbb{E}[\nabla\log\pi \cdot \hat{A}_t]$ | A2C policy gradient |
| $\lambda = 0$: low var, high bias | TD regime |
| $\lambda = 1$: high var, low bias | MC regime |

---

## 6. References

- Mnih, V. et al. "Asynchronous Methods for Deep Reinforcement Learning," *ICML*, 2016.
- Schulman, J. et al. "High-Dimensional Continuous Control Using Generalized Advantage Estimation," *ICLR*, 2016.
- Konda, V. R. & Tsitsiklis, J. N. "Actor-Critic Algorithms," *NeurIPS*, 2000.
- Sutton, R. S. & Barto, A. G. *Reinforcement Learning*, 2nd ed., MIT Press, 2018, Ch. 13.
