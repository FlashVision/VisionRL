![Module Logo](logo.png)

# Module 06: Deep Reinforcement Learning

## Overview

This module combines the neural network foundations from Module 05 with the RL theory from Modules 03-04 to build **Deep RL agents that learn from images**. We progress from DQN (the algorithm that first demonstrated superhuman Atari play) through policy gradients, actor-critic methods, PPO, and essential training tricks — all applied to image processing environments where the agent learns to enhance, filter, and transform images.

**The key insight:** Classical RL (Module 04) used tabular methods — a lookup table for every state-action pair. With images as states ($\sim 10^{7700}$ possible pixel combinations for a 32×32 RGB image), tables are impossible. Deep neural networks (Module 05) serve as **function approximators** that generalize across similar states.

---

## Module Structure

| # | Topic | Key Math | Folder |
|---|-------|----------|--------|
| 6.1 | DQN | Q-network loss, experience replay, target networks | `01_DQN_Deep_Q_Network/` |
| 6.2 | Policy Gradient (REINFORCE) | Policy gradient theorem, score function, baselines | `02_Policy_Gradient_REINFORCE/` |
| 6.3 | Actor-Critic (A2C) | Advantage estimation, GAE(λ), dual networks | `03_Actor_Critic_A2C/` |
| 6.4 | PPO | TRPO theory, clipped objective, KL penalty | `04_PPO_Algorithm/` |
| 6.5 | Experience Replay & Tricks | PER, Double DQN, Dueling, N-step returns | `05_Experience_Replay_And_Tricks/` |

---

## The Deep RL Architecture

```
Image State s_t                     Neural Network                    Action a_t
[32 × 32 × 3] ──CNN Encoder──▶ [Feature Vector] ──Policy Head──▶ [Enhancement Action]
                                        │
                                  ──Value Head──▶ [V(s) or Q(s,a)]
```

## Mathematical Framework

The core optimization problem in Deep RL:

**Value-based (DQN):** Find $\theta$ such that:
$$Q(s, a; \theta) \approx \mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^t r_t \mid s_0 = s, a_0 = a, \pi\right]$$

**Policy-based (REINFORCE, PPO):** Find $\theta$ that maximizes:
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T}\gamma^t r_t\right]$$

**Actor-Critic (A2C):** Jointly optimize:
$$\pi_\theta(a|s) \quad \text{(actor)} \quad \text{and} \quad V_\phi(s) \quad \text{(critic)}$$

---

## Image Processing Environment

All notebooks in this module use a custom **ImageEnhancementEnv** built on CIFAR-10:

```
State:      Degraded image (noise, blur, color shift added to CIFAR-10 image)
Actions:    Enhancement operations (sharpen, denoise, adjust contrast, etc.)
Reward:     Image quality metric (PSNR, SSIM) relative to clean original
Terminal:   After K steps or when quality threshold reached
```

This creates a natural testbed where RL concepts directly solve an image processing problem.

---

## Prerequisites

- **Module 05**: Neural networks, CNNs, backpropagation, BatchNorm
- **Module 03-04**: MDPs, Bellman equations, value functions, Q-learning
- **PyTorch**: Tensor operations, autograd, nn.Module
- **Math**: Probability, expected values, gradients

## Datasets Used

All datasets **auto-download** — no manual setup:

- **CIFAR-10** (via `torchvision.datasets.CIFAR10`) — used as source images for the ImageEnhancementEnv
- **MNIST** (via `torchvision.datasets.MNIST`) — used for filter selection environments

## What You'll Build

By the end of this module, you will have implemented:
1. A DQN agent that learns to enhance degraded images
2. A REINFORCE agent that selects optimal image filters
3. An A2C agent with advantage estimation for continuous-like image enhancement
4. A PPO agent with clipped objectives for stable multi-step enhancement
5. Advanced techniques (PER, Double DQN, Dueling) with ablation studies

## Connection to Later Modules

| Module 06 Algorithm | Applied In |
|---|---|
| DQN | Module 07 (Enhancement), Module 09 (Detection) |
| REINFORCE | Module 08 (Segmentation) |
| A2C | Module 10 (Generation) |
| PPO | Module 07-12 (preferred algorithm for most applications) |
| Advanced tricks | Module 11 (Advanced topics) |
