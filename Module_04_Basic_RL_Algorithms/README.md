![Module Logo](logo.png)

# Module 04: Basic RL Algorithms

## Overview

This module bridges the gap between mathematical theory (Module 03) and practical implementation. We develop the foundational RL algorithms — from multi-armed bandits to Q-learning — with full mathematical derivations of their convergence properties. All methods here are **tabular**: they maintain explicit tables for value functions, making them ideal for understanding the core ideas before scaling to neural network function approximation in later modules.

Each algorithm is derived from the mathematical principles established in Module 03, and every convergence proof is presented step by step.

## Prerequisites

| Requirement | Where Covered |
|------------|---------------|
| Bellman equations and their derivations | [Module 03.03](../Module_03_RL_Mathematical_Foundations/03_Bellman_Equations/) |
| Value functions (\(V^\pi\), \(Q^\pi\), \(A^\pi\)) | [Module 03.04](../Module_03_RL_Mathematical_Foundations/04_Value_Functions/) |
| Policy improvement theorem | [Module 03.05](../Module_03_RL_Mathematical_Foundations/05_Policy_And_Optimality/) |
| Contraction mappings and fixed-point theory | [Module 03.03](../Module_03_RL_Mathematical_Foundations/03_Bellman_Equations/) |
| Python, NumPy, basic plotting | General |

## Module Structure

| Sub-module | Topic | Key Algorithm | Learning Type |
|-----------|-------|--------------|---------------|
| [01](./01_Multi_Armed_Bandits/) | Multi-Armed Bandits | UCB, Thompson Sampling | No states, immediate reward |
| [02](./02_Dynamic_Programming/) | Dynamic Programming | Policy Iteration, Value Iteration | Model-based, full knowledge |
| [03](./03_Monte_Carlo_Methods/) | Monte Carlo Methods | First-visit MC, Off-policy MC | Model-free, episodic |
| [04](./04_TD_Learning_SARSA/) | TD Learning & SARSA | TD(0), SARSA, TD(λ) | Model-free, bootstrapping, on-policy |
| [05](./05_Q_Learning/) | Q-Learning | Q-learning, Double Q-learning | Model-free, bootstrapping, off-policy |

## Algorithm Taxonomy

```
RL Algorithms (Tabular)
├── No State (Bandits)
│   ├── ε-greedy
│   ├── UCB
│   └── Thompson Sampling
├── Model-Based (DP)
│   ├── Policy Evaluation
│   ├── Policy Iteration
│   └── Value Iteration
└── Model-Free
    ├── Monte Carlo
    │   ├── First-visit MC
    │   ├── Every-visit MC
    │   └── Off-policy MC (importance sampling)
    └── Temporal Difference
        ├── On-policy: SARSA, TD(λ)
        └── Off-policy: Q-learning, Double Q-learning
```

## Key Convergence Results

| Algorithm | Converges To | Conditions | Proved In |
|-----------|-------------|------------|-----------|
| UCB | \(O(\sqrt{T \log T})\) regret | — | Sub-module 01 |
| Policy Iteration | \(\pi^*\) | Finite MDP | Sub-module 02 |
| Value Iteration | \(V^*\) | \(\gamma < 1\) | Sub-module 02 |
| First-visit MC | \(V^\pi\) | All states visited ∞ | Sub-module 03 |
| SARSA | \(Q^\pi\) | Robbins-Monro + GLIE | Sub-module 04 |
| Q-learning | \(Q^*\) | Robbins-Monro | Sub-module 05 |

## Connection to Image Processing

Throughout this module, all algorithms are demonstrated on image processing tasks:
- **Bandits** → selecting the best image filter from a set
- **DP** → optimal image processing pipelines when dynamics are known
- **MC** → evaluating image enhancement sequences by running complete episodes
- **TD/SARSA** → online learning of image processing strategies
- **Q-learning** → learning optimal filter sequences without following the current best strategy

## Next Steps

After mastering tabular methods, proceed to [Module 05](../Module_05_Deep_Q_Networks/) where we replace tables with neural networks to handle the enormous state spaces of real images.
