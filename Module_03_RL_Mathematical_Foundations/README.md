![Module Logo](logo.png)

# Module 03: Reinforcement Learning — Mathematical Foundations

## Overview

This module builds the complete mathematical machinery required for reinforcement learning from first principles. We start with probability theory and systematically construct the framework of Markov Decision Processes, Bellman equations, value functions, and optimality conditions. Every concept is derived rigorously with full proofs, ensuring you understand *why* each result holds — not just *what* it states.

By the end of this module, you will have the theoretical foundation to understand any RL algorithm at a deep level, and you will see how these abstract mathematical structures apply concretely to image processing and computer vision tasks.

## Prerequisites

| Area | Topics You Should Know |
|------|----------------------|
| **Calculus** | Derivatives, chain rule, integration, Taylor series, gradients |
| **Linear Algebra** | Matrix multiplication, eigenvalues, norms, positive definite matrices |
| **Probability** | Basic probability rules, distributions (Gaussian, Bernoulli), expectation |
| **Programming** | Python, NumPy basics, familiarity with PyTorch or TensorFlow |

## Module Structure

| Sub-module | Topic | Key Mathematical Concepts |
|-----------|-------|--------------------------|
| [01](./01_Probability_For_RL/) | Probability for RL | Kolmogorov axioms, σ-algebras, conditional expectation, Markov property |
| [02](./02_Markov_Decision_Process/) | Markov Decision Processes | MDP tuple \((S, A, P, R, \gamma)\), transition kernels, policy definition |
| [03](./03_Bellman_Equations/) | Bellman Equations | Bellman expectation/optimality, contraction mappings, fixed-point theorems |
| [04](./04_Value_Functions/) | Value Functions | State-value \(V^\pi\), action-value \(Q^\pi\), advantage functions, TD error |
| [05](./05_Policy_And_Optimality/) | Policy & Optimality | Policy ordering, improvement theorem, policy/value iteration convergence |

## Mathematical Notation Used Throughout

| Symbol | Meaning |
|--------|---------|
| \(S\) | State space |
| \(A\) | Action space |
| \(P(s' \mid s, a)\) | Transition probability |
| \(R(s, a)\) | Reward function |
| \(\gamma \in [0, 1)\) | Discount factor |
| \(\pi(a \mid s)\) | Stochastic policy |
| \(V^\pi(s)\) | State-value function under policy \(\pi\) |
| \(Q^\pi(s, a)\) | Action-value function under policy \(\pi\) |
| \(G_t\) | Discounted return from time \(t\) |
| \(\mathbb{E}_\pi[\cdot]\) | Expectation under policy \(\pi\) |

## Connection to Vision & Image Processing

Throughout this module, every abstract concept is grounded with image processing examples:

- **States** are images or image feature vectors
- **Actions** are image transformations (filters, crops, adjustments)
- **Rewards** measure image quality (PSNR, SSIM, perceptual metrics)
- **Policies** select which processing operation to apply next

This framing prepares you for Modules 05–10 where these ideas power real vision systems.

## How to Use This Module

1. **Read each sub-module's README** for the full mathematical treatment
2. **Work through the Jupyter notebooks** for hands-on implementation
3. **Prove the exercises** — mathematical maturity comes from doing, not reading
4. **Connect to vision** — after each concept, think about how it maps to image processing

## Next Steps

After completing this module, proceed to [Module 04: Basic RL Algorithms](../Module_04_Basic_RL_Algorithms/) where these mathematical foundations are turned into working algorithms.
