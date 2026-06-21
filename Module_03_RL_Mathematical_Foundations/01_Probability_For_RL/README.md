![Module Logo](../logo.png)

# Probability for Reinforcement Learning

## Overview

Probability theory is the mathematical language of reinforcement learning. Every RL concept — states, transitions, rewards, policies — is defined through probability distributions. This document develops the Kolmogorov axioms, provides complete proofs of Bayes' theorem and the tower property (law of iterated expectations), gives the formal definition of the Markov property, and proves that image sequences under common assumptions satisfy it.

## Prerequisites

- Set theory (sigma-algebras, measurable spaces)
- Calculus (integration, series)
- Basic combinatorics

---

## 1. Mathematical Foundations

### 1.1 Core Definition — Kolmogorov Probability Axioms

A probability space is a triple $(\Omega, \mathcal{F}, P)$ where:

- $\Omega$ is the **sample space** (set of all outcomes)
- $\mathcal{F} \subseteq 2^\Omega$ is a **sigma-algebra** (collection of events, closed under complement and countable union)
- $P: \mathcal{F} \to [0, 1]$ is a **probability measure** satisfying:

**Axiom 1 (Non-negativity):** $P(A) \geq 0$ for all $A \in \mathcal{F}$.

**Axiom 2 (Normalization):** $P(\Omega) = 1$.

**Axiom 3 (Countable additivity):** For any countable collection of pairwise disjoint events $A_1, A_2, \ldots$:

$$P\!\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)$$

**Immediate consequences:**

- $P(\emptyset) = 0$ (take $A_1 = \emptyset, A_2 = \emptyset, \ldots$)
- $P(A^c) = 1 - P(A)$ (since $A \cup A^c = \Omega$ and $A \cap A^c = \emptyset$)
- $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ (inclusion-exclusion)

### 1.2 Full Proof of Bayes' Theorem

**Theorem (Bayes):** For events $A, B$ with $P(B) > 0$:

$$P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B)}$$

**Proof:**

**Step 1:** Define conditional probability. The conditional probability of $A$ given $B$ is:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

**Step 2:** By the same definition applied to $P(B \mid A)$:

$$P(B \mid A) = \frac{P(A \cap B)}{P(A)}, \quad P(A) > 0$$

**Step 3:** From Step 2, express the joint probability:

$$P(A \cap B) = P(B \mid A) \cdot P(A)$$

**Step 4:** Substitute into Step 1:

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

**Result:**

$$\boxed{P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B)}}$$
$\blacksquare$

**Generalization (Law of Total Probability):** If $\{A_1, \ldots, A_n\}$ is a partition of $\Omega$ (mutually exclusive, exhaustive), then:

$$P(B) = \sum_{i=1}^{n} P(B \mid A_i) P(A_i)$$

Substituting into Bayes:

$$P(A_k \mid B) = \frac{P(B \mid A_k) P(A_k)}{\sum_{i=1}^{n} P(B \mid A_i) P(A_i)}$$

**Intuition:** Bayes' theorem inverts conditioning. In RL: given an observed outcome (reward), Bayes' theorem tells us how to update our belief about the environment's state. This is the foundation of Bayesian RL, model estimation, and belief-state MDPs.

### 1.3 Full Proof of the Tower Property (Law of Iterated Expectations)

**Theorem:** For random variables $X$ and $Y$:

$$\mathbb{E}[\mathbb{E}[X \mid Y]] = \mathbb{E}[X]$$

More generally, if $\mathcal{G} \subseteq \mathcal{H}$ are sub-sigma-algebras:

$$\mathbb{E}[\mathbb{E}[X \mid \mathcal{H}] \mid \mathcal{G}] = \mathbb{E}[X \mid \mathcal{G}]$$

**Proof (discrete case):**

**Step 1:** Write the outer expectation by summing over values of $Y$:

$$\mathbb{E}[\mathbb{E}[X \mid Y]] = \sum_y \mathbb{E}[X \mid Y = y] \cdot P(Y = y)$$

**Step 2:** Expand the inner expectation:

$$= \sum_y \left[\sum_x x \cdot P(X = x \mid Y = y)\right] P(Y = y)$$

**Step 3:** Use the definition of conditional probability: $P(X = x \mid Y = y) \cdot P(Y = y) = P(X = x, Y = y)$:

$$= \sum_y \sum_x x \cdot P(X = x, Y = y)$$

**Step 4:** Interchange the order of summation:

$$= \sum_x x \sum_y P(X = x, Y = y) = \sum_x x \cdot P(X = x) = \mathbb{E}[X]$$

**Result:**

$$\boxed{\mathbb{E}[\mathbb{E}[X \mid Y]] = \mathbb{E}[X]}$$
$\blacksquare$

**Intuition:** Averaging over subsets, then averaging the subset averages, gives the overall average. In RL: the expected value function $\mathbb{E}_\pi[V(s')]$ can be computed by first conditioning on the next state $s'$ (inner expectation), then averaging over all possible next states (outer expectation). This is the essence of the Bellman equation.

### 1.4 Formal Definition of the Markov Property

**Definition:** A stochastic process $\{X_t\}_{t \geq 0}$ has the **Markov property** if:

$$P(X_{t+1} = x \mid X_0, X_1, \ldots, X_t) = P(X_{t+1} = x \mid X_t) \quad \forall t, \forall x$$

Equivalently, conditioned on the present, the future is independent of the past:

$$X_{t+1} \perp\!\!\!\perp (X_0, \ldots, X_{t-1}) \mid X_t$$

**Time-homogeneous Markov chain:** If $P(X_{t+1} = j \mid X_t = i) = p_{ij}$ does not depend on $t$, the chain is time-homogeneous with transition matrix $\mathbf{P} = [p_{ij}]$.

### 1.5 Proof That Image Sequences Satisfy the Markov Property (Under Standard Assumptions)

**Theorem:** Under the assumption that the environment dynamics depend only on the current image state and action, the image-based RL process $\{(I_t, a_t, r_t)\}$ is Markov.

**Proof:**

**Step 1:** Define the state as the full image: $s_t = I_t \in \mathbb{R}^{M \times N \times C}$.

**Step 2:** Assume the environment transition model is:

$$P(s_{t+1} \mid s_0, a_0, s_1, a_1, \ldots, s_t, a_t) = P(s_{t+1} \mid s_t, a_t) = T(s_{t+1} \mid s_t, a_t)$$

This holds when:
- The image $I_t$ encodes all information about the environment's current configuration.
- No hidden variables exist that affect future states beyond what's captured in $I_t$.

**Step 3:** Similarly, the reward depends only on the current transition:

$$P(r_t \mid s_0, a_0, \ldots, s_t, a_t) = P(r_t \mid s_t, a_t)$$

**Step 4:** Therefore the tuple $(s_t, a_t, r_t, s_{t+1})$ is fully characterized by the current state-action pair, satisfying the Markov property.

**Step 5 — When the Markov property fails:** If the image observation is partial (e.g., a cropped or occluded view), the Markov property is violated. Solutions:
- **Frame stacking:** Use $s_t = (I_{t-k}, \ldots, I_t)$ to capture temporal context.
- **Recurrent networks:** Use LSTM/GRU to maintain hidden state.
- **Belief states:** Maintain a probability distribution over possible true states (POMDP framework).
$\blacksquare$

---

## 2. Algorithm / Method

### 2.1 Pseudocode: Bayesian State Estimation

```
Algorithm: Bayesian_State_Update
Input: Prior P(s), observation o, observation model P(o|s)
Output: Posterior P(s|o)

1. For each state s:
     likelihood = P(o | s)
     unnormalized_posterior[s] = likelihood × P(s)
2. Z = sum(unnormalized_posterior)
3. For each state s:
     P(s | o) = unnormalized_posterior[s] / Z
4. Return P(s | o)
```

### 2.2 Complexity Analysis

- **Bayesian update (finite state space):** $O(|\mathcal{S}|)$ per observation
- **Particle filter (continuous):** $O(N_{\text{particles}})$ per step
- **Space:** $O(|\mathcal{S}|)$ for the belief distribution

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

- **State:** $s_t \in \mathcal{S}$ — environment state (possibly an image)
- **Action:** $a_t \in \mathcal{A}$ — agent's chosen action
- **Reward:** $r_t \sim R(s_t, a_t)$ — stochastic reward signal
- **Transition:** $s_{t+1} \sim T(\cdot \mid s_t, a_t)$ — Markov transition kernel

### 3.2 Why RL?

Every RL algorithm rests on the probability axioms: value functions are expectations, policies are conditional distributions, and the Bellman equation is an application of the tower property. Understanding these foundations ensures correct implementation and debugging of RL systems.

---

## 4. Dataset

- **Name:** Synthetic Markov chains and image sequences
- **Size:** Generated programmatically
- **Auto-download:**

```python
import numpy as np
P = np.array([[0.7, 0.3], [0.4, 0.6]])
states = [0]
for _ in range(1000):
    states.append(np.random.choice([0,1], p=P[states[-1]]))
```

---

## 5. Key Equations Summary

| Equation | Meaning |
|----------|---------|
| $P(A \mid B) = P(B \mid A)P(A)/P(B)$ | Bayes' theorem |
| $\mathbb{E}[\mathbb{E}[X \mid Y]] = \mathbb{E}[X]$ | Tower property |
| $P(X_{t+1} \mid X_0,\ldots,X_t) = P(X_{t+1} \mid X_t)$ | Markov property |
| $P(B) = \sum_i P(B \mid A_i)P(A_i)$ | Law of total probability |
| $P(\Omega) = 1, \; P(A) \geq 0$ | Kolmogorov axioms |

---

## 6. References

- Kolmogorov, A. N. *Foundations of the Theory of Probability*, Chelsea Publishing, 1950.
- Grimmett, G. R. & Stirzaker, D. R. *Probability and Random Processes*, 3rd ed., Oxford, 2001.
- Bertsekas, D. P. & Tsitsiklis, J. N. *Introduction to Probability*, 2nd ed., Athena Scientific, 2008.
- Sutton, R. S. & Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018.
