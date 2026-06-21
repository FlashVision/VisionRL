![Module Logo](../logo.png)

# Creative Image Agent via Reinforcement Learning

## Overview

This module develops an RL agent for creative image generation that balances aesthetic quality with novelty. The mathematical framework covers aesthetic score modeling through preference learning, curiosity-driven exploration for creative novelty, information-theoretic novelty measures, and multi-objective RL for navigating the quality-novelty Pareto frontier.

## Prerequisites

- Preference learning (regression, pairwise comparisons)
- Information theory (surprise, self-information, entropy)
- Multi-objective optimization (Pareto optimality, scalarization)
- Exploration theory (count-based, curiosity-driven)
- Reinforcement learning (intrinsic motivation, multi-objective RL)

---

## 1. Mathematical Foundations

### 1.1 Aesthetic Score Modeling

**Step 1 (Regression from ratings):** Given a dataset $\{(x_i, s_i)\}$ of images and aesthetic scores:

$$\hat{s}(x; \theta) = f_\theta(\text{CNN}(x))$$

Train via MSE: $\mathcal{L} = \frac{1}{N}\sum_i(s_i - \hat{s}(x_i))^2$.

**Step 2 (Pairwise preference learning):** More robust than absolute scores. Given preference pairs $(x_w, x_l)$:

$$P(x_w \succ x_l) = \sigma(\hat{s}(x_w) - \hat{s}(x_l))$$

$$\mathcal{L}_{pref} = -\sum\log\sigma(\hat{s}(x_w) - \hat{s}(x_l))$$

**Step 3 (Features for aesthetics):** The aesthetic model typically captures:
- Composition rules (rule of thirds, golden ratio): $R_{comp}(x)$
- Color harmony: $R_{color}(x)$
- Technical quality (sharpness, exposure): $R_{tech}(x)$
- Emotional impact: $R_{emotion}(x)$

**Step 4 (Combined aesthetic reward):**

$$R_{aesthetic}(x) = w_1 R_{comp}(x) + w_2 R_{color}(x) + w_3 R_{tech}(x) + w_4 R_{emotion}(x)$$

**Step 5 (Uncertainty estimation):** Use ensemble of $M$ aesthetic models:

$$\hat{s}(x) = \frac{1}{M}\sum_{m=1}^M f_{\theta_m}(x), \quad \sigma^2(x) = \frac{1}{M}\sum_m (f_{\theta_m}(x) - \hat{s}(x))^2$$

### 1.2 Novelty Search: Curiosity-Driven Creativity

**Step 1 (Behavioral characterization):** Define a behavior descriptor $\mathbf{b}(x) \in \mathbb{R}^d$ that maps images to a behavioral space (e.g., style features, color palette, composition vector).

**Step 2 (Novelty score):** The novelty of an image with respect to an archive $\mathcal{A}$:

$$N(x) = \frac{1}{k}\sum_{i=1}^k \|\mathbf{b}(x) - \mathbf{b}(x_i^{nn})\|$$

where $x_1^{nn}, \ldots, x_k^{nn}$ are the $k$ nearest neighbors in the archive.

**Step 3 (Information-theoretic novelty):**

$$N_{info}(x) = -\log p_{visited}(x) = -\log\frac{|\{x' \in \mathcal{A} : \|\mathbf{b}(x) - \mathbf{b}(x')\| < \epsilon\}| + 1}{|\mathcal{A}| + 1}$$

This is the self-information (surprisal) — rare behaviors have high novelty.

**Step 4 (Kernel density novelty):** Estimate the density of visited behaviors:

$$\hat{p}(x) = \frac{1}{|\mathcal{A}|}\sum_{x' \in \mathcal{A}} K_h(\mathbf{b}(x) - \mathbf{b}(x'))$$

where $K_h$ is a Gaussian kernel with bandwidth $h$. Then $N(x) = -\log\hat{p}(x)$.

**Step 5 (Connection to maximum entropy):** Maximizing $E[N_{info}(x)]$ is equivalent to maximizing entropy of the behavioral distribution:

$$\max_\pi H(\mathbf{b}) = \max_\pi E_\pi[-\log p(\mathbf{b})]$$

This encourages the agent to cover the space of possible creative outputs uniformly.

### 1.3 Multi-Objective RL: Quality vs. Novelty

**Step 1 (Pareto optimality):** A solution $\pi^*$ is Pareto-optimal if no other policy dominates it in all objectives:

$$\nexists \pi': J_1(\pi') \geq J_1(\pi^*) \text{ and } J_2(\pi') \geq J_2(\pi^*) \text{ with at least one strict}$$

where $J_1 = E[R_{quality}]$ and $J_2 = E[R_{novelty}]$.

**Step 2 (Linear scalarization):**

$$J_\lambda(\pi) = \lambda J_1(\pi) + (1-\lambda)J_2(\pi), \quad \lambda \in [0, 1]$$

Limitation: only finds solutions on the convex hull of the Pareto front.

**Step 3 (Chebyshev scalarization):** Finds all Pareto-optimal points:

$$\min_\pi \max_i \lambda_i(J_i^* - J_i(\pi))$$

where $J_i^*$ is the ideal point (best achievable for objective $i$).

**Step 4 (Envelope Q-learning for multi-objective):** Maintain a set of Q-functions parameterized by preference $\omega$:

$$Q^\omega(s, a) = E\left[\sum_t \gamma^t(\omega \cdot \mathbf{r}_t) | s_0=s, a_0=a\right]$$

where $\mathbf{r}_t = (r_{quality,t}, r_{novelty,t})$ is the vector reward.

**Step 5 (Pareto front approximation):** Solve for multiple $\lambda$ values and collect the envelope:

$$\mathcal{F}_{Pareto} = \{(J_1(\pi^*_\lambda), J_2(\pi^*_\lambda)) : \lambda \in [0,1]\}$$

### 1.4 Proof: Novelty Maximization Prevents Mode Collapse

**Theorem:** An agent maximizing $R = R_{quality} + \eta N_{info}$ with $\eta > 0$ cannot converge to a single output mode.

**Proof:** Assume by contradiction that $\pi^*$ collapses to a single output $x_0$, producing it with probability 1. Then:

$$p_{visited}(x_0) \to 1, \quad N_{info}(x_0) = -\log(1) = 0$$

For any alternative policy $\pi'$ producing diverse outputs:

$$E_{\pi'}[N_{info}] = E[-\log p_{visited}(x)] > 0$$

Therefore $J(\pi') > J(\pi^*)$ whenever $R_{quality}(\pi') + \eta E_{\pi'}[N] > R_{quality}(\pi^*) + 0$. Since diverse policies can maintain quality while increasing novelty (the creative space is rich), the collapsed solution is suboptimal. $\blacksquare$

---

## 2. Algorithm / Method

### 2.1 Pseudocode

```
Algorithm: Creative Image Agent with Multi-Objective RL
────────────────────────────────────────────────────────
Input: Generation model G, aesthetic model A, archive 𝒜
Output: Creative images on Pareto front

State: s = (prompt, style_parameters, generation_history)
Action: a = (style_vector, composition_params, technique_selection)
Reward: r = (r_quality, r_novelty) ∈ ℝ²

Initialize policy π_θ, archive 𝒜 ← ∅
Sample preference weight λ ~ Uniform[0,1] each episode

for episode = 1 to M do
    λ ~ Uniform(0, 1)   // Sample trade-off
    for t = 0 to T do
        aₜ ~ π_θ(·|sₜ, λ)
        xₜ ← G(sₜ, aₜ)
        r_quality ← A(xₜ)            // Aesthetic score
        r_novelty ← -log p_𝒜(xₜ)    // Information novelty
        rₜ ← λ·r_quality + (1-λ)·r_novelty
    end for
    Update 𝒜 ← 𝒜 ∪ {best outputs}
    Update π_θ via PPO with reward rₜ
end for
```

### 2.2 Complexity Analysis

- **Image generation:** $O(HW \cdot C^2 K^2 L)$
- **Aesthetic evaluation:** $O(HW \cdot C_{A}^2)$
- **KNN novelty computation:** $O(|\mathcal{A}| \cdot d)$ brute force, $O(\log|\mathcal{A}| \cdot d)$ with KD-tree
- **Kernel density estimation:** $O(|\mathcal{A}| \cdot d)$
- **Multi-objective optimization:** Same as single-objective per $\lambda$

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, \mathbf{R}, \gamma)$$

- **State:** Current creative context (prompt, style history, generation count)
- **Action:** Creative parameters (style weights, color palette, composition decisions)
- **Vector Reward:** $\mathbf{r} = (r_{quality}, r_{novelty})$
- **Preference-conditioned policy:** $\pi_\theta(a|s, \lambda)$ adapts to desired quality-novelty trade-off

### 3.2 Why RL?

1. **Non-differentiable aesthetics:** Human perception of beauty is hard to differentiate through
2. **Exploration for creativity:** RL's exploration mechanisms naturally drive novel output generation
3. **Multi-objective balance:** RL handles the quality-novelty Pareto trade-off naturally
4. **Open-ended generation:** No fixed target — the agent discovers new creative directions

---

## 4. Dataset

| Dataset | Size | Type | Description |
|---------|------|------|-------------|
| AVA | 250K | Aesthetic ratings | DPChallenge photos |
| AADB | 10K | Aesthetic attributes | Multi-attribute scores |
| WikiArt | 80K | Artistic styles | Paintings by period/style |
| BAM! | 2.5M | Creative media | Diverse artistic content |

---

## 5. Key Equations Summary

| Equation | Description |
|----------|-------------|
| $N_{info}(x) = -\log p_{visited}(x)$ | Information-theoretic novelty |
| $N(x) = \frac{1}{k}\sum_{i=1}^k\|\mathbf{b}(x)-\mathbf{b}(x_i^{nn})\|$ | KNN novelty |
| $J_\lambda = \lambda J_{quality} + (1-\lambda)J_{novelty}$ | Linear scalarization |
| $\max_\pi H(\mathbf{b}) = \max_\pi E[-\log p(\mathbf{b})]$ | Maximum entropy exploration |
| $P(x_w\succ x_l) = \sigma(\hat{s}(x_w)-\hat{s}(x_l))$ | Aesthetic preference model |

---

## 6. References

1. Lehman, J., & Stanley, K. O. (2011). Abandoning objectives: Evolution through the search for novelty alone. *Evolutionary Computation*, 19(2), 189-223.
2. Murray, N., Marchesotti, L., & Perronnin, F. (2012). AVA: A large-scale database for aesthetic visual analysis. *CVPR*.
3. Pathak, D., et al. (2017). Curiosity-driven exploration by self-predictive forward dynamics. *ICML*.
4. Yang, R., Sun, X., & Narasimhan, K. (2019). A generalized algorithm for multi-objective reinforcement learning and policy adaptation. *NeurIPS*.
