![Module Logo](../logo.png)

# Attention-Based Visual Search via Reinforcement Learning

## Overview

This module develops an RL agent that performs object detection through sequential attention, mimicking biological foveal vision by selectively attending to informative image regions. The mathematical framework derives the scaled dot-product attention mechanism, proves the REINFORCE gradient estimator for hard attention, establishes information-theoretic justification for foveal vision, and formulates sequential attention as a Partially Observable Markov Decision Process (POMDP).

## Prerequisites

- Linear algebra (projections, bilinear forms, low-rank approximations)
- Probability theory (score function estimator, variance reduction)
- Information theory (entropy, mutual information, channel capacity)
- Reinforcement learning (REINFORCE, baselines, POMDPs)
- Computational neuroscience (saccadic eye movements, foveal vision)

---

## 1. Mathematical Foundations

### 1.1 Attention Mechanism Derivation

**Scaled Dot-Product Attention:**

Given queries $Q \in \mathbb{R}^{n \times d_k}$, keys $K \in \mathbb{R}^{m \times d_k}$, and values $V \in \mathbb{R}^{m \times d_v}$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Step 1 (Why dot product?):** The dot product $q^Tk$ measures alignment between query and key. For orthonormal bases, $q^Tk = \|q\|\|k\|\cos\theta$, naturally encoding similarity.

**Step 2 (Why scale by $\sqrt{d_k}$?):** Assume $q_i, k_i \sim \mathcal{N}(0, 1)$ independently. Then:

$$E[q^Tk] = \sum_{i=1}^{d_k} E[q_i k_i] = 0$$

$$\text{Var}[q^Tk] = \sum_{i=1}^{d_k} \text{Var}[q_i k_i] = \sum_{i=1}^{d_k} E[q_i^2]E[k_i^2] = d_k$$

**Step 3:** Without scaling, as $d_k$ grows, $q^Tk$ has standard deviation $\sqrt{d_k}$, pushing softmax into saturated regions where gradients vanish.

**Step 4:** Dividing by $\sqrt{d_k}$ normalizes the variance:

$$\text{Var}\left[\frac{q^Tk}{\sqrt{d_k}}\right] = \frac{d_k}{d_k} = 1$$

This keeps the softmax in a well-behaved gradient regime regardless of dimension.

**Step 5 (Multi-head attention):** Split into $h$ heads with dimension $d_k/h$ each:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Total computation: $O(n \cdot m \cdot d_k)$ per head, $O(n \cdot m \cdot d_k)$ total (same as single head with full dimension).

### 1.2 REINFORCE for Hard Attention: Full Derivation

**Problem:** Hard attention samples a discrete location $l_t \in \{1, \ldots, N\}$ at each step. The objective:

$$J(\theta) = E_{\tau \sim p_\theta(\tau)}\left[\sum_{t=0}^T r_t\right]$$

where $\tau = (l_0, l_1, \ldots, l_T)$ is a trajectory of attention locations.

**Step 1:** Write the expected reward:

$$J(\theta) = \sum_\tau p_\theta(\tau) R(\tau)$$

where $R(\tau) = \sum_t r_t$ is the total reward.

**Step 2:** Compute the gradient:

$$\nabla_\theta J = \sum_\tau \nabla_\theta p_\theta(\tau) \cdot R(\tau) = \sum_\tau p_\theta(\tau) \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} R(\tau)$$

**Step 3:** Apply the log-derivative trick: $\nabla_\theta \log p_\theta(\tau) = \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)}$:

$$\nabla_\theta J = E_{\tau \sim p_\theta}\left[\nabla_\theta \log p_\theta(\tau) \cdot R(\tau)\right]$$

**Step 4:** Factor the trajectory probability:

$$p_\theta(\tau) = \prod_{t=0}^T \pi_\theta(l_t | s_t)$$

$$\log p_\theta(\tau) = \sum_{t=0}^T \log \pi_\theta(l_t | s_t)$$

**Step 5:** The gradient becomes:

$$\nabla_\theta J = E\left[\sum_{t=0}^T \nabla_\theta \log\pi_\theta(l_t|s_t) \cdot \left(\sum_{t'=t}^T r_{t'}\right)\right]$$

(using causality: actions at time $t$ don't affect past rewards)

**Step 6 (Baseline for variance reduction):** Subtract a baseline $b(s_t)$:

$$\nabla_\theta J = E\left[\sum_t \nabla_\theta\log\pi_\theta(l_t|s_t)(R_t - b(s_t))\right]$$

**Step 7:** Prove the baseline doesn't introduce bias:

$$E[\nabla_\theta\log\pi_\theta(l_t|s_t) \cdot b(s_t)] = b(s_t) \cdot E_{l_t}[\nabla_\theta\log\pi_\theta(l_t|s_t)]$$
$$= b(s_t) \cdot \nabla_\theta\sum_{l_t}\pi_\theta(l_t|s_t) = b(s_t) \cdot \nabla_\theta 1 = 0 \quad \blacksquare$$

**Step 8:** Optimal baseline (minimizing variance):

$$b^*(s_t) = \frac{E[(\nabla_\theta\log\pi)^2 \cdot R_t]}{E[(\nabla_\theta\log\pi)^2]}$$

In practice, use a learned value function: $b(s_t) \approx V_\phi(s_t)$.

### 1.3 Foveal Vision: Information-Theoretic Justification

**Step 1:** Model the eye as an information channel with capacity constraint $C$ bits per fixation.

**Step 2:** The retina has resolution function $\rho(r) = \rho_0 e^{-r/r_0}$ where $r$ is eccentricity from the fovea.

**Step 3:** Information gathered per fixation at location $l$:

$$I(l) = \int_\Omega \rho(|x - l|) \cdot h(I(x)) \, dx$$

where $h(I(x))$ is the local entropy of the image at pixel $x$.

**Step 4:** The optimal fixation sequence maximizes total information:

$$\{l_t^*\} = \arg\max_{\{l_t\}} \sum_t I(l_t) \text{ subject to no redundancy}$$

**Step 5:** This is equivalent to maximizing mutual information between observations and target identity:

$$I^* = \arg\max I(Y; C | l_1, \ldots, l_T)$$

where $Y$ are observations and $C$ is the object class.

### 1.4 Sequential Attention as POMDP

**Definition:** The POMDP tuple $(\mathcal{S}, \mathcal{A}, \mathcal{O}, P, R, O, \gamma)$:

- **True state** $s$: Full image content + object locations
- **Observation** $o_t$: High-resolution glimpse at attended location $l_t$
- **Action** $a_t$: Next attention location $l_{t+1}$ OR classification decision
- **Observation function:** $O(o|s, l) = P(\text{glimpse} | \text{image}, \text{location})$

The belief state $b_t = P(s | o_1, \ldots, o_t, l_1, \ldots, l_t)$ is updated via Bayes' rule:

$$b_{t+1}(s) \propto O(o_{t+1}|s, l_{t+1}) \sum_{s'} P(s|s') b_t(s')$$

---

## 2. Algorithm / Method

### 2.1 Pseudocode

```
Algorithm: Recurrent Attention Model (RAM) with RL
────────────────────────────────────────────────────
Input: Image I, number of glimpses T
Output: Classification label or detection boxes

Initialize glimpse network g_θ, recurrent core h_φ, 
         location network l_ψ, classification network c_ω

for episode = 1 to M do
    l₀ ~ Uniform(image_grid)
    h₀ ← zeros
    for t = 0 to T-1 do
        gₜ ← g_θ(I, lₜ)             // Extract glimpse features
        hₜ ← h_φ(hₜ₋₁, gₜ)          // Update hidden state (RNN)
        lₜ₊₁ ~ π_ψ(·|hₜ)            // Sample next location (Gaussian)
        Store log π_ψ(lₜ₊₁|hₜ)
    end for
    ŷ ← c_ω(h_T)                     // Final classification
    R ← 𝟙[ŷ = y*]                    // Reward
    ∇θ ← Σₜ ∇log π_ψ(lₜ|hₜ₋₁)(R - b)  // REINFORCE
    Update all parameters
end for
```

### 2.2 Complexity Analysis

- **Glimpse extraction:** $O(g^2 \cdot C)$ for glimpse size $g \times g$
- **RNN update:** $O(d_h^2)$ for hidden dimension $d_h$
- **Location sampling:** $O(d_h)$
- **Total per image:** $O(T \cdot (g^2 C + d_h^2))$ — linear in glimpses, not image size
- **Advantage over dense attention:** $O(T \cdot g^2)$ vs $O(N^2)$ for full image

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

- **State:** RNN hidden state (belief about image content from past glimpses)
- **Action:** 2D attention location $(x, y) \in [-1, 1]^2$ (continuous)
- **Reward:** +1 for correct classification/detection, 0 otherwise (sparse)
- **Episode:** Fixed number of glimpses $T$

### 3.2 Why RL?

1. **Hard attention is non-differentiable:** Discrete selection of image regions requires REINFORCE or similar
2. **Computational efficiency:** Agent processes $O(T)$ patches rather than full image — proportional to task complexity, not image size
3. **Active perception:** Agent learns where to look based on what it has seen — true visual search behavior
4. **Sparse reward handling:** RL naturally handles delayed gratification (many glimpses before final classification)

---

## 4. Dataset

| Dataset | Size | Task | Description |
|---------|------|------|-------------|
| MNIST | 70K | Digit classification | Cluttered/translated versions |
| SVHN | 600K | Street numbers | Real-world digit recognition |
| CUB-200 | 11,788 | Fine-grained | Bird species (subtle differences) |
| ImageNet | 1.2M | Classification | Large-scale visual recognition |

---

## 5. Key Equations Summary

| Equation | Description |
|----------|-------------|
| $\text{Attn}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V$ | Scaled dot-product attention |
| $\nabla_\theta J = E[\nabla\log\pi_\theta(\tau)(R-b)]$ | REINFORCE gradient |
| $\text{Var}[q^Tk] = d_k$ (unscaled) | Motivation for scaling |
| $b^* = E[(\nabla\log\pi)^2 R]/E[(\nabla\log\pi)^2]$ | Optimal baseline |
| $b_{t+1}(s) \propto O(o_{t+1}|s,l_{t+1})\sum_{s'}P(s|s')b_t(s')$ | POMDP belief update |

---

## 6. References

1. Mnih, V., Heess, N., Graves, A., & Kavukcuoglu, K. (2014). Recurrent models of visual attention. *NeurIPS*.
2. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
3. Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3), 229-256.
4. Ba, J., Mnih, V., & Kavukcuoglu, K. (2015). Multiple object recognition with visual attention. *ICLR*.
