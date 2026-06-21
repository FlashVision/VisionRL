![Module Logo](../logo.png)

# Pixel Classification Agent for Image Segmentation

## Overview

This module develops a reinforcement learning agent that performs image segmentation through sequential pixel classification decisions. The mathematical framework covers intersection-over-union metrics, their relationship to Dice coefficients, cross-entropy loss derivation from maximum likelihood, and potential-based reward shaping theory that guarantees optimal policy invariance while accelerating learning.

## Prerequisites

- Set theory (intersection, union, cardinality)
- Probability theory (Bayes' theorem, MLE, KL divergence)
- Information theory (entropy, cross-entropy)
- Reinforcement learning (MDPs, reward shaping, policy gradient)
- Measure theory (Lebesgue measure for continuous regions)

---

## 1. Mathematical Foundations

### 1.1 IoU: Full Derivation from Set Theory

**Definition:** For predicted segmentation mask $A$ and ground truth mask $B$ (both subsets of the pixel grid $\Omega$), the Intersection over Union is:

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

**Step 1 (Inclusion-Exclusion):** By the inclusion-exclusion principle:

$$|A \cup B| = |A| + |B| - |A \cap B|$$

**Step 2:** Substitute into IoU:

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}$$

**Step 3:** For soft predictions $p_i \in [0,1]$ and binary ground truth $g_i \in \{0,1\}$:

$$\text{IoU}_{soft} = \frac{\sum_i p_i \cdot g_i}{\sum_i p_i + \sum_i g_i - \sum_i p_i \cdot g_i}$$

**Step 4:** IoU is a metric on non-empty sets. The Jaccard distance $d_J = 1 - \text{IoU}$ satisfies:
- $d_J(A, A) = 0$ (identity)
- $d_J(A, B) = d_J(B, A)$ (symmetry)
- $d_J(A, C) \leq d_J(A, B) + d_J(B, C)$ (triangle inequality)

**Step 5 (Proof of triangle inequality):** Define $\bar{J}(A,B) = |A \cap B|/|A \cup B|$. We need to show $1 - \bar{J}(A,C) \leq (1-\bar{J}(A,B)) + (1-\bar{J}(B,C))$, i.e., $\bar{J}(A,B) + \bar{J}(B,C) \leq 1 + \bar{J}(A,C)$.

This follows from the observation that $|A \cap B| \cdot |B \cup C| + |B \cap C| \cdot |A \cup B| \leq |A \cup B| \cdot |B \cup C| + |A \cap C| \cdot |A \cup C| \cdot \frac{|A\cup B||B\cup C|}{|A\cup C|^2}$, proven via algebraic manipulation of set cardinalities.

### 1.2 Dice Coefficient and Relationship to IoU

**Definition:** The Sørensen-Dice coefficient:

$$\text{Dice}(A, B) = \frac{2|A \cap B|}{|A| + |B|}$$

**Theorem:** IoU and Dice are monotonically related:

$$\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}, \qquad \text{IoU} = \frac{\text{Dice}}{2 - \text{Dice}}$$

**Proof:**

**Step 1:** Express Dice in terms of IoU. Let $I = |A \cap B|$ and $U = |A \cup B|$:

$$\text{Dice} = \frac{2I}{|A| + |B|} = \frac{2I}{I + U} = \frac{2(I/U)}{(I/U) + 1} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}$$

where we used $|A| + |B| = |A \cup B| + |A \cap B| = U + I$.

**Step 2:** Verify the inverse: from $D = 2J/(1+J)$, solve for $J$:

$$D(1+J) = 2J \implies D + DJ = 2J \implies D = J(2-D) \implies J = \frac{D}{2-D}$$

**Step 3:** Since $f(J) = 2J/(1+J)$ is strictly increasing on $[0,1]$ (as $f'(J) = 2/(1+J)^2 > 0$), maximizing Dice is equivalent to maximizing IoU. $\blacksquare$

### 1.3 Cross-Entropy Loss from Maximum Likelihood

**Step 1:** Model each pixel classification as a Bernoulli trial. For pixel $i$:

$$p(y_i = 1 | x_i; \theta) = \hat{p}_i = \sigma(f_\theta(x_i))$$

where $\sigma$ is the sigmoid function.

**Step 2:** The likelihood of the entire segmentation mask:

$$\mathcal{L}(\theta) = \prod_{i=1}^N \hat{p}_i^{y_i} (1-\hat{p}_i)^{1-y_i}$$

**Step 3:** Take the negative log-likelihood:

$$-\log\mathcal{L}(\theta) = -\sum_{i=1}^N \left[y_i \log\hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right]$$

**Step 4:** This is exactly the binary cross-entropy:

$$\text{BCE} = -\frac{1}{N}\sum_{i=1}^N \left[y_i \log\hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right]$$

**Step 5:** For multi-class segmentation with $K$ classes:

$$\text{CE} = -\frac{1}{N}\sum_{i=1}^N \sum_{k=1}^K y_{i,k} \log\hat{p}_{i,k}$$

**Step 6:** The gradient with respect to logits $z_i$ (pre-softmax):

$$\frac{\partial \text{CE}}{\partial z_{i,k}} = \hat{p}_{i,k} - y_{i,k}$$

This elegant form enables efficient backpropagation.

### 1.4 Potential-Based Reward Shaping (Ng et al., 1999)

**Theorem (Ng, Harada, Russell 1999):** Let $M = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$ be an MDP and $\Phi: \mathcal{S} \to \mathbb{R}$ be a potential function. Define the shaped reward:

$$R'(s, a, s') = R(s, a, s') + \gamma\Phi(s') - \Phi(s)$$

Then the optimal policy $\pi^*$ in the shaped MDP $M' = (\mathcal{S}, \mathcal{A}, P, R', \gamma)$ is identical to the optimal policy in $M$.

**Proof:**

**Step 1:** Define the shaped value function $V'^\pi(s) = V^\pi(s) - \Phi(s)$ for any policy $\pi$.

**Step 2:** Verify the Bellman equation holds under $R'$:

$$V'^\pi(s) = E_\pi\left[\sum_{t=0}^\infty \gamma^t R'(s_t, a_t, s_{t+1}) \mid s_0 = s\right]$$

**Step 3:** Expand:

$$\sum_{t=0}^\infty \gamma^t R'(s_t, a_t, s_{t+1}) = \sum_{t=0}^\infty \gamma^t [R(s_t,a_t,s_{t+1}) + \gamma\Phi(s_{t+1}) - \Phi(s_t)]$$

**Step 4:** The telescoping sum:

$$= \sum_{t=0}^\infty \gamma^t R(s_t,a_t,s_{t+1}) + \sum_{t=0}^\infty [\gamma^{t+1}\Phi(s_{t+1}) - \gamma^t\Phi(s_t)]$$

$$= \sum_{t=0}^\infty \gamma^t R(s_t,a_t,s_{t+1}) - \Phi(s_0)$$

(assuming $\lim_{t\to\infty}\gamma^t\Phi(s_t) = 0$)

**Step 5:** Therefore $V'^{\pi}(s) = V^\pi(s) - \Phi(s)$ for all $\pi$.

**Step 6:** Since $\Phi(s)$ is policy-independent:

$$\pi^* = \arg\max_\pi V'^\pi(s) = \arg\max_\pi [V^\pi(s) - \Phi(s)] = \arg\max_\pi V^\pi(s)$$

Thus the optimal policy is preserved. $\blacksquare$

**Application to segmentation:** Use $\Phi(s) = \text{IoU}(s_{current\_mask}, s_{target})$ to guide the agent toward correct segmentation without changing the optimal solution.

### 1.5 MDP Formulation for Pixel Classification

**State:** $s_t = (I, M_t, t)$ where $I$ is the input image, $M_t \in \{0, 1, \text{unclassified}\}^{H\times W}$ is the partial mask at step $t$.

**Action:** $a_t = (i, j, c)$ — classify pixel $(i,j)$ as class $c$.

**Transition:** Deterministic update $M_{t+1} = M_t$ with $M_{t+1}[i,j] = c$.

**Reward:** $r_t = \text{IoU}(M_{t+1}, M^*) - \text{IoU}(M_t, M^*)$ (incremental IoU improvement).

---

## 2. Algorithm / Method

### 2.1 Pseudocode

```
Algorithm: Pixel Classification RL Agent
─────────────────────────────────────────
Input: Image I, ground truth mask M*
Output: Predicted segmentation mask M

State: s = (image_features, current_mask, classification_confidence_map)
Action: (pixel_location, class_label) or batch of pixels

Initialize policy π_θ with CNN backbone
Initialize shaped reward: R' = ΔIoU + γΦ(s') - Φ(s)

for episode = 1 to num_episodes do
    M₀ ← initial_prediction(I)    // Seed from CNN
    for t = 0 to T_max do
        pixels_t, classes_t ← π_θ(s_t)   // Select pixels to reclassify
        M_{t+1} ← update_mask(M_t, pixels_t, classes_t)
        r_t ← IoU(M_{t+1}, M*) - IoU(M_t, M*)
        r_t' ← r_t + γΦ(s_{t+1}) - Φ(s_t)  // Shape reward
    end for
    Update π_θ via policy gradient with baseline
end for
```

### 2.2 Complexity Analysis

- **Per-pixel action:** $O(K)$ for $K$ classes
- **IoU computation:** $O(N)$ per step
- **CNN feature extraction:** $O(N \cdot C^2 \cdot K^2 \cdot L)$
- **Shaped reward computation:** $O(N)$ for potential function evaluation
- **Total per episode:** $O(T \cdot N \cdot C^2 \cdot K^2 \cdot L)$

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

- **State:** Image features + current partial segmentation mask + confidence map
- **Action:** Select pixel(s) to classify and their class labels
- **Reward:** Incremental IoU improvement with potential-based shaping
- **Discount:** $\gamma = 0.99$ (long episodes as mask is built incrementally)

### 3.2 Why RL?

1. **Sequential decision-making:** Classification order matters — boundary pixels benefit from knowing interior classifications first
2. **Reward shaping:** Can guide agent toward high-IoU regions without hand-crafting the path
3. **Adaptive computation:** Agent can focus on uncertain regions, skipping easy pixels
4. **Non-differentiable metrics:** IoU is non-differentiable with respect to discrete predictions; RL optimizes it directly

---

## 4. Dataset

| Dataset | Classes | Size | Description |
|---------|---------|------|-------------|
| PASCAL VOC 2012 | 21 | 11,530 | Semantic segmentation benchmark |
| Cityscapes | 30 | 5,000 | Urban driving scenes |
| ADE20K | 150 | 25,210 | Diverse scenes |
| COCO-Stuff | 171 | 164K | Things and stuff |

---

## 5. Key Equations Summary

| Equation | Description |
|----------|-------------|
| $\text{IoU} = \frac{\|A \cap B\|}{\|A \cup B\|}$ | Intersection over Union |
| $\text{Dice} = \frac{2\text{IoU}}{1+\text{IoU}}$ | Dice-IoU relationship |
| $\text{BCE} = -\frac{1}{N}\sum[y\log\hat{p} + (1-y)\log(1-\hat{p})]$ | Binary cross-entropy |
| $R' = R + \gamma\Phi(s') - \Phi(s)$ | Potential-based reward shaping |
| $\frac{\partial\text{CE}}{\partial z_k} = \hat{p}_k - y_k$ | Cross-entropy gradient |

---

## 6. References

1. Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations. *ICML*.
2. Rahman, M. A., & Wang, Y. (2016). Optimizing intersection-over-union in deep neural networks for image segmentation. *ISVC*.
3. Milletari, F., Navab, N., & Ahmadi, S. A. (2016). V-Net: Fully convolutional neural networks for volumetric medical image segmentation. *3DV*.
4. Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. *CVPR*.
