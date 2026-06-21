![Module Logo](../logo.png)

# Active Object Localization via Reinforcement Learning

## Overview

This module formulates object localization as an active perception problem where a DQN agent sequentially transforms a bounding box to localize objects. The mathematical framework covers the value of information in active perception, Bayesian posterior updates for object search, information gain derivation, and the complete DQN architecture for active localization with carefully designed state, action, and reward representations.

## Prerequisites

- Bayesian inference (prior/posterior, sequential updating)
- Information theory (entropy, KL divergence, mutual information)
- Decision theory (value of information, optimal stopping)
- Reinforcement learning (DQN, experience replay, target networks)
- Optimization (Bellman optimality, fixed-point iteration)

---

## 1. Mathematical Foundations

### 1.1 Active Perception: Value of Information

**Definition:** The Value of Information (VoI) of an observation $o$ before taking action $a$ in state $s$:

$$\text{VoI}(o | s) = E_o\left[\max_a Q^*(s', a)\right] - \max_a Q^*(s, a)$$

where $s' = \text{update}(s, o)$ is the posterior state after observation.

**Step 1:** In the object search setting, the state is a belief $b$ over possible object locations $\ell \in \mathcal{L}$:

$$b(\ell) = P(\text{object at } \ell | \text{observations so far})$$

**Step 2:** An action $a_t$ (look at region $R$) yields observation $o_t$. The VoI of looking at $R$:

$$\text{VoI}(R) = E_{o|R}\left[V^*(b_{t+1})\right] - V^*(b_t)$$

where $b_{t+1}$ is the updated belief after observing $R$.

**Step 3:** The optimal search strategy maximizes cumulative VoI while minimizing search cost:

$$\pi^* = \arg\max_\pi E_\pi\left[\sum_t \gamma^t(r_t - c)\right]$$

where $c$ is the per-step observation cost.

### 1.2 Bayesian Object Search: Posterior Update

**Step 1:** Prior belief over locations: $b_0(\ell) = P_0(\ell)$ (e.g., uniform).

**Step 2:** Observation model: $P(o_t | \ell, a_t)$ — probability of seeing $o_t$ when object is at $\ell$ and we look at region $a_t$.

**Step 3:** Bayesian update:

$$b_{t+1}(\ell) = P(\ell | o_1, \ldots, o_{t+1}) = \frac{P(o_{t+1} | \ell, a_{t+1}) \cdot b_t(\ell)}{\sum_{\ell'} P(o_{t+1} | \ell', a_{t+1}) \cdot b_t(\ell')}$$

**Step 4:** For a detection-type observation (object present/absent in region $R$):

$$P(o_t = \text{present} | \ell, R) = \begin{cases} p_d & \text{if } \ell \in R \\ p_{fa} & \text{if } \ell \notin R \end{cases}$$

where $p_d$ is detection probability and $p_{fa}$ is false alarm rate.

**Step 5:** After observing "absent" in region $R$:

$$b_{t+1}(\ell) \propto \begin{cases} (1-p_d) \cdot b_t(\ell) & \ell \in R \\ (1-p_{fa}) \cdot b_t(\ell) & \ell \notin R \end{cases}$$

**Step 6:** The belief concentrates on un-searched regions as negative evidence accumulates.

### 1.3 Information Gain Derivation

**Definition:** Information Gain (IG) from observing $Y$ about variable $X$:

$$IG = I(X; Y) = H(X) - H(X|Y)$$

**Step 1:** Expand entropy:

$$H(X) = -\sum_x P(x)\log P(x)$$

**Step 2:** Conditional entropy:

$$H(X|Y) = -\sum_y P(y)\sum_x P(x|y)\log P(x|y)$$

**Step 3:** Therefore:

$$IG = H(X) - H(X|Y) = \sum_y P(y) \text{KL}(P(X|Y=y) \| P(X))$$

**Step 4:** Equivalently using KL divergence:

$$IG = E_Y\left[\text{KL}(P(X|Y) \| P(X))\right] = E_Y\left[\sum_x P(x|y)\log\frac{P(x|y)}{P(x)}\right]$$

**Step 5:** For active localization, $X = \ell$ (object location), $Y = o$ (observation at selected region):

$$IG(R) = H(b_t) - E_{o|R}[H(b_{t+1})]$$

**Step 6:** This can be computed in closed form for discrete location spaces:

$$IG(R) = -\sum_\ell b_t(\ell)\log b_t(\ell) + \sum_o P(o|R) \sum_\ell P(\ell|o)\log P(\ell|o)$$

**Step 7 (Submodularity of information gain):** For a set of observations $\mathcal{O}_A$:

$$IG(\mathcal{O}_A \cup \{o\}) - IG(\mathcal{O}_A) \geq IG(\mathcal{O}_B \cup \{o\}) - IG(\mathcal{O}_B) \quad \text{for } \mathcal{O}_A \subseteq \mathcal{O}_B$$

This guarantees the greedy strategy achieves $(1-1/e)$ of optimal. $\blacksquare$

### 1.4 DQN for Active Localization

**Step 1 (Action space):** Define 9 transformation actions:

$$\mathcal{A} = \{\text{left, right, up, down, bigger, smaller, fatter, taller, trigger}\}$$

Each action modifies the current box $b_t$:
- Translations: shift by $\alpha \cdot w$ or $\alpha \cdot h$ (typically $\alpha = 0.2$)
- Scale changes: multiply dimensions by $(1 \pm \alpha)$
- Aspect ratio: adjust $w/h$
- Trigger: declare localization complete

**Step 2 (State representation):**

$$s_t = [\text{CNN}(I_{b_t}); \mathbf{h}_t]$$

where $\text{CNN}(I_{b_t})$ is the feature vector from the current box region and $\mathbf{h}_t$ encodes action history.

**Step 3 (DQN Bellman optimality):**

$$Q^*(s, a) = E\left[r + \gamma\max_{a'} Q^*(s', a') | s, a\right]$$

**Step 4 (Loss function):**

$$\mathcal{L}(\theta) = E_{(s,a,r,s')\sim\mathcal{D}}\left[\left(r + \gamma\max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a)\right)^2\right]$$

where $\theta^-$ are target network parameters (periodically copied from $\theta$).

**Step 5 (Reward design):**

$$r_t = \begin{cases} +\eta & \text{if IoU}(b_{t+1}, b_{gt}) > \text{IoU}(b_t, b_{gt}) \\ -\eta & \text{if IoU}(b_{t+1}, b_{gt}) < \text{IoU}(b_t, b_{gt}) \\ +\omega & \text{if trigger and IoU} > \tau_{pos} \\ -\omega & \text{if trigger and IoU} < \tau_{pos} \end{cases}$$

---

## 2. Algorithm / Method

### 2.1 Pseudocode

```
Algorithm: DQN Active Object Localization
─────────────────────────────────────────
Input: Image I, initial box b₀ (full image or proposal)
Output: Localized bounding box b*

Initialize Q_θ, target Q_θ⁻, replay buffer D

for episode = 1 to M do
    b₀ ← initialize_box(I)
    s₀ ← [CNN(I, b₀); history₀]
    for t = 0 to T_max do
        aₜ ← ε-greedy: argmax_a Q_θ(sₜ, a) w.p. 1-ε, random w.p. ε
        b_{t+1} ← transform(bₜ, aₜ)
        rₜ ← sign(IoU(b_{t+1}, b_gt) - IoU(bₜ, b_gt)) · η
        s_{t+1} ← [CNN(I, b_{t+1}); update_history(aₜ)]
        Store (sₜ, aₜ, rₜ, s_{t+1}) in D
        if aₜ == trigger: break
        
        // Learn
        Sample minibatch from D
        y ← r + γ max_a Q_θ⁻(s', a)
        ∇θ L = ∇θ (y - Q_θ(s, a))²
        Periodically: θ⁻ ← θ
    end for
end for
```

### 2.2 Complexity Analysis

- **CNN feature extraction:** $O(k^2 \cdot C^2 \cdot L)$ per box
- **Q-network forward pass:** $O(|\theta|)$
- **Action application:** $O(1)$
- **IoU computation:** $O(1)$
- **Per episode:** $O(T \cdot k^2 C^2 L)$ dominated by CNN
- **Replay buffer operations:** $O(1)$ amortized

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

- **State:** CNN features of current box region + action history vector
- **Action:** Discrete set of box transformations + trigger action
- **Reward:** Positive/negative signal based on IoU improvement + terminal bonus
- **Transition:** Deterministic box transformation
- **Horizon:** Maximum $T$ steps before forced termination

### 3.2 Why RL?

1. **Coarse-to-fine search:** Agent naturally learns hierarchical search — large moves first, then fine adjustments
2. **Non-greedy strategies:** RL can learn to temporarily move away from the target to achieve better long-term alignment
3. **Learned stopping:** The trigger action is a natural decision boundary learned through experience
4. **Transfer across categories:** The action semantics transfer across object classes

---

## 4. Dataset

| Dataset | Task | Size | Description |
|---------|------|------|-------------|
| VOC 2007/2012 | Detection | 11,530 | 20-class localization |
| MS COCO | Detection | 330K | 80-class detection |
| ImageNet LOC | Localization | 476K | 1000-class localization |
| Object Discovery | Unsupervised | Variable | Co-localization |

---

## 5. Key Equations Summary

| Equation | Description |
|----------|-------------|
| $\text{VoI}(o|s) = E_o[\max_a Q^*(s',a)] - \max_a Q^*(s,a)$ | Value of Information |
| $b_{t+1}(\ell) \propto P(o_{t+1}|\ell,a_{t+1})b_t(\ell)$ | Bayesian belief update |
| $IG(R) = H(b_t) - E_{o|R}[H(b_{t+1})]$ | Information gain |
| $Q^*(s,a) = E[r + \gamma\max_{a'}Q^*(s',a')]$ | Bellman optimality |
| $\mathcal{L}(\theta) = (r + \gamma\max_{a'}Q_{\theta^-}(s',a') - Q_\theta(s,a))^2$ | DQN loss |

---

## 6. References

1. Caicedo, J. C., & Lazebnik, S. (2015). Active object localization with deep reinforcement learning. *ICCV*.
2. Gonzalez-Garcia, A., Veber, A., Lapin, M., & Schiele, B. (2015). Active search for real-time vision. *ICRA*.
3. Bellver, M., Giro-i-Nieto, X., Marques, F., & Torres, J. (2016). Hierarchical object detection with deep reinforcement learning. *NeurIPS Workshop*.
4. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529-533.
