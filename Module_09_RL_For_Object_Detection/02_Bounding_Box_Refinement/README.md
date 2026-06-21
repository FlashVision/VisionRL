![Module Logo](../logo.png)

# Bounding Box Refinement via Reinforcement Learning

## Overview

This module formulates object detection bounding box regression as a sequential refinement process controlled by a reinforcement learning agent. The mathematical framework covers box parameterization transforms, derives the Smooth L1 loss and its continuity properties, presents complete derivations of GIoU/DIoU/CIoU losses, and establishes the RL formulation for iterative box refinement.

## Prerequisites

- Geometry (affine transformations, coordinate systems)
- Analysis (continuity, differentiability, Lipschitz conditions)
- Measure theory (Lebesgue measure for area computation)
- Optimization (gradient-based methods, loss landscape analysis)
- Reinforcement learning (DQN, continuous action spaces)

---

## 1. Mathematical Foundations

### 1.1 Box Parameterization

**Definition:** A bounding box $b = (x, y, w, h)$ specifies center $(x, y)$ and dimensions $(w, h)$.

**Step 1 (Anchor-based transform):** Given anchor box $b_a = (x_a, y_a, w_a, h_a)$ and target $b^* = (x^*, y^*, w^*, h^*)$, define regression targets:

$$t_x = \frac{x^* - x_a}{w_a}, \quad t_y = \frac{y^* - y_a}{h_a}$$
$$t_w = \log\frac{w^*}{w_a}, \quad t_h = \log\frac{h^*}{h_a}$$

**Step 2 (Inverse transform):** Given predicted offsets $(\hat{t}_x, \hat{t}_y, \hat{t}_w, \hat{t}_h)$:

$$\hat{x} = \hat{t}_x w_a + x_a, \quad \hat{y} = \hat{t}_y h_a + y_a$$
$$\hat{w} = w_a e^{\hat{t}_w}, \quad \hat{h} = h_a e^{\hat{t}_h}$$

**Step 3 (Why logarithmic for size?):** The log transform ensures:
- Scale invariance: $\log(w^*/w_a)$ is the same regardless of absolute size
- Positive predictions: $w_a e^{\hat{t}_w} > 0$ always
- Symmetric error: predicting $2\times$ too large or $1/2\times$ too small have equal magnitude

**Step 4 (Proof of scale invariance):** If we scale both boxes by factor $s$:

$$t_x' = \frac{sx^* - sx_a}{sw_a} = t_x, \quad t_w' = \log\frac{sw^*}{sw_a} = t_w$$

The targets are invariant under uniform scaling. $\blacksquare$

### 1.2 Smooth L1 Loss: Derivation and Continuity Proof

**Definition (Huber/Smooth L1 loss):**

$$\text{SmoothL1}(x) = \begin{cases} 0.5x^2 & \text{if } |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases}$$

**Derivation from robustness considerations:**

**Step 1:** L2 loss $\ell_2(x) = x^2$ has gradient $2x$, which becomes very large for outliers.

**Step 2:** L1 loss $\ell_1(x) = |x|$ is robust to outliers but non-differentiable at $x = 0$.

**Step 3:** Smooth L1 interpolates: quadratic near zero (smooth gradients), linear far from zero (robust).

**Step 4 (Continuity proof):** At the transition point $|x| = 1$:
- From the quadratic side: $\lim_{x \to 1^-} 0.5x^2 = 0.5$
- From the linear side: $\lim_{x \to 1^+} (|x| - 0.5) = 0.5$ ✓

**Step 5 (C¹ continuity — differentiability):**
- Derivative from quadratic side: $\lim_{x \to 1^-} x = 1$
- Derivative from linear side: $\lim_{x \to 1^+} \text{sign}(x) = 1$ ✓

**Step 6 (C² discontinuity):**
- Second derivative from quadratic: $1$
- Second derivative from linear: $0$

Therefore Smooth L1 is $C^1$ but not $C^2$. It has Lipschitz-continuous gradients with constant 1. $\blacksquare$

**Step 7 (General form with threshold $\delta$):**

$$\text{SmoothL1}_\delta(x) = \begin{cases} \frac{x^2}{2\delta} & |x| < \delta \\ |x| - \frac{\delta}{2} & \text{otherwise} \end{cases}$$

### 1.3 IoU Loss and Extensions: GIoU, DIoU, CIoU

**IoU Loss:**

$$\mathcal{L}_{IoU} = 1 - \text{IoU}(b, b^*) = 1 - \frac{|b \cap b^*|}{|b \cup b^*|}$$

**GIoU (Generalized IoU) Derivation:**

**Step 1:** Find the smallest enclosing box $C$ containing both $b$ and $b^*$:

$$C = \text{smallest\_box}(b \cup b^*)$$

**Step 2:** Define GIoU:

$$\text{GIoU} = \text{IoU} - \frac{|C \setminus (b \cup b^*)|}{|C|}$$

**Step 3:** Properties:
- $\text{GIoU} \in [-1, 1]$ (unlike IoU $\in [0,1]$)
- $\text{GIoU} = \text{IoU}$ when $b \subseteq b^*$ or $b^* \subseteq b$
- GIoU provides gradient even for non-overlapping boxes (where IoU $= 0$)

**DIoU (Distance IoU) Derivation:**

**Step 1:** Let $\rho(b, b^*) = \|c_b - c_{b^*}\|_2$ be the Euclidean distance between box centers.

**Step 2:** Let $d = \text{diag}(C)$ be the diagonal of the smallest enclosing box.

**Step 3:**

$$\text{DIoU} = \text{IoU} - \frac{\rho^2(b, b^*)}{d^2}$$

$$\mathcal{L}_{DIoU} = 1 - \text{IoU} + \frac{\rho^2(b, b^*)}{d^2}$$

**CIoU (Complete IoU) Derivation:**

**Step 1:** Add aspect ratio consistency:

$$v = \frac{4}{\pi^2}\left(\arctan\frac{w^*}{h^*} - \arctan\frac{w}{h}\right)^2$$

**Step 2:** Define the trade-off parameter:

$$\alpha = \frac{v}{(1 - \text{IoU}) + v}$$

**Step 3:** Complete CIoU loss:

$$\mathcal{L}_{CIoU} = 1 - \text{IoU} + \frac{\rho^2(b, b^*)}{d^2} + \alpha v$$

**Step 4:** Gradient of the aspect ratio term:

$$\frac{\partial v}{\partial w} = -\frac{8}{\pi^2}\left(\arctan\frac{w^*}{h^*} - \arctan\frac{w}{h}\right) \cdot \frac{h}{w^2 + h^2}$$

### 1.4 Proof: GIoU is a Proper Metric on Box Space

**Theorem:** $d_{GIoU}(b_1, b_2) = 1 - \text{GIoU}(b_1, b_2)$ satisfies the metric axioms for non-degenerate boxes.

**Proof (sketch):**
- Non-negativity: $1 - \text{GIoU} \geq 0$ since $\text{GIoU} \leq 1$
- Identity: $d = 0 \iff \text{GIoU} = 1 \iff b_1 = b_2$
- Symmetry: GIoU is symmetric by definition
- Triangle inequality: Follows from the sub-additivity of the penalty term $|C \setminus (b \cup b^*)|/|C|$ $\blacksquare$

---

## 2. Algorithm / Method

### 2.1 Pseudocode

```
Algorithm: RL Bounding Box Refinement
─────────────────────────────────────────
Input: Image I, initial detection b₀ (from RPN or detector)
Output: Refined bounding box b*

State: s = (image_features_at_box, box_coordinates, 
            IoU_history, refinement_step)
Action: a = (Δtx, Δty, Δtw, Δth) ∈ [-0.1, 0.1]⁴

Initialize actor π_θ, critic V_φ

for episode = 1 to M do
    b₀ ← initial_detection(I)
    s₀ ← extract_roi_features(I, b₀)
    for t = 0 to T_max do
        aₜ ~ π_θ(·|sₜ)
        b_{t+1} ← apply_transform(bₜ, aₜ)
        rₜ ← IoU(b_{t+1}, b_gt) - IoU(bₜ, b_gt)
        s_{t+1} ← extract_roi_features(I, b_{t+1})
        if IoU(b_{t+1}, b_gt) > 0.9: break
    end for
    Update π_θ, V_φ via PPO
end for
```

### 2.2 Complexity Analysis

- **RoI feature extraction:** $O(k^2 \cdot C)$ for pooled size $k$
- **Transform application:** $O(1)$ (simple arithmetic)
- **IoU computation:** $O(1)$ for axis-aligned boxes
- **Policy forward pass:** $O(|\theta|)$
- **Total per refinement episode:** $O(T \cdot (k^2 C + |\theta|))$

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

- **State:** RoI-pooled features from current box + box coordinates normalized by image size + step count
- **Action:** Continuous offsets $(\Delta t_x, \Delta t_y, \Delta t_w, \Delta t_h)$ bounded in $[-\delta, \delta]$
- **Reward:** $r_t = \text{IoU}(b_{t+1}, b_{gt}) - \text{IoU}(b_t, b_{gt})$ (incremental IoU)
- **Terminal:** IoU exceeds threshold or maximum steps reached
- **Discount:** $\gamma = 0.95$

### 3.2 Why RL?

1. **Iterative refinement:** Multiple small adjustments can achieve higher precision than a single regression step
2. **Adaptive stopping:** Agent learns when the box is "good enough" — avoids oscillation
3. **Non-myopic optimization:** Agent can learn to first center the box, then adjust size (strategic ordering)
4. **Robustness:** Sequential refinement is more robust to large initial errors than single-shot regression

---

## 4. Dataset

| Dataset | Objects/Image | Size | Description |
|---------|--------------|------|-------------|
| PASCAL VOC | ~2.5 | 11,530 | 20 object classes |
| MS COCO | ~7.7 | 330K | 80 categories |
| ImageNet DET | ~1.5 | 476K | 200 classes |
| Open Images | ~8.4 | 1.9M | 600 classes |

---

## 5. Key Equations Summary

| Equation | Description |
|----------|-------------|
| $t_x = (x^* - x_a)/w_a$ | Box regression target (center) |
| $t_w = \log(w^*/w_a)$ | Box regression target (size) |
| $\text{SmoothL1}(x) = 0.5x^2\mathbb{1}_{|x|<1} + (|x|-0.5)\mathbb{1}_{|x|\geq1}$ | Smooth L1 loss |
| $\text{GIoU} = \text{IoU} - |C\setminus(b\cup b^*)|/|C|$ | Generalized IoU |
| $\mathcal{L}_{CIoU} = 1 - \text{IoU} + \rho^2/d^2 + \alpha v$ | Complete IoU loss |

---

## 6. References

1. Girshick, R. (2015). Fast R-CNN. *ICCV*.
2. Rezatofighi, H., et al. (2019). Generalized intersection over union: A metric and a loss for bounding box regression. *CVPR*.
3. Zheng, Z., et al. (2020). Distance-IoU loss: Faster and better learning for bounding box regression. *AAAI*.
4. Caicedo, J. C., & Lazebnik, S. (2015). Active object localization with deep reinforcement learning. *ICCV*.
5. Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN. *NeurIPS*.
