![Module Logo](../logo.png)

# Medical Image Enhancement via Reinforcement Learning

## Overview

This module applies reinforcement learning to medical image enhancement, combining rigorous quality assessment metrics with safety-aware RL for clinical applications. The mathematical framework derives sensitivity/specificity from the confusion matrix, proves the probabilistic interpretation of ROC-AUC, derives Dice loss gradients for medical segmentation, and formulates constrained MDPs for safe RL in medical imaging where errors have serious consequences.

## Prerequisites

- Biostatistics (sensitivity, specificity, predictive values)
- Probability theory (ROC analysis, rank statistics)
- Differential calculus (gradient derivation for soft metrics)
- Constrained optimization (Lagrangian methods, barrier functions)
- Reinforcement learning (constrained MDPs, safe exploration)

---

## 1. Mathematical Foundations

### 1.1 Sensitivity and Specificity: Derivation from Confusion Matrix

**Definition:** For a binary classifier with confusion matrix entries:
- True Positives ($TP$): correctly identified positives
- True Negatives ($TN$): correctly identified negatives
- False Positives ($FP$): incorrectly identified as positive (Type I error)
- False Negatives ($FN$): incorrectly identified as negative (Type II error)

**Step 1 (Sensitivity / True Positive Rate):**

$$\text{Sensitivity} = \text{TPR} = P(\hat{Y}=1 | Y=1) = \frac{TP}{TP + FN}$$

This measures the ability to detect disease when present.

**Step 2 (Specificity / True Negative Rate):**

$$\text{Specificity} = \text{TNR} = P(\hat{Y}=0 | Y=0) = \frac{TN}{TN + FP}$$

This measures the ability to correctly rule out disease.

**Step 3 (Positive Predictive Value):** By Bayes' theorem:

$$\text{PPV} = P(Y=1|\hat{Y}=1) = \frac{\text{Sensitivity} \cdot \text{Prevalence}}{\text{Sensitivity} \cdot \text{Prevalence} + (1-\text{Specificity}) \cdot (1-\text{Prevalence})}$$

**Step 4 (F1 Score as harmonic mean):**

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$

**Step 5 (Relationship to Dice coefficient):** For binary segmentation masks:

$$\text{Dice} = \frac{2|A \cap B|}{|A| + |B|} = \frac{2TP}{2TP + FP + FN} = F_1$$

The Dice coefficient and F1 score are identical for binary classification.

### 1.2 ROC-AUC as Probability: Proof

**Theorem:** The area under the ROC curve equals the probability that a randomly chosen positive example has a higher score than a randomly chosen negative example:

$$\text{AUC} = P(s(x^+) > s(x^-))$$

where $s(\cdot)$ is the classifier's score function.

**Proof:**

**Step 1:** The ROC curve plots TPR vs FPR as the threshold $\tau$ varies:

$$\text{TPR}(\tau) = P(s(x) > \tau | Y=1), \quad \text{FPR}(\tau) = P(s(x) > \tau | Y=0)$$

**Step 2:** The AUC is:

$$\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(t)) \, dt$$

**Step 3:** Change variables. Let $f_0(s)$ and $f_1(s)$ be the score densities for negative and positive classes:

$$\text{AUC} = \int_{-\infty}^{\infty} \text{TPR}(\tau) \cdot f_0(\tau) \, d\tau$$

(since $d(\text{FPR}) = -f_0(\tau)d\tau$ and we integrate right-to-left)

**Step 4:** Substitute TPR:

$$\text{AUC} = \int_{-\infty}^{\infty} P(s(x^+) > \tau) \cdot f_0(\tau) \, d\tau$$

$$= \int_{-\infty}^{\infty} \int_\tau^{\infty} f_1(s) \, ds \cdot f_0(\tau) \, d\tau$$

**Step 5:** This double integral is:

$$= \int\int_{\{s > \tau\}} f_1(s) f_0(\tau) \, ds \, d\tau = P(s(x^+) > s(x^-)) \quad \blacksquare$$

**Step 6 (Wilcoxon-Mann-Whitney statistic):** The empirical AUC:

$$\widehat{\text{AUC}} = \frac{1}{n_+ n_-}\sum_{i: y_i=1}\sum_{j: y_j=0}\mathbb{1}[s(x_i) > s(x_j)]$$

### 1.3 Dice Loss Gradient Derivation

**Step 1:** Soft Dice loss for segmentation with predictions $p_i \in [0,1]$ and targets $g_i \in \{0,1\}$:

$$\mathcal{L}_{Dice} = 1 - \frac{2\sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}$$

where $\epsilon$ is a smoothing term.

**Step 2:** Let $A = 2\sum_i p_i g_i + \epsilon$ (numerator) and $B = \sum_i p_i + \sum_i g_i + \epsilon$ (denominator).

**Step 3:** Gradient with respect to prediction $p_j$:

$$\frac{\partial\mathcal{L}_{Dice}}{\partial p_j} = -\frac{\frac{\partial A}{\partial p_j} B - A\frac{\partial B}{\partial p_j}}{B^2}$$

**Step 4:** Compute partial derivatives:

$$\frac{\partial A}{\partial p_j} = 2g_j, \quad \frac{\partial B}{\partial p_j} = 1$$

**Step 5:** Substitute:

$$\frac{\partial\mathcal{L}_{Dice}}{\partial p_j} = -\frac{2g_j \cdot B - A \cdot 1}{B^2} = \frac{A - 2g_j B}{B^2}$$

$$= \frac{2\sum_i p_i g_i + \epsilon - 2g_j(\sum_i p_i + \sum_i g_i + \epsilon)}{(\sum_i p_i + \sum_i g_i + \epsilon)^2}$$

**Step 6 (Vectorized form):**

$$\nabla_\mathbf{p}\mathcal{L}_{Dice} = \frac{2[\mathbf{g}(\sum p_i + \sum g_i) - (\sum p_i g_i)\mathbf{1}]}{(\sum p_i + \sum g_i + \epsilon)^2}$$

### 1.4 Safe RL: Constrained MDP for Medical Imaging

**Definition:** A Constrained MDP (CMDP) adds safety constraints:

$$\mathcal{M}_C = (\mathcal{S}, \mathcal{A}, P, R, \{C_k\}_{k=1}^K, \{d_k\}_{k=1}^K, \gamma)$$

where $C_k: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ are cost functions and $d_k$ are cost thresholds.

**Step 1 (Constrained objective):**

$$\max_\pi E_\pi\left[\sum_t\gamma^t R(s_t, a_t)\right] \quad \text{s.t.} \quad E_\pi\left[\sum_t\gamma^t C_k(s_t, a_t)\right] \leq d_k \quad \forall k$$

**Step 2 (Lagrangian relaxation):**

$$\mathcal{L}(\pi, \boldsymbol{\lambda}) = J_R(\pi) - \sum_k \lambda_k(J_{C_k}(\pi) - d_k)$$

**Step 3 (Primal-dual optimization):**

$$\pi^* = \arg\max_\pi \min_{\lambda \geq 0} \mathcal{L}(\pi, \boldsymbol{\lambda})$$

Update policy: $\theta \leftarrow \theta + \eta_\theta\nabla_\theta\mathcal{L}$

Update multipliers: $\lambda_k \leftarrow \max(0, \lambda_k + \eta_\lambda(J_{C_k}(\pi) - d_k))$

**Step 4 (Medical constraints):**
- $C_1$: Maximum false negative rate (missing disease) $\leq d_1$
- $C_2$: Maximum radiation dose (for enhancement involving acquisition) $\leq d_2$
- $C_3$: Maximum processing time (real-time clinical requirement) $\leq d_3$

**Step 5 (Safety guarantee):** Under Slater's condition (strict feasibility), strong duality holds, and the primal-dual method converges to the constrained optimum. $\blacksquare$

---

## 2. Algorithm / Method

### 2.1 Pseudocode

```
Algorithm: Safe RL for Medical Image Enhancement
─────────────────────────────────────────────────
Input: Medical images, quality/safety constraints
Output: Enhanced images satisfying clinical constraints

State: s = (image_features, current_quality_metrics, 
            processing_time, enhancement_history)
Action: a = (enhancement_operation, parameter_values)
Constraints: FNR ≤ d₁, processing_time ≤ d₂

Initialize policy π_θ, Lagrange multipliers λ

for episode = 1 to M do
    I₀ ← sample_medical_image()
    for t = 0 to T_max do
        aₜ ~ π_θ(·|sₜ)
        I_{t+1} ← apply_enhancement(Iₜ, aₜ)
        
        // Compute reward and costs
        rₜ ← ΔDice + ΔAUC (segmentation/detection improvement)
        c₁ₜ ← false_negative_rate(I_{t+1})
        c₂ₜ ← time_elapsed
        
        s_{t+1} ← update_state(I_{t+1})
    end for
    
    // Constrained PPO update
    L ← J_R(π) - λ₁(J_C₁ - d₁) - λ₂(J_C₂ - d₂)
    Update θ to maximize L
    Update λₖ ← max(0, λₖ + η(J_Cₖ - dₖ))
end for
```

### 2.2 Complexity Analysis

- **Medical image processing:** $O(HW \cdot C)$ per operation
- **Dice computation:** $O(HW)$ for full image
- **AUC computation:** $O(N\log N)$ for $N$ pixels (sorting)
- **Constraint evaluation:** $O(HW)$ per constraint
- **Policy update:** $O(|\theta|)$ plus Lagrange multiplier update $O(K)$

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

$$\mathcal{M}_C = (\mathcal{S}, \mathcal{A}, P, R, C, d, \gamma)$$

- **State:** Medical image features + current diagnostic quality metrics
- **Action:** Enhancement operations (contrast adjustment, noise reduction, sharpening) with continuous parameters
- **Reward:** Improvement in downstream task (segmentation Dice, detection AUC)
- **Constraints:** FNR $\leq$ threshold, processing time $\leq$ budget, no artifact introduction
- **Safety guarantee:** Lagrangian method ensures constraint satisfaction at convergence

### 3.2 Why RL?

1. **Safety-critical:** Constrained RL naturally enforces clinical requirements
2. **Task-driven enhancement:** Optimize for downstream diagnostic performance, not just visual quality
3. **Patient-adaptive:** Different pathologies/modalities need different enhancement strategies
4. **Regulatory compliance:** Explicit constraints align with medical device regulations

---

## 4. Dataset

| Dataset | Modality | Size | Task |
|---------|----------|------|------|
| ChestX-ray14 | X-ray | 112,120 | 14 pathologies |
| BraTS | MRI | 2,000+ | Brain tumor segmentation |
| ISIC 2018 | Dermoscopy | 10,015 | Skin lesion analysis |
| DRIVE | Retinal | 40 images | Vessel segmentation |

---

## 5. Key Equations Summary

| Equation | Description |
|----------|-------------|
| $\text{AUC} = P(s(x^+) > s(x^-))$ | AUC probabilistic interpretation |
| $\text{Sens} = TP/(TP+FN)$ | Sensitivity (recall) |
| $\nabla_{p_j}\mathcal{L}_{Dice} = \frac{A-2g_jB}{B^2}$ | Dice loss gradient |
| $\max_\pi J_R(\pi)$ s.t. $J_{C_k}(\pi)\leq d_k$ | Constrained MDP |
| $\lambda_k \leftarrow \max(0,\lambda_k+\eta(J_{C_k}-d_k))$ | Dual variable update |

---

## 6. References

1. Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861-874.
2. Milletari, F., Navab, N., & Ahmadi, S. A. (2016). V-Net: Fully convolutional neural networks for volumetric medical image segmentation. *3DV*.
3. Achiam, J., Held, D., Tamar, A., & Abbeel, P. (2017). Constrained policy optimization. *ICML*.
4. Litjens, G., et al. (2017). A survey on deep learning in medical image analysis. *Medical Image Analysis*, 42, 60-88.
