![Module Logo](../logo.png)

# Autonomous Image Editor via Reinforcement Learning

## Overview

This module builds a complete autonomous image editing system using PPO-based reinforcement learning, combining actor-critic architectures for visual processing with multi-metric reward design. The mathematical framework provides the full PPO derivation for image enhancement, Pareto-optimal multi-metric reward design, the actor-critic architecture derivation, and statistical significance testing for evaluation.

## Prerequisites

- Policy optimization (trust regions, clipping, importance sampling)
- Multi-objective optimization (Pareto frontiers, scalarization)
- Neural network architectures (actor-critic, shared representations)
- Statistics (hypothesis testing, t-tests, confidence intervals)
- Reinforcement learning (advantage estimation, PPO)

---

## 1. Mathematical Foundations

### 1.1 Complete PPO Derivation for Image Enhancement

**Step 1 (Policy gradient foundation):** The objective is to maximize expected return:

$$J(\theta) = E_{\tau\sim\pi_\theta}\left[\sum_{t=0}^T \gamma^t r_t\right]$$

**Step 2 (Policy gradient theorem):**

$$\nabla_\theta J(\theta) = E_{\pi_\theta}\left[\sum_t\nabla_\theta\log\pi_\theta(a_t|s_t) A^{\pi_\theta}(s_t, a_t)\right]$$

where $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ is the advantage function.

**Step 3 (Importance sampling for off-policy correction):** When using old data from $\pi_{old}$:

$$J(\theta) = E_{\pi_{old}}\left[\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)} A^{\pi_{old}}(s, a)\right]$$

Define the probability ratio: $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$.

**Step 4 (Trust region motivation):** Unconstrained importance sampling can lead to destructively large updates. TRPO constrains:

$$\max_\theta E[r_t(\theta)A_t] \quad \text{s.t.} \quad \text{KL}(\pi_{old}\|\pi_\theta) \leq \delta$$

**Step 5 (PPO clipped objective):** Replace the constraint with clipping:

$$L^{CLIP}(\theta) = E_t\left[\min\left(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t\right)\right]$$

**Step 6 (Derivation of clipping behavior):**
- When $A_t > 0$ (good action): $L = \min(r_t A_t, (1+\epsilon)A_t)$. Caps the benefit at $(1+\epsilon)A_t$.
- When $A_t < 0$ (bad action): $L = \min(r_t A_t, (1-\epsilon)A_t) = \max(r_t|A_t|, (1-\epsilon)|A_t|) \cdot (-1)$. Prevents too much decrease.

**Step 7 (Value function loss):**

$$L^{VF}(\phi) = E_t\left[(V_\phi(s_t) - V_t^{target})^2\right]$$

where $V_t^{target} = \sum_{k=0}^{T-t}\gamma^k r_{t+k}$ (or GAE target).

**Step 8 (Entropy bonus for exploration):**

$$S[\pi_\theta](s_t) = -\sum_a \pi_\theta(a|s_t)\log\pi_\theta(a|s_t)$$

**Step 9 (Total PPO objective):**

$$L^{PPO}(\theta) = E_t\left[L^{CLIP}(\theta) - c_1 L^{VF}(\phi) + c_2 S[\pi_\theta](s_t)\right]$$

**Step 10 (Generalized Advantage Estimation):**

$$\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}^V$$

where $\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD residual.

### 1.2 Multi-Metric Reward Design (Pareto Optimality)

**Step 1 (Vector reward):** For image enhancement, define:

$$\mathbf{r}_t = (r_t^{PSNR}, r_t^{SSIM}, r_t^{LPIPS}, r_t^{aesthetic})$$

**Step 2 (Pareto dominance):** Policy $\pi_1$ Pareto-dominates $\pi_2$ if:

$$J_k(\pi_1) \geq J_k(\pi_2) \quad \forall k \quad \text{and} \quad \exists k: J_k(\pi_1) > J_k(\pi_2)$$

**Step 3 (Weighted scalarization):**

$$R_t = \sum_k w_k r_t^k, \quad \mathbf{w} \in \Delta^{K-1} \text{ (simplex)}$$

**Step 4 (Adaptive weight selection):** The weights should adapt based on current metric values. Normalize each metric to $[0,1]$ and use:

$$w_k \propto \exp(-\beta \cdot J_k^{current} / J_k^{target})$$

Metrics far from their target get higher weight.

**Step 5 (Constraint-based multi-objective):** Alternatively, optimize one metric subject to constraints on others:

$$\max_\pi J_{PSNR}(\pi) \quad \text{s.t.} \quad J_{SSIM}(\pi) \geq \tau_{SSIM}, \quad J_{LPIPS}(\pi) \leq \tau_{LPIPS}$$

### 1.3 Actor-Critic Architecture for Images

**Step 1 (Shared visual backbone):** Process the image through a CNN:

$$\mathbf{h} = f_{backbone}(I_t; \theta_{shared}) \in \mathbb{R}^d$$

**Step 2 (Actor head):** Maps features to action distribution:

$$\mu(s_t) = f_{actor}(\mathbf{h}; \theta_\mu) \in \mathbb{R}^{|\mathcal{A}|}$$
$$\sigma(s_t) = \text{softplus}(f_{std}(\mathbf{h}; \theta_\sigma))$$
$$\pi_\theta(a|s) = \mathcal{N}(a; \mu(s), \text{diag}(\sigma(s)^2))$$

**Step 3 (Critic head):** Maps features to value estimate:

$$V_\phi(s_t) = f_{critic}(\mathbf{h}; \theta_V) \in \mathbb{R}$$

**Step 4 (Why shared backbone?):**
- Parameter efficiency: $|\theta_{shared}| \gg |\theta_{actor}| + |\theta_{critic}|$
- Feature reuse: visual features useful for both policy and value estimation
- Gradient sharing: critic gradients improve visual features used by actor

**Step 5 (Gradient derivation for actor):**

$$\nabla_{\theta_\mu} L^{CLIP} = E_t\left[\nabla_{\theta_\mu}\log\pi_\theta(a_t|s_t) \cdot \text{clip\_grad}(r_t, A_t)\right]$$

For Gaussian policy:

$$\nabla_{\theta_\mu}\log\pi_\theta(a|s) = \frac{(a - \mu(s))}{\sigma(s)^2}\nabla_{\theta_\mu}\mu(s)$$

### 1.4 Statistical Significance Testing (t-test Derivation)

**Step 1 (Paired t-test for comparing methods):** Given $n$ image pairs, let $d_i = \text{metric}(\text{method}_A, I_i) - \text{metric}(\text{method}_B, I_i)$.

**Step 2 (Test statistic):**

$$t = \frac{\bar{d}}{s_d / \sqrt{n}}$$

where $\bar{d} = \frac{1}{n}\sum_i d_i$ and $s_d = \sqrt{\frac{1}{n-1}\sum_i(d_i - \bar{d})^2}$.

**Step 3 (Distribution under $H_0: \mu_d = 0$):**

$$t \sim t_{n-1} \quad \text{(Student's t-distribution with } n-1 \text{ degrees of freedom)}$$

**Step 4 (p-value):**

$$p = 2 \cdot P(T_{n-1} > |t|)$$

Reject $H_0$ if $p < \alpha$ (typically $\alpha = 0.05$).

**Step 5 (Confidence interval for improvement):**

$$\bar{d} \pm t_{\alpha/2, n-1} \cdot \frac{s_d}{\sqrt{n}}$$

**Step 6 (Effect size — Cohen's d):**

$$d_{Cohen} = \frac{\bar{d}}{s_d}$$

Interpretation: $|d| < 0.2$ (small), $0.2-0.8$ (medium), $> 0.8$ (large).

---

## 2. Algorithm / Method

### 2.1 Pseudocode

```
Algorithm: PPO Autonomous Image Editor
─────────────────────────────────────────
Input: Image dataset, quality targets
Output: Trained editing policy π*

Initialize: Actor π_θ, Critic V_φ (shared backbone)

for iteration = 1 to N do
    // Collect trajectories
    for episode = 1 to B do
        I₀ ~ dataset
        for t = 0 to T do
            aₜ ~ π_θ(·|sₜ)   // Sample edit action
            Iₜ₊₁ ← apply_edit(Iₜ, aₜ)
            rₜ ← w₁·ΔPSNR + w₂·ΔSSIM + w₃·Δaesthetic
            Store (sₜ, aₜ, rₜ, V_φ(sₜ))
        end for
    end for
    
    // Compute GAE advantages
    Â_t = Σₗ (γλ)ˡ δₜ₊ₗ
    
    // PPO update (K epochs)
    for epoch = 1 to K do
        for minibatch in buffer do
            rₜ(θ) = π_θ(aₜ|sₜ) / π_old(aₜ|sₜ)
            L_clip = min(rₜÂₜ, clip(rₜ, 1±ε)Âₜ)
            L_value = (V_φ(sₜ) - V_target)²
            L = L_clip - c₁·L_value + c₂·entropy
            Update θ, φ
        end for
    end for
end for
```

### 2.2 Complexity Analysis

- **Image encoding (backbone):** $O(HW \cdot C^2 K^2 L)$
- **Actor forward pass:** $O(d \cdot |\mathcal{A}|)$
- **Critic forward pass:** $O(d)$
- **GAE computation:** $O(T)$ per trajectory
- **PPO update (per epoch):** $O(B \cdot T \cdot |\theta|)$
- **Total per iteration:** $O(BT(HWC^2K^2L + |\theta|) + K \cdot BT|\theta|)$

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

- **State:** Current image features + editing history + quality metric vector
- **Action:** Continuous editing parameters (brightness, contrast, saturation, sharpness, color balance)
- **Reward:** Multi-metric improvement: $r_t = \mathbf{w}^T\Delta\mathbf{m}_t$ where $\mathbf{m}$ is metric vector
- **Transition:** Deterministic image transformation
- **PPO specifics:** Clipping $\epsilon = 0.2$, GAE $\lambda = 0.95$, $K = 10$ epochs

### 3.2 Why RL (PPO)?

1. **Stable training:** PPO's clipping prevents catastrophic policy updates
2. **Sample efficiency:** Multiple epochs over collected data maximize sample utility
3. **Continuous actions:** Gaussian policy naturally handles continuous editing parameters
4. **Multi-metric reward:** PPO handles complex reward signals without modification

---

## 4. Dataset

| Dataset | Size | Type | Description |
|---------|------|------|-------------|
| MIT-Adobe FiveK | 5,000 | Expert retouched | Professional edits |
| PPR10K | 11,161 | Portrait retouching | Three expert versions |
| LOL | 500 pairs | Low-light | Enhancement pairs |
| DPED | 22,000 | Phone-DSLR | Cross-device |

---

## 5. Key Equations Summary

| Equation | Description |
|----------|-------------|
| $L^{CLIP} = E[\min(r_t A_t, \text{clip}(r_t)A_t)]$ | PPO clipped objective |
| $\hat{A}_t^{GAE} = \sum_l(\gamma\lambda)^l\delta_{t+l}$ | Generalized Advantage Estimation |
| $r_t(\theta) = \pi_\theta(a_t|s_t)/\pi_{old}(a_t|s_t)$ | Importance ratio |
| $t = \bar{d}/(s_d/\sqrt{n})$ | Paired t-test statistic |
| $R_t = \sum_k w_k r_t^k$ | Multi-metric reward |

---

## 6. References

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.
2. Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). High-dimensional continuous control using generalized advantage estimation. *ICLR*.
3. Park, J., Lee, J.-Y., Yoo, D., & Kweon, I. S. (2018). Distort-and-recover: Color enhancement using deep reinforcement learning. *CVPR*.
4. Hu, Y., He, H., Xu, C., Wang, B., & Lin, S. (2018). Exposure: A white-box photo post-processing framework. *ACM TOG*, 37(2).
