![Module Logo](../logo.png)

# Text-to-Image Generation with RLHF

## Overview

This module applies Reinforcement Learning from Human Feedback (RLHF) to text-to-image generation, aligning image generation models with human preferences. The mathematical framework derives the CLIP contrastive loss, the Bradley-Terry preference model for reward learning, the complete RLHF pipeline, and Direct Preference Optimization (DPO) as a closed-form alternative that eliminates the need for explicit reward modeling.

## Prerequisites

- Contrastive learning (InfoNCE, temperature scaling)
- Preference learning (pairwise comparisons, Elo ratings)
- Optimization (KL-constrained optimization, Lagrangian methods)
- Probability theory (logistic models, maximum likelihood)
- Reinforcement learning (PPO, KL regularization, reward modeling)

---

## 1. Mathematical Foundations

### 1.1 CLIP Loss Derivation (Contrastive Learning)

**Step 1:** Given a batch of $N$ image-text pairs $\{(I_i, T_i)\}_{i=1}^N$, encode:

$$\mathbf{v}_i = f_{image}(I_i) \in \mathbb{R}^d, \quad \mathbf{t}_i = f_{text}(T_i) \in \mathbb{R}^d$$

Normalize: $\hat{\mathbf{v}}_i = \mathbf{v}_i/\|\mathbf{v}_i\|$, $\hat{\mathbf{t}}_i = \mathbf{t}_i/\|\mathbf{t}_i\|$.

**Step 2:** Compute similarity matrix:

$$s_{ij} = \hat{\mathbf{v}}_i^T \hat{\mathbf{t}}_j / \tau$$

where $\tau$ is a learnable temperature parameter.

**Step 3:** Image-to-text contrastive loss (InfoNCE):

$$\mathcal{L}_{i \to t} = -\frac{1}{N}\sum_{i=1}^N \log\frac{\exp(s_{ii})}{\sum_{j=1}^N \exp(s_{ij})}$$

**Step 4:** Text-to-image contrastive loss:

$$\mathcal{L}_{t \to i} = -\frac{1}{N}\sum_{i=1}^N \log\frac{\exp(s_{ii})}{\sum_{j=1}^N \exp(s_{ji})}$$

**Step 5:** Total CLIP loss:

$$\mathcal{L}_{CLIP} = \frac{1}{2}(\mathcal{L}_{i \to t} + \mathcal{L}_{t \to i})$$

**Step 6 (Connection to mutual information):** The InfoNCE loss provides a lower bound on mutual information:

$$I(V; T) \geq \log N - \mathcal{L}_{i \to t}$$

As $N \to \infty$, the bound becomes tight and minimizing $\mathcal{L}_{CLIP}$ maximizes $I(V; T)$.

**Step 7 (Temperature role):** $\tau$ controls the concentration of the softmax:
- Small $\tau$: sharp distribution, focuses on hard negatives
- Large $\tau$: uniform distribution, easy optimization but less discriminative

### 1.2 Bradley-Terry Preference Model

**Step 1:** Given two images $x_w$ (preferred/winner) and $x_l$ (rejected/loser) for prompt $c$, the Bradley-Terry model defines:

$$P(x_w \succ x_l | c) = \sigma(r(c, x_w) - r(c, x_l))$$

where $\sigma(z) = 1/(1+e^{-z})$ is the sigmoid function and $r(c, x)$ is the reward model.

**Step 2:** The reward model is trained via maximum likelihood on human preference data $\mathcal{D} = \{(c_i, x_w^i, x_l^i)\}$:

$$\mathcal{L}_{RM} = -E_{(c,x_w,x_l)\sim\mathcal{D}}\left[\log\sigma(r_\phi(c, x_w) - r_\phi(c, x_l))\right]$$

**Step 3 (Derivation from Plackett-Luce):** The Bradley-Terry model is a special case of Plackett-Luce for pairwise comparisons:

$$P(\text{ranking } \pi) = \prod_{i=1}^K \frac{\exp(r(x_{\pi(i)}))}{\sum_{j=i}^K \exp(r(x_{\pi(j)}))}$$

For $K=2$:

$$P(x_w \succ x_l) = \frac{\exp(r(x_w))}{\exp(r(x_w)) + \exp(r(x_l))} = \sigma(r(x_w) - r(x_l))$$

**Step 4 (Identifiability):** The reward is only defined up to a constant: $r'(x) = r(x) + c$ gives the same preferences. Typically normalized by fixing $E_x[r(x)] = 0$.

### 1.3 RLHF Pipeline for Text-to-Image

**Step 1 (KL-constrained RL objective):** Given a reward model $r_\phi$ and reference policy $\pi_{ref}$ (the pre-trained model):

$$\max_{\pi_\theta} E_{c\sim\mathcal{D}, x\sim\pi_\theta(\cdot|c)}\left[r_\phi(c, x)\right] - \beta\text{KL}(\pi_\theta(\cdot|c) \| \pi_{ref}(\cdot|c))$$

**Step 2:** The KL term prevents the policy from collapsing to a degenerate solution that exploits reward model inaccuracies.

**Step 3 (Closed-form optimal policy):** The solution to the KL-constrained problem is:

$$\pi^*(x|c) = \frac{1}{Z(c)}\pi_{ref}(x|c)\exp\left(\frac{r_\phi(c,x)}{\beta}\right)$$

where $Z(c) = \sum_x \pi_{ref}(x|c)\exp(r_\phi(c,x)/\beta)$ is the partition function.

**Step 4 (PPO implementation):** Since computing $\pi^*$ directly is intractable, approximate with PPO:

$$\mathcal{L}_{PPO} = E\left[\min\left(\frac{\pi_\theta(x|c)}{\pi_{old}(x|c)}A, \text{clip}\left(\frac{\pi_\theta}{\pi_{old}}, 1\pm\epsilon\right)A\right)\right]$$

where $A = r_\phi(c, x) - V_\psi(c)$ is the advantage.

### 1.4 Direct Preference Optimization (DPO) Derivation

**Step 1:** Start from the optimal policy of the RLHF objective:

$$\pi^*(x|c) = \frac{1}{Z(c)}\pi_{ref}(x|c)\exp(r(c,x)/\beta)$$

**Step 2:** Solve for the reward:

$$r(c, x) = \beta\log\frac{\pi^*(x|c)}{\pi_{ref}(x|c)} + \beta\log Z(c)$$

**Step 3:** Substitute into the Bradley-Terry preference model:

$$P(x_w \succ x_l | c) = \sigma(r(c,x_w) - r(c,x_l))$$

$$= \sigma\left(\beta\log\frac{\pi^*(x_w|c)}{\pi_{ref}(x_w|c)} - \beta\log\frac{\pi^*(x_l|c)}{\pi_{ref}(x_l|c)}\right)$$

Note: the $\log Z(c)$ terms cancel!

**Step 4 (DPO loss):** Directly optimize the policy to match preferences without a reward model:

$$\mathcal{L}_{DPO}(\theta) = -E_{(c,x_w,x_l)}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(x_w|c)}{\pi_{ref}(x_w|c)} - \beta\log\frac{\pi_\theta(x_l|c)}{\pi_{ref}(x_l|c)}\right)\right]$$

**Step 5 (Gradient):**

$$\nabla_\theta\mathcal{L}_{DPO} = -\beta E\left[\sigma(-\hat{r}_\theta)\left(\nabla_\theta\log\pi_\theta(x_w|c) - \nabla_\theta\log\pi_\theta(x_l|c)\right)\right]$$

where $\hat{r}_\theta = \beta\log\frac{\pi_\theta(x_w|c)}{\pi_{ref}(x_w|c)} - \beta\log\frac{\pi_\theta(x_l|c)}{\pi_{ref}(x_l|c)}$.

**Step 6 (Interpretation):** The gradient increases the probability of $x_w$ and decreases $x_l$, weighted by how much the current policy disagrees with the preference ($\sigma(-\hat{r}_\theta)$ is large when the model thinks $x_l$ is better).

---

## 2. Algorithm / Method

### 2.1 Pseudocode

```
Algorithm: RLHF for Text-to-Image
─────────────────────────────────────────
Phase 1: Train reward model
    Collect preference data {(c, x_w, x_l)} from humans
    Train r_φ via: min -E[log σ(r_φ(c,x_w) - r_φ(c,x_l))]

Phase 2: RL fine-tuning  
    for iteration = 1 to N do
        Sample prompts {c₁,...,cₘ} from dataset
        Generate images: xᵢ ~ π_θ(·|cᵢ)
        Compute rewards: Rᵢ = r_φ(cᵢ, xᵢ)
        Compute KL penalty: KLᵢ = log(π_θ(xᵢ|cᵢ)/π_ref(xᵢ|cᵢ))
        Adjusted reward: R̃ᵢ = Rᵢ - β·KLᵢ
        Update π_θ via PPO with reward R̃
    end for

Alternative: DPO (single phase)
    for iteration = 1 to N do
        Sample (c, x_w, x_l) from preference data
        L = -log σ(β·log(π_θ(x_w|c)/π_ref(x_w|c)) 
                   - β·log(π_θ(x_l|c)/π_ref(x_l|c)))
        Update θ via gradient descent on L
    end for
```

### 2.2 Complexity Analysis

- **Image generation (diffusion):** $O(T \cdot HW \cdot C^2 K^2 L)$ for $T$ denoising steps
- **Reward model evaluation:** $O(HW \cdot C_{RM}^2)$
- **CLIP score:** $O(d^2)$ after encoding
- **PPO update:** $O(|\theta|)$
- **DPO update:** $O(|\theta|)$ (no reward model needed)

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

- **State:** Text prompt $c$ (or denoising trajectory state for diffusion models)
- **Action:** Generated image $x$ (or individual denoising steps)
- **Reward:** Human preference reward model $r_\phi(c, x)$
- **KL constraint:** $\text{KL}(\pi_\theta \| \pi_{ref}) \leq \epsilon$
- **Policy:** The image generation model $\pi_\theta(x|c)$

### 3.2 Why RL?

1. **Human alignment:** RL optimizes for human preferences that are hard to express as differentiable losses
2. **Beyond CLIP:** CLIP score doesn't capture all aspects of image quality; human reward models are richer
3. **Safety and control:** KL constraint prevents degenerate outputs while improving quality
4. **Compositionality:** RL can reward compositional accuracy (correct spatial relationships, counting)

---

## 4. Dataset

| Dataset | Size | Type | Description |
|---------|------|------|-------------|
| LAION-5B | 5B pairs | Image-text | Pre-training data |
| Pick-a-Pic | 500K | Preferences | Human preference pairs |
| HPS v2 | 798K | Preferences | Human preference scores |
| DiffusionDB | 14M | Prompts+images | Stable Diffusion outputs |

---

## 5. Key Equations Summary

| Equation | Description |
|----------|-------------|
| $\mathcal{L}_{CLIP} = -\frac{1}{N}\sum_i\log\frac{e^{s_{ii}}}{\sum_j e^{s_{ij}}}$ | CLIP contrastive loss |
| $P(x_w\succ x_l) = \sigma(r(x_w)-r(x_l))$ | Bradley-Terry model |
| $\pi^*(x|c) \propto \pi_{ref}(x|c)\exp(r(c,x)/\beta)$ | Optimal RLHF policy |
| $\mathcal{L}_{DPO} = -\log\sigma(\beta\log\frac{\pi_\theta(x_w)}{\pi_{ref}(x_w)} - \beta\log\frac{\pi_\theta(x_l)}{\pi_{ref}(x_l)})$ | DPO loss |
| $I(V;T) \geq \log N - \mathcal{L}_{InfoNCE}$ | MI lower bound |

---

## 6. References

1. Radford, A., et al. (2021). Learning transferable visual models from natural language supervision (CLIP). *ICML*.
2. Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS*.
3. Rafailov, R., et al. (2023). Direct preference optimization: Your language model is secretly a reward model. *NeurIPS*.
4. Black, K., et al. (2024). Training diffusion models with reinforcement learning. *ICLR*.
5. Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block designs. *Biometrika*, 39, 324-345.
