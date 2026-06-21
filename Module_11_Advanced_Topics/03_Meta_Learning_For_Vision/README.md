![Module Logo](../logo.png)

# Meta-Learning for Vision with Reinforcement Learning

## Overview

This module combines meta-learning with reinforcement learning for rapid adaptation to new visual tasks. The mathematical framework provides complete derivations of Model-Agnostic Meta-Learning (MAML), including inner/outer loop gradient computation, Hessian analysis, the first-order approximation (FOMAML), and connects to few-shot learning through PAC-Bayes generalization bounds.

## Prerequisites

- Calculus (second-order derivatives, Hessian matrices, Taylor expansion)
- Optimization (gradient descent, implicit differentiation)
- Statistical learning theory (PAC learning, generalization bounds)
- Few-shot learning (support/query sets, episode training)
- Reinforcement learning (policy gradient, meta-RL)

---

## 1. Mathematical Foundations

### 1.1 MAML: Meta-Gradient Computation

**Problem setup:** Given a distribution of tasks $p(\mathcal{T})$, find initialization $\theta$ that enables fast adaptation to any new task.

**Step 1 (Inner loop — task adaptation):** For task $\mathcal{T}_i$ with loss $\mathcal{L}_{\mathcal{T}_i}$, take one (or few) gradient steps:

$$\theta_i' = \theta - \alpha\nabla_\theta\mathcal{L}_{\mathcal{T}_i}(f_\theta)$$

where $\alpha$ is the inner learning rate and $f_\theta$ is the model.

**Step 2 (Outer loop — meta-optimization):** Optimize the initialization across tasks:

$$\theta \leftarrow \theta - \beta\nabla_\theta\sum_{i=1}^B \mathcal{L}_{\mathcal{T}_i}(f_{\theta_i'})$$

where $\beta$ is the meta learning rate and $B$ is the number of tasks in a meta-batch.

**Step 3 (Expand the meta-gradient):** The key challenge is computing $\nabla_\theta\mathcal{L}_{\mathcal{T}_i}(f_{\theta_i'})$. Apply the chain rule:

$$\nabla_\theta\mathcal{L}_{\mathcal{T}_i}(f_{\theta_i'}) = \nabla_{\theta_i'}\mathcal{L}_{\mathcal{T}_i}(f_{\theta_i'}) \cdot \frac{\partial\theta_i'}{\partial\theta}$$

**Step 4 (Jacobian of the inner update):**

$$\frac{\partial\theta_i'}{\partial\theta} = \frac{\partial}{\partial\theta}\left[\theta - \alpha\nabla_\theta\mathcal{L}_{\mathcal{T}_i}(f_\theta)\right] = I - \alpha\nabla_\theta^2\mathcal{L}_{\mathcal{T}_i}(f_\theta)$$

where $\nabla_\theta^2\mathcal{L}_{\mathcal{T}_i}$ is the Hessian matrix.

**Step 5 (Full meta-gradient):**

$$\nabla_\theta\mathcal{L}_{\mathcal{T}_i}(f_{\theta_i'}) = \left(I - \alpha H_{\mathcal{T}_i}\right)^T \nabla_{\theta_i'}\mathcal{L}_{\mathcal{T}_i}(f_{\theta_i'})$$

where $H_{\mathcal{T}_i} = \nabla_\theta^2\mathcal{L}_{\mathcal{T}_i}(f_\theta) \in \mathbb{R}^{|\theta|\times|\theta|}$.

**Step 6 (Multi-step inner loop):** For $K$ inner steps:

$$\theta_i^{(k)} = \theta_i^{(k-1)} - \alpha\nabla_{\theta_i^{(k-1)}}\mathcal{L}_{\mathcal{T}_i}$$

The Jacobian becomes:

$$\frac{\partial\theta_i^{(K)}}{\partial\theta} = \prod_{k=0}^{K-1}\left(I - \alpha H_{\mathcal{T}_i}^{(k)}\right)$$

### 1.2 Hessian Computation and FOMAML

**Step 1 (Hessian-vector product):** Computing the full Hessian $H \in \mathbb{R}^{|\theta|\times|\theta|}$ is prohibitive for large models. Instead, compute Hessian-vector products:

$$H\mathbf{v} = \nabla_\theta(\nabla_\theta\mathcal{L}^T\mathbf{v})$$

This requires only one additional backward pass.

**Step 2 (FOMAML — First-Order approximation):** Drop the Hessian term entirely:

$$\frac{\partial\theta_i'}{\partial\theta} \approx I$$

giving the simplified meta-gradient:

$$\nabla_\theta^{FOMAML}\mathcal{L}_{\mathcal{T}_i}(f_{\theta_i'}) \approx \nabla_{\theta_i'}\mathcal{L}_{\mathcal{T}_i}(f_{\theta_i'})$$

**Step 3 (Justification):** FOMAML works well because:
- Near initialization, $\alpha H \ll I$ so $(I - \alpha H)^T \approx I$
- The direction of the gradient matters more than its magnitude
- Empirically, FOMAML performs within 1-2% of full MAML

**Step 4 (Reptile — another first-order method):**

$$\theta \leftarrow \theta + \epsilon(\theta_i^{(K)} - \theta)$$

This can be shown to approximate:

$$\nabla_\theta^{Reptile} \approx \nabla_\theta\mathcal{L}_{\mathcal{T}_i} + \frac{\alpha(K-1)}{2}\nabla_\theta\|\nabla_\theta\mathcal{L}_{\mathcal{T}_i}\|^2 + O(\alpha^2 K^2)$$

The second term encourages low curvature, improving generalization.

### 1.3 Meta-RL for Vision Tasks

**Step 1:** In meta-RL, the inner loop performs RL within a task (vision episode):

$$\theta_i' = \theta + \alpha\nabla_\theta E_{\pi_\theta}\left[\sum_t r_t^{(\mathcal{T}_i)}\right]$$

**Step 2:** The outer loop optimizes for fast adaptation across visual tasks:

$$\theta \leftarrow \theta + \beta\nabla_\theta\sum_i E_{\pi_{\theta_i'}}\left[\sum_t r_t^{(\mathcal{T}_i)}\right]$$

**Step 3:** The meta-policy gradient (using REINFORCE):

$$\nabla_\theta J_{meta} = E_\mathcal{T}\left[E_{\tau\sim\pi_{\theta'}}\left[\sum_t\nabla_\theta\log\pi_{\theta'}(a_t|s_t) \cdot R_t \cdot \frac{\partial\theta'}{\partial\theta}\right]\right]$$

### 1.4 PAC-Bayes Bound for Few-Shot Learning

**Theorem (PAC-Bayes):** For any prior $P$ over hypotheses (chosen independently of the data), and any posterior $Q$, with probability $\geq 1-\delta$ over training sets of size $m$:

$$E_{h\sim Q}[\mathcal{L}(h)] \leq E_{h\sim Q}[\hat{\mathcal{L}}(h)] + \sqrt{\frac{\text{KL}(Q\|P) + \log(m/\delta)}{2m}}$$

**Application to MAML:**

**Step 1:** The prior $P$ is centered at the meta-learned initialization $\theta$: $P = \mathcal{N}(\theta, \lambda^{-1}I)$.

**Step 2:** The posterior after $K$ gradient steps: $Q = \mathcal{N}(\theta_i', \sigma^2 I)$ where $\theta_i' = \theta - \alpha\nabla\mathcal{L}$.

**Step 3:** The KL divergence:

$$\text{KL}(Q\|P) = \frac{\lambda}{2}\|\theta_i' - \theta\|^2 + \frac{d}{2}\left(\lambda\sigma^2 - 1 - \log(\lambda\sigma^2)\right)$$

**Step 4:** For few-shot ($m$ small), minimizing KL means staying close to $\theta$ — exactly what MAML's few inner steps achieve.

**Step 5 (Generalization guarantee):** With $N$-way $K$-shot (support size $m = NK$):

$$\text{Test error} \leq \text{Train error} + O\left(\sqrt{\frac{\|\theta'-\theta\|^2 + d}{NK}}\right)$$

This explains why MAML with few inner steps generalizes: small $\|\theta'-\theta\|$ keeps the bound tight. $\blacksquare$

---

## 2. Algorithm / Method

### 2.1 Pseudocode

```
Algorithm: MAML for Few-Shot Visual Classification
────────────────────────────────────────────────────
Input: Task distribution p(T), inner LR α, meta LR β
Output: Meta-learned initialization θ

Initialize θ randomly

for meta_iteration = 1 to M do
    Sample batch of tasks {T₁, ..., T_B} ~ p(T)
    
    for each task Tᵢ do
        // Inner loop: fast adaptation
        Sample support set Dᵢˢ (K shots per class)
        Sample query set Dᵢᵠ
        
        θᵢ' ← θ - α∇θ L(fθ, Dᵢˢ)  // One gradient step
        
        // Evaluate on query set
        Lᵢ ← L(f_{θᵢ'}, Dᵢᵠ)
    end for
    
    // Outer loop: meta-update
    θ ← θ - β∇θ Σᵢ Lᵢ(f_{θᵢ'})
    // Requires second-order gradient (Hessian)
    // Or use FOMAML: θ ← θ - β Σᵢ ∇_{θᵢ'} Lᵢ
end for
```

### 2.2 Complexity Analysis

- **Inner loop (per task):** $O(K \cdot N \cdot |\theta|)$ for $K$-shot $N$-way
- **Full MAML meta-gradient:** $O(|\theta|^2)$ for Hessian-vector product
- **FOMAML meta-gradient:** $O(|\theta|)$ (same as standard gradient)
- **Per meta-iteration:** $O(B \cdot (K \cdot N + 1) \cdot |\theta|^2)$ (full MAML)
- **Memory:** $O(B \cdot |\theta|)$ for storing task-specific parameters

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

$$\mathcal{M}_{meta} = (\mathcal{S}_{meta}, \mathcal{A}_{meta}, P_{meta}, R_{meta}, \gamma)$$

- **Meta-state:** Current model parameters $\theta$ + task context
- **Meta-action:** Gradient step direction and magnitude
- **Meta-reward:** Performance on query set after adaptation
- **Inner MDP:** Standard vision task (classification, detection, segmentation)

### 3.2 Why Meta-RL for Vision?

1. **Few-shot adaptation:** New visual concepts (objects, textures) appear constantly; meta-learning enables rapid adaptation from few examples
2. **Task distribution:** Visual tasks share low-level features; MAML learns transferable representations
3. **Efficient exploration:** Meta-learned initialization provides informed starting point for RL exploration
4. **Curriculum learning:** The meta-learning process naturally sequences tasks of increasing difficulty

---

## 4. Dataset

| Dataset | Classes | Shots | Type | Description |
|---------|---------|-------|------|-------------|
| Mini-ImageNet | 100 | 1-5 | Classification | 64/16/20 split |
| Omniglot | 1,623 | 1-20 | Characters | Handwritten alphabets |
| CIFAR-FS | 100 | 1-5 | Classification | CIFAR-100 based |
| Meta-Dataset | 10 datasets | Variable | Multi-domain | Large-scale meta-learning |

---

## 5. Key Equations Summary

| Equation | Description |
|----------|-------------|
| $\theta_i' = \theta - \alpha\nabla_\theta\mathcal{L}_{\mathcal{T}_i}(f_\theta)$ | Inner loop adaptation |
| $\theta \leftarrow \theta - \beta\nabla_\theta\sum_i\mathcal{L}_{\mathcal{T}_i}(f_{\theta_i'})$ | Outer loop meta-update |
| $\frac{\partial\theta_i'}{\partial\theta} = I - \alpha H_{\mathcal{T}_i}$ | Inner loop Jacobian |
| $\nabla^{FOMAML} \approx \nabla_{\theta_i'}\mathcal{L}(f_{\theta_i'})$ | First-order approximation |
| $E_Q[\mathcal{L}] \leq E_Q[\hat{\mathcal{L}}] + \sqrt{\frac{KL(Q\|P)+\log(m/\delta)}{2m}}$ | PAC-Bayes bound |

---

## 6. References

1. Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks. *ICML*.
2. Nichol, A., Achiam, J., & Schulman, J. (2018). On first-order meta-learning algorithms. *arXiv:1803.02999*.
3. Amit, R., & Meir, R. (2018). Meta-learning by adjusting priors based on extended PAC-Bayes theory. *ICML*.
4. Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical networks for few-shot learning. *NeurIPS*.
5. Rothfuss, J., et al. (2019). ProMP: Proximal meta-policy search. *ICLR*.
