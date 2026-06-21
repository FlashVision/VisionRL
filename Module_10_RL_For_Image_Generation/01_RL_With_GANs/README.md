![Module Logo](../logo.png)

# Reinforcement Learning with Generative Adversarial Networks

## Overview

This module explores the deep connection between Reinforcement Learning and Generative Adversarial Networks (GANs), deriving the minimax objective, proving the optimal discriminator, establishing the JS divergence minimization interpretation, and analyzing mode collapse as a failure of exploration. The generator is reinterpreted as an RL policy that learns to produce samples that fool the discriminator.

## Prerequisites

- Probability theory (density functions, divergences, expectations)
- Optimization (minimax games, saddle points, Nash equilibria)
- Information theory (KL divergence, Jensen-Shannon divergence)
- Measure theory (absolute continuity, Radon-Nikodym)
- Reinforcement learning (policy as generator, reward from discriminator)

---

## 1. Mathematical Foundations

### 1.1 GAN Minimax Objective

**Definition:** The GAN training objective is a two-player minimax game:

$$\min_G \max_D V(D, G) = E_{x\sim p_{data}}[\log D(x)] + E_{z\sim p_z}[\log(1 - D(G(z)))]$$

where:
- $G: \mathcal{Z} \to \mathcal{X}$ is the generator mapping noise $z$ to data space
- $D: \mathcal{X} \to [0,1]$ is the discriminator (probability of being real)
- $p_{data}$ is the true data distribution
- $p_z$ is the noise prior (typically $\mathcal{N}(0, I)$)

### 1.2 Optimal Discriminator Proof

**Theorem:** For fixed $G$, the optimal discriminator is:

$$D^*_G(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$$

where $p_g$ is the distribution induced by $G$.

**Proof:**

**Step 1:** Write the value function as an integral:

$$V(D, G) = \int_x p_{data}(x)\log D(x) + p_g(x)\log(1-D(x)) \, dx$$

**Step 2:** For each $x$, maximize the integrand $f(D) = a\log D + b\log(1-D)$ where $a = p_{data}(x)$ and $b = p_g(x)$.

**Step 3:** Take derivative and set to zero:

$$\frac{df}{dD} = \frac{a}{D} - \frac{b}{1-D} = 0$$

**Step 4:** Solve:

$$a(1-D) = bD \implies a - aD = bD \implies D^* = \frac{a}{a+b} = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$$

**Step 5:** Verify it's a maximum: $\frac{d^2f}{dD^2} = -\frac{a}{D^2} - \frac{b}{(1-D)^2} < 0$ for $a, b > 0$. $\blacksquare$

### 1.3 JS Divergence Minimization Proof

**Theorem:** Given optimal $D^*$, the generator minimizes the Jensen-Shannon divergence:

$$\min_G V(D^*_G, G) = -\log 4 + 2\text{JSD}(p_{data} \| p_g)$$

**Proof:**

**Step 1:** Substitute $D^*_G$ into the value function:

$$V(D^*, G) = E_{x\sim p_{data}}\left[\log\frac{p_{data}(x)}{p_{data}(x) + p_g(x)}\right] + E_{x\sim p_g}\left[\log\frac{p_g(x)}{p_{data}(x) + p_g(x)}\right]$$

**Step 2:** Define the mixture $m(x) = \frac{1}{2}(p_{data}(x) + p_g(x))$. Then:

$$V(D^*, G) = E_{p_{data}}\left[\log\frac{p_{data}}{2m}\right] + E_{p_g}\left[\log\frac{p_g}{2m}\right]$$

$$= E_{p_{data}}\left[\log\frac{p_{data}}{m}\right] + E_{p_g}\left[\log\frac{p_g}{m}\right] - 2\log 2$$

**Step 3:** Recognize the KL divergences:

$$= \text{KL}(p_{data}\|m) + \text{KL}(p_g\|m) - \log 4$$

**Step 4:** By definition of Jensen-Shannon divergence:

$$\text{JSD}(p_{data}\|p_g) = \frac{1}{2}\text{KL}(p_{data}\|m) + \frac{1}{2}\text{KL}(p_g\|m)$$

**Step 5:** Therefore:

$$V(D^*, G) = 2\text{JSD}(p_{data}\|p_g) - \log 4$$

**Step 6:** Since $\text{JSD} \geq 0$ with equality iff $p_{data} = p_g$, the global minimum $V^* = -\log 4$ is achieved when $p_g = p_{data}$. $\blacksquare$

### 1.4 RL-GAN Connection: Generator as Policy

**Step 1:** Reinterpret the GAN framework in RL terms:
- **Policy (Generator):** $\pi_\theta(a|s) \equiv G_\theta(z)$ where $s = z$ (noise state), $a = G(z)$ (generated sample)
- **Reward (Discriminator):** $R(a) = \log D(a)$ or $R(a) = -\log(1 - D(a))$
- **Environment:** Single-step MDP (one action per episode)

**Step 2:** The generator's objective under the non-saturating loss:

$$J_G = E_{z\sim p_z}[\log D(G(z))] = E_{z\sim p_z}[R(G(z))]$$

This is exactly the policy gradient objective for a single-step MDP.

**Step 3:** The policy gradient:

$$\nabla_\theta J_G = E_z\left[\nabla_\theta G_\theta(z) \cdot \nabla_x \log D(x)\big|_{x=G_\theta(z)}\right]$$

**Step 4 (Reparameterization vs. REINFORCE):** For continuous generators, we use the reparameterization trick (pathwise gradient), which has lower variance than REINFORCE.

**Step 5:** For discrete outputs (e.g., text GANs), REINFORCE is required:

$$\nabla_\theta J_G = E_{x \sim \pi_\theta}\left[\nabla_\theta \log\pi_\theta(x) \cdot R(x)\right]$$

### 1.5 Mode Collapse Analysis

**Definition:** Mode collapse occurs when $G$ maps all $z$ to a small subset of the data support, producing limited diversity.

**Step 1 (Game-theoretic view):** In the simultaneous gradient update, the generator can "cycle" between modes. For a mixture $p_{data} = \frac{1}{K}\sum_k \mathcal{N}(\mu_k, \sigma^2)$:

**Step 2:** If $D$ focuses on distinguishing mode $k$, $G$ collapses to mode $j \neq k$. Then $D$ adapts, and $G$ switches again.

**Step 3:** The generator's effective objective is:

$$\max_G \min_D V(D, G)$$

but gradient-based alternating optimization doesn't guarantee convergence to the saddle point.

**Step 4 (Sufficient condition for convergence):** If both $V(D, G)$ is convex in $G$ and concave in $D$ (which holds in function space but not parameter space), then gradient descent-ascent converges.

---

## 2. Algorithm / Method

### 2.1 Pseudocode

```
Algorithm: RL-GAN Training
─────────────────────────────────────────
Policy (Generator): G_θ(z) → image
Reward (Discriminator): D_φ(x) → [0,1]

for iteration = 1 to N do
    // Update Discriminator (reward model)
    for k = 1 to n_critic do
        Sample {x₁,...,xₘ} from p_data
        Sample {z₁,...,zₘ} from p_z
        L_D = -1/m Σ[log D_φ(xᵢ) + log(1-D_φ(G_θ(zᵢ)))]
        φ ← φ - η_D ∇φ L_D
    end for
    
    // Update Generator (policy improvement)
    Sample {z₁,...,zₘ} from p_z
    // Non-saturating loss (RL reward maximization):
    L_G = -1/m Σ log D_φ(G_θ(zᵢ))   // Maximize reward
    θ ← θ - η_G ∇θ L_G
end for
```

### 2.2 Complexity Analysis

- **Generator forward pass:** $O(d_z \cdot C^2 \cdot K^2 \cdot L_G)$
- **Discriminator forward pass:** $O(HW \cdot C^2 \cdot K^2 \cdot L_D)$
- **Per iteration:** $O(m \cdot (n_D + 1) \cdot HW \cdot C^2 K^2 L)$
- **Gradient computation:** Same order as forward pass (backprop)
- **Total training:** $O(N \cdot m \cdot HW \cdot C^2 K^2 L)$

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

- **State:** Noise vector $z \sim p_z$ (or conditioning information)
- **Action:** Generated image $G(z)$
- **Reward:** $R(a) = \log D(a)$ (discriminator feedback)
- **Transition:** Single-step (episodic with length 1)
- **Policy:** Generator $G_\theta$ parameterized by neural network

### 3.2 Why RL?

1. **Non-differentiable reward:** When the discriminator provides discrete feedback or the generator outputs discrete tokens, RL is necessary
2. **Reward shaping:** RL framework allows incorporating additional rewards (diversity, quality, constraint satisfaction)
3. **Multi-step generation:** Sequential GANs (e.g., for image completion) naturally fit the RL framework
4. **Exploration-exploitation:** RL techniques (entropy bonus, curiosity) address mode collapse

---

## 4. Dataset

| Dataset | Size | Resolution | Description |
|---------|------|-----------|-------------|
| CelebA-HQ | 30K | 1024×1024 | High-quality faces |
| LSUN | 3M+ | 256×256 | Scenes (bedroom, church) |
| CIFAR-10 | 60K | 32×32 | 10-class objects |
| ImageNet | 1.2M | 256×256 | 1000-class diverse |

---

## 5. Key Equations Summary

| Equation | Description |
|----------|-------------|
| $\min_G\max_D E[\log D(x)] + E[\log(1-D(G(z)))]$ | GAN minimax |
| $D^*(x) = p_{data}(x)/(p_{data}(x)+p_g(x))$ | Optimal discriminator |
| $V(D^*,G) = 2\text{JSD}(p_{data}\|p_g) - \log4$ | JS divergence form |
| $\nabla_\theta J_G = E[\nabla_\theta G_\theta(z)\nabla_x\log D(x)]$ | Generator gradient |
| $\text{JSD}(p\|q) = \frac{1}{2}\text{KL}(p\|m) + \frac{1}{2}\text{KL}(q\|m)$ | Jensen-Shannon divergence |

---

## 6. References

1. Goodfellow, I. J., et al. (2014). Generative adversarial nets. *NeurIPS*.
2. Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein generative adversarial networks. *ICML*.
3. Pfau, D., & Vinyals, O. (2016). Connecting generative adversarial networks and actor-critic methods. *arXiv:1610.01945*.
4. Yu, L., Zhang, W., Wang, J., & Yu, Y. (2017). SeqGAN: Sequence generative adversarial nets with policy gradient. *AAAI*.
