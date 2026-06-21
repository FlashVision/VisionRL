![Module Logo](../logo.png)

# HDR Tone Mapping via Reinforcement Learning

## Overview

This module addresses High Dynamic Range (HDR) tone mapping through reinforcement learning, where an agent learns to compress the luminance range of HDR images for display on standard dynamic range devices. The mathematical framework covers tone mapping operator theory, local adaptation models, and bilateral filtering, with the RL agent learning to select optimal local tone curve parameters for each image region.

## Prerequisites

- Photometry (luminance, radiance, exposure)
- Differential equations (PDEs, diffusion equations)
- Signal processing (bilateral filtering, scale-space theory)
- Psychophysics (Weber-Fechner law, visual adaptation)
- Reinforcement learning (actor-critic methods)

---

## 1. Mathematical Foundations

### 1.1 Tone Mapping Operator Theory

**Definition:** A tone mapping operator (TMO) is a function $f: \mathbb{R}^+ \to [0, 1]$ that maps scene luminance $L_w$ (world luminance) to display luminance $L_d$:

$$L_d = f(L_w)$$

**Requirements for a valid TMO:**
1. Monotonicity: $L_{w_1} > L_{w_2} \implies f(L_{w_1}) \geq f(L_{w_2})$
2. Boundedness: $f(L_w) \in [0, 1]$ for all $L_w \geq 0$
3. Normalization: $f(0) = 0$ (or approaches 0)

**Dynamic range compression ratio:**

$$\rho = \frac{\log(L_{w,max}/L_{w,min})}{\log(L_{d,max}/L_{d,min})}$$

Typical HDR scenes: $\rho \in [10^4, 10^8]$; displays: $\sim 10^2$ to $10^3$.

### 1.2 Reinhard Global Operator Derivation

**Step 1:** Compute the log-average luminance (key value) of the scene:

$$\bar{L}_w = \exp\left(\frac{1}{N}\sum_{x,y} \log(\delta + L_w(x,y))\right)$$

where $\delta$ is a small constant to avoid $\log(0)$.

**Step 2:** Scale luminances to the "key" of the scene using parameter $a$ (typically $a = 0.18$):

$$L_m(x,y) = \frac{a}{\bar{L}_w} L_w(x,y)$$

**Step 3:** Apply the basic Reinhard compression (inspired by photographic film response):

$$L_d(x,y) = \frac{L_m(x,y)}{1 + L_m(x,y)}$$

**Step 4:** Verify properties:
- As $L_m \to 0$: $L_d \approx L_m$ (linear for dark regions)
- As $L_m \to \infty$: $L_d \to 1$ (asymptotic saturation)
- Derivative: $\frac{dL_d}{dL_m} = \frac{1}{(1+L_m)^2} > 0$ (monotone)

**Step 5:** Extended operator with white point burn-out:

$$L_d(x,y) = \frac{L_m(x,y)\left(1 + \frac{L_m(x,y)}{L_{white}^2}\right)}{1 + L_m(x,y)}$$

where $L_{white}$ is the smallest luminance mapped to pure white.

**Step 6:** Verify burn-out: when $L_m = L_{white}$:

$$L_d = \frac{L_{white}(1 + 1)}{1 + L_{white}} = \frac{2L_{white}}{1 + L_{white}} \approx 2 \text{ (for large } L_{white}\text{)}$$

Setting $L_{white} = L_{max}$ preserves all detail.

### 1.3 Local Adaptation Operator

**Step 1:** Replace the global $\bar{L}_w$ with a local adaptation luminance $V(x,y,s)$:

$$L_d(x,y) = \frac{L_m(x,y)}{1 + V(x,y,s)}$$

**Step 2:** $V(x,y,s)$ is computed at an appropriate scale $s$ using Gaussian convolutions:

$$V_s(x,y) = L_m(x,y) * G_s(x,y)$$

where $G_s$ is a Gaussian kernel with standard deviation $s$.

**Step 3:** Select the largest scale $s$ that avoids halos (gradient reversals):

$$\frac{|V_{s_i}(x,y) - V_{s_{i+1}}(x,y)|}{2^{\phi} a / s_i^2 + V_{s_i}(x,y)} < \epsilon$$

**Step 4:** This finds the coarsest local average that is still "consistent" with the original luminance structure.

### 1.4 Bilateral Filter for Local Adaptation

**Definition:** The bilateral filter separates the image into base layer (large-scale) and detail layer:

$$BF[I](x) = \frac{1}{W_x}\sum_{y \in \Omega} G_{\sigma_s}(\|x - y\|) \cdot G_{\sigma_r}(|I(x) - I(y)|) \cdot I(y)$$

where $W_x = \sum_y G_{\sigma_s}(\|x-y\|) \cdot G_{\sigma_r}(|I(x)-I(y)|)$.

**Step 1:** Decompose into spatial and range kernels:

$$w(x, y) = \underbrace{\exp\left(-\frac{\|x-y\|^2}{2\sigma_s^2}\right)}_{\text{spatial proximity}} \cdot \underbrace{\exp\left(-\frac{(I(x)-I(y))^2}{2\sigma_r^2}\right)}_{\text{intensity similarity}}$$

**Step 2:** For tone mapping, apply bilateral filter to log-luminance:

$$\text{base}(x,y) = BF[\log L_w](x,y)$$
$$\text{detail}(x,y) = \log L_w(x,y) - \text{base}(x,y)$$

**Step 3:** Compress only the base layer:

$$\text{base}'(x,y) = \gamma \cdot (\text{base}(x,y) - \max(\text{base}))$$

where $\gamma = \frac{\log(\text{target contrast})}{\max(\text{base}) - \min(\text{base})}$.

**Step 4:** Recombine:

$$\log L_d(x,y) = \text{base}'(x,y) + \text{detail}(x,y) + \text{offset}$$

**Step 5:** The bilateral filter's edge-preserving property ensures that $\text{detail}$ captures local contrast without large-scale illumination variation, preventing halo artifacts.

### 1.5 Proof: Reinhard Operator Maps $[0, \infty) \to [0, 1)$

**Theorem:** $f(L) = L/(1+L)$ maps $\mathbb{R}_{\geq 0}$ bijectively to $[0, 1)$.

**Proof:**
- $f(0) = 0$ ✓
- $\lim_{L\to\infty} f(L) = \lim_{L\to\infty} \frac{1}{1/L + 1} = 1$ (approached but never reached) ✓
- $f'(L) = 1/(1+L)^2 > 0$ for all $L \geq 0$ (strictly monotone) ✓
- Inverse: $L = f^{-1}(L_d) = L_d/(1-L_d)$, defined for $L_d \in [0,1)$ ✓

Therefore $f$ is a bijection from $[0,\infty)$ to $[0,1)$. $\blacksquare$

---

## 2. Algorithm / Method

### 2.1 Pseudocode

```
Algorithm: RL HDR Tone Mapping Agent
─────────────────────────────────────────
Input: HDR image L_w(x,y), display parameters
Output: Tone-mapped LDR image L_d(x,y)

State: s = (luminance_histogram, log_avg_luminance, 
            local_contrast_map, dynamic_range_stats)
Actions: a = (key_value_a, white_point, local_scale_s, 
              bilateral_σs, bilateral_σr, detail_boost)

Initialize policy π_θ, value function V_φ

for episode = 1 to M do
    Randomly sample HDR image
    s₀ ← compute_HDR_features(L_w)
    for t = 0 to T_max do
        aₜ ~ π_θ(·|sₜ)
        L_d^(t) ← tone_map(L_w, aₜ)
        rₜ ← TMQI(L_d^(t)) + λ·naturalness_score
        s_{t+1} ← compute_features(L_d^(t))
    end for
    Update θ, φ via PPO
end for
```

### 2.2 Complexity Analysis

- **Log-average luminance:** $O(N)$ where $N = H \times W$
- **Bilateral filter:** $O(N \cdot k^2)$ naively, $O(N)$ with fast approximation
- **Gaussian pyramid (local adaptation):** $O(N \cdot S)$ for $S$ scales
- **Detail recombination:** $O(N)$
- **Total tone mapping:** $O(N \cdot k^2)$ dominated by bilateral filter

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

- **State:** HDR statistics — dynamic range, histogram shape, local contrast distribution, spatial structure
- **Action:** TMO parameters — key value $a \in [0.09, 0.72]$, white point $L_{white}$, local adaptation scale $s$, bilateral filter parameters $(\sigma_s, \sigma_r)$, detail boost factor
- **Reward:** Tone Mapping Quality Index (TMQI) + perceptual naturalness + detail preservation
- **Transition:** Deterministic (parameter selection → tone-mapped result)

### 3.2 Why RL?

1. **Scene-dependent parameters:** Optimal TMO parameters vary dramatically across scenes (sunsets, interiors, night scenes)
2. **Multi-criterion optimization:** Simultaneously preserve detail, avoid artifacts, maintain natural appearance
3. **No single optimal operator:** The "best" TMO is subjective; RL can learn from diverse human preferences
4. **Local vs. global trade-off:** Agent learns when to use local adaptation vs. global compression

---

## 4. Dataset

| Dataset | Size | Dynamic Range | Description |
|---------|------|--------------|-------------|
| Fairchild HDR | 106 images | Up to 10⁸:1 | Diverse HDR scenes |
| HDR-Eye | 46 images | Variable | With eye-tracking data |
| HDRI Haven | 500+ HDRIs | Variable | Environment maps |
| Funt HDR | 105 images | Up to 10⁶:1 | Natural scenes |

---

## 5. Key Equations Summary

| Equation | Description |
|----------|-------------|
| $L_d = \frac{L_m}{1 + L_m}$ | Reinhard global operator |
| $L_d = \frac{L_m(1 + L_m/L_w^2)}{1 + L_m}$ | Extended Reinhard with burn-out |
| $\bar{L}_w = \exp(\frac{1}{N}\sum\log(\delta + L_w))$ | Log-average luminance |
| $BF[I](x) = \frac{1}{W}\sum_y G_{\sigma_s}G_{\sigma_r} I(y)$ | Bilateral filter |
| $\log L_d = \gamma(\text{base} - \max) + \text{detail}$ | Bilateral tone mapping |

---

## 6. References

1. Reinhard, E., Stark, M., Shirley, P., & Ferwerda, J. (2002). Photographic tone reproduction for digital images. *ACM TOG (SIGGRAPH)*, 21(3), 267-276.
2. Durand, F., & Dorsey, J. (2002). Fast bilateral filtering for the display of high-dynamic-range images. *ACM TOG (SIGGRAPH)*, 21(3), 257-266.
3. Yeganeh, H., & Wang, Z. (2013). Objective quality assessment of tone-mapped images. *IEEE TIP*, 22(2), 657-667.
4. Tomasi, C., & Manduchi, R. (1998). Bilateral filtering for gray and color images. *ICCV*.
5. Debevec, P. E., & Malik, J. (2008). Recovering high dynamic range radiance maps from photographs. *ACM SIGGRAPH 2008 classes*.
