![Module Logo](../logo.png)

# CNN From Scratch

## Overview

Convolutional Neural Networks exploit the spatial structure of images through local connectivity, weight sharing, and translation equivariance. This document derives the convolution output size formula, proves the receptive field calculation by induction, provides a formal proof of translation equivariance, and analyzes parameter sharing efficiency compared to fully connected networks.

## Prerequisites

- Neural network basics (Module 05.1)
- Convolution theory (Module 02.1)
- Linear algebra (tensor operations)

---

## 1. Mathematical Foundations

### 1.1 Core Definition — Convolutional Layer

A 2D convolutional layer with $C_{\text{in}}$ input channels and $C_{\text{out}}$ output channels applies:

$$Y_{j}[m, n] = b_j + \sum_{i=1}^{C_{\text{in}}} \sum_{p=0}^{K-1}\sum_{q=0}^{K-1} W_{j,i}[p, q] \cdot X_i[m \cdot s + p, \; n \cdot s + q]$$

for output channel $j = 1, \ldots, C_{\text{out}}$, where $K$ is the kernel size and $s$ is the stride.

### 1.2 Convolution Output Size Derivation

**Step 1:** Consider a 1D input of size $n$ convolved with a kernel of size $k$, stride $s$, and padding $p$.

**Step 2:** With padding, the effective input size is $n + 2p$.

**Step 3:** The first valid output position has the kernel starting at position 0. The last valid position has the kernel ending at position $n + 2p - 1$, starting at $n + 2p - k$.

**Step 4:** With stride $s$, the valid starting positions are $0, s, 2s, \ldots$. The number of positions is:

$$\left\lfloor\frac{n + 2p - k}{s}\right\rfloor + 1$$

**Step 5:** For 2D convolution with input size $H \times W$:

$$H_{\text{out}} = \left\lfloor\frac{H + 2p_H - K_H}{s_H}\right\rfloor + 1, \quad W_{\text{out}} = \left\lfloor\frac{W + 2p_W - K_W}{s_W}\right\rfloor + 1$$

**Result:**

$$\boxed{O = \left\lfloor\frac{I + 2P - K}{S}\right\rfloor + 1}$$
$\blacksquare$

**Special cases:**
- "Valid" padding ($P=0$): $O = \lfloor(I - K)/S\rfloor + 1$
- "Same" padding ($O = I$ with $S = 1$): requires $P = (K-1)/2$

### 1.3 Receptive Field Calculation — Proof by Induction

**Theorem:** For a network with $L$ convolutional layers, each with kernel size $k_l$ and stride $s_l$, the receptive field size $r_L$ at the output is:

$$r_L = 1 + \sum_{l=1}^{L}(k_l - 1)\prod_{j=1}^{l-1}s_j$$

**Proof by induction:**

**Base case ($L = 1$):** A single layer with kernel size $k_1$ and stride $s_1$:

$$r_1 = 1 + (k_1 - 1) \cdot 1 = k_1 \quad \checkmark$$

Each output pixel depends on exactly $k_1$ input pixels.

**Inductive step:** Assume the formula holds for $L-1$ layers with receptive field $r_{L-1}$.

**Step 1:** Layer $L$ has kernel size $k_L$ and stride $s_L$. Each output pixel of layer $L$ depends on $k_L$ positions in layer $L-1$'s output.

**Step 2:** These $k_L$ positions are separated by stride $s_L$ in layer $L$'s input (which is layer $L-1$'s output).

**Step 3:** Each of these $k_L$ positions has receptive field $r_{L-1}$ in the original input.

**Step 4:** The first and last of the $k_L$ positions span $(k_L - 1) \cdot \prod_{j=1}^{L-1} s_j$ input pixels (the stride products convert layer $L-1$ units to input pixels).

**Step 5:** The total receptive field is:

$$r_L = r_{L-1} + (k_L - 1)\prod_{j=1}^{L-1}s_j$$

**Step 6:** Expanding the recursion:

$$r_L = 1 + \sum_{l=1}^{L}(k_l - 1)\prod_{j=1}^{l-1}s_j$$

where the empty product $\prod_{j=1}^{0}s_j = 1$.
$\blacksquare$

**Example:** Three $3 \times 3$ conv layers with stride 1: $r = 1 + 2 + 2 + 2 = 7$. With stride 2 per layer: $r = 1 + 2 + 2 \cdot 2 + 2 \cdot 4 = 15$.

### 1.4 Proof of Translation Equivariance

**Theorem:** Convolution is translation equivariant: if the input is shifted, the output shifts by the same amount.

**Proof:**

**Step 1:** Let $\mathcal{T}_\tau$ denote translation by $\tau$: $(\mathcal{T}_\tau f)(x) = f(x - \tau)$.

**Step 2:** Compute the convolution of the translated input:

$$((\mathcal{T}_\tau f) * h)(x) = \sum_k h(k) f(x - k - \tau) = \sum_k h(k) f((x - \tau) - k) = (f * h)(x - \tau) = \mathcal{T}_\tau(f * h)(x)$$

**Step 3:** Therefore:

$$\mathcal{T}_\tau \circ (f * h) = (\mathcal{T}_\tau f) * h$$

Or in operator notation: translation and convolution commute.

**Result:**

$$\boxed{\text{Conv}(\mathcal{T}_\tau(\mathbf{X})) = \mathcal{T}_\tau(\text{Conv}(\mathbf{X}))}$$
$\blacksquare$

**Intuition:** If a cat moves 10 pixels to the right in an image, the feature map also shifts 10 pixels to the right. The CNN detects the same features regardless of where they appear. This is why CNNs are natural for vision: objects can appear anywhere in the image.

**Important distinction:** Convolution is translation *equivariant* (output shifts with input), not translation *invariant* (output unchanged). Invariance is achieved by subsequent pooling operations.

### 1.5 Parameter Sharing Efficiency

**Fully connected layer:** For input size $C_{\text{in}} \times H \times W$ and output size $C_{\text{out}} \times H' \times W'$:

$$\text{Parameters}_{\text{FC}} = C_{\text{in}} \cdot H \cdot W \cdot C_{\text{out}} \cdot H' \cdot W'$$

**Convolutional layer:** With kernel size $K$:

$$\text{Parameters}_{\text{Conv}} = C_{\text{in}} \cdot C_{\text{out}} \cdot K^2 + C_{\text{out}}$$

**Efficiency ratio:**

$$\frac{\text{Parameters}_{\text{Conv}}}{\text{Parameters}_{\text{FC}}} = \frac{K^2}{H \cdot W \cdot H' \cdot W'} \ll 1$$

For $H = W = 32$, $K = 3$: ratio $= 9/(32^2 \cdot 30^2) \approx 10^{-5}$. CNNs use $\sim 100,000\times$ fewer parameters.

---

## 2. Algorithm / Method

### 2.1 Pseudocode: Forward Pass of a CNN

```
Algorithm: CNN_Forward
Input: Image X ∈ ℝ^{C_in × H × W}, Conv layers with weights {W_l, b_l}
Output: Feature maps or class scores

1. For each conv layer l:
     For each output channel j:
       Y_j[m,n] = b_j + Σ_i Σ_p Σ_q W_{j,i}[p,q] · X_i[m·s+p, n·s+q]
     X = activation(Y)  (e.g., ReLU)
2. For each pooling layer:
     X[m,n] = pool(X[m·p:(m+1)·p, n·p:(n+1)·p])
3. Flatten X, apply FC layers
4. Return output
```

### 2.2 Complexity Analysis

- **Conv layer forward:** $O(C_{\text{out}} \cdot C_{\text{in}} \cdot K^2 \cdot H_{\text{out}} \cdot W_{\text{out}})$
- **Conv layer backward:** Same order as forward
- **Space:** $O(\sum_l C_l \cdot H_l \cdot W_l)$ for activations (dominates memory)

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

- **State:** $s_t = \mathbf{I}_t \in \mathbb{R}^{C \times H \times W}$ — image observation
- **Action:** Output of CNN policy head
- **Reward:** Task-specific signal
- **Transition:** Environment step

### 3.2 Why RL?

CNNs are the standard state encoder in image-based RL (DQN, A2C, PPO). Translation equivariance means the agent can recognize objects regardless of position. The receptive field determines what spatial context the agent can use for decision-making.

---

## 4. Dataset

- **Name:** CIFAR-10
- **Size:** 50K train / 10K test, 32×32×3 images
- **Auto-download:**

```python
from torchvision import datasets, transforms
cifar = datasets.CIFAR10('./data', train=True, download=True,
                         transform=transforms.ToTensor())
```

---

## 5. Key Equations Summary

| Equation | Meaning |
|----------|---------|
| $O = \lfloor(I + 2P - K)/S\rfloor + 1$ | Output size formula |
| $r_L = 1 + \sum_l(k_l-1)\prod_{j<l}s_j$ | Receptive field |
| $\text{Conv}(\mathcal{T}_\tau\mathbf{X}) = \mathcal{T}_\tau(\text{Conv}(\mathbf{X}))$ | Translation equivariance |
| $\text{Params}_{\text{Conv}} = C_{\text{in}}C_{\text{out}}K^2$ | Parameter count |

---

## 6. References

- LeCun, Y. et al. "Gradient-Based Learning Applied to Document Recognition," *Proc. IEEE*, 86(11):2278–2324, 1998.
- Luo, W. et al. "Understanding the Effective Receptive Field in Deep CNNs," *NeurIPS*, 2016.
- Cohen, T. & Welling, M. "Group Equivariant Convolutional Networks," *ICML*, 2016.
- Goodfellow, I., Bengio, Y., & Courville, A. *Deep Learning*, MIT Press, 2016, Ch. 9.
