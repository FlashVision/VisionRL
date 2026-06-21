![Module Logo](../logo.png)

# Interactive Segmentation with Reinforcement Learning

## Overview

This module develops an RL agent for interactive image segmentation that intelligently selects which pixels to query from a user (or oracle) to achieve high-quality segmentation with minimal interaction. The mathematical framework covers graph cut theory, the max-flow min-cut theorem, GrabCut with GMM modeling, and active learning strategies based on information gain — all orchestrated by an RL agent that learns optimal query strategies.

## Prerequisites

- Graph theory (max-flow, min-cut, s-t graphs)
- Probabilistic modeling (Gaussian Mixture Models, EM algorithm)
- Information theory (entropy, mutual information, KL divergence)
- Optimization (combinatorial optimization, submodularity)
- Reinforcement learning (exploration strategies, active learning)

---

## 1. Mathematical Foundations

### 1.1 Graph Cut Theory: Max-Flow Min-Cut Theorem

**Definition:** For an image segmentation graph $G = (V, E)$ with source $s$ (foreground) and sink $t$ (background), define a cut $C = (S, T)$ as a partition $V = S \cup T$ with $s \in S, t \in T$.

The cost of a cut:

$$\text{cost}(S, T) = \sum_{\substack{(u,v) \in E \\ u \in S, v \in T}} w(u, v)$$

**Max-Flow Min-Cut Theorem (Ford-Fulkerson):**

$$\max_f \text{value}(f) = \min_{(S,T)} \text{cost}(S, T)$$

**Proof:**

**Step 1 (Weak duality):** For any flow $f$ and any cut $(S,T)$:

$$\text{value}(f) = \sum_{(s,v)\in E} f(s,v) = \sum_{\substack{(u,v)\in E\\u\in S,v\in T}} f(u,v) - \sum_{\substack{(v,u)\in E\\v\in T,u\in S}} f(v,u)$$

Since $f(u,v) \leq w(u,v)$ (capacity constraint):

$$\text{value}(f) \leq \sum_{\substack{(u,v)\\u\in S,v\in T}} w(u,v) = \text{cost}(S,T)$$

**Step 2 (Strong duality):** When $f$ is a maximum flow, the residual graph $G_f$ has no augmenting $s$-$t$ path. Define $S^* = \{v : \text{reachable from } s \text{ in } G_f\}$ and $T^* = V \setminus S^*$.

**Step 3:** For every edge $(u,v)$ with $u \in S^*, v \in T^*$: $f(u,v) = w(u,v)$ (saturated), otherwise $v$ would be reachable.

**Step 4:** For every edge $(v,u)$ with $v \in T^*, u \in S^*$: $f(v,u) = 0$ (empty), otherwise $v$ would be reachable via reverse edge.

**Step 5:** Therefore:

$$\text{value}(f^*) = \sum_{\substack{u\in S^*\\v\in T^*}} w(u,v) - 0 = \text{cost}(S^*, T^*)$$

Maximum flow equals minimum cut. $\blacksquare$

### 1.2 Graph Cut for Segmentation

**Step 1:** Construct the graph with:
- **Terminal edges (t-links):** $w(v, s) = -\log P(I_v | \text{FG})$ and $w(v, t) = -\log P(I_v | \text{BG})$
- **Neighbor edges (n-links):** $w(u, v) = \frac{\lambda}{d(u,v)} \exp\left(-\frac{(I_u - I_v)^2}{2\sigma^2}\right)$

**Step 2:** The segmentation energy:

$$E(\mathbf{x}) = \sum_v D_v(x_v) + \sum_{(u,v)\in\mathcal{N}} V_{uv}(x_u, x_v)$$

where $D_v$ is the data term (t-link weights) and $V_{uv}$ is the smoothness term (n-link weights).

**Step 3:** For binary labels, this energy is graph-representable and exactly minimized by min-cut.

### 1.3 GrabCut: GMM + Graph Cut

**Step 1 (Color model):** Model foreground and background using GMMs with $K$ components:

$$p(\mathbf{z}|\text{FG}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

**Step 2 (EM algorithm for GMM):** Initialize GMMs from user-provided trimap.

**E-step:** Assign component responsibilities:

$$r_{nk} = \frac{\pi_k \mathcal{N}(\mathbf{z}_n; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{z}_n; \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$

**M-step:** Update parameters:

$$\pi_k^{new} = \frac{N_k}{N}, \quad N_k = \sum_n r_{nk}$$

$$\boldsymbol{\mu}_k^{new} = \frac{1}{N_k}\sum_n r_{nk}\mathbf{z}_n$$

$$\boldsymbol{\Sigma}_k^{new} = \frac{1}{N_k}\sum_n r_{nk}(\mathbf{z}_n - \boldsymbol{\mu}_k^{new})(\mathbf{z}_n - \boldsymbol{\mu}_k^{new})^T$$

**Step 3 (GrabCut Gibbs energy):**

$$E(\boldsymbol{\alpha}, \mathbf{k}, \boldsymbol{\theta}, \mathbf{z}) = \sum_n D(\alpha_n, k_n, \boldsymbol{\theta}, z_n) + \sum_{(m,n)\in\mathcal{C}} V(\alpha_m, \alpha_n, \mathbf{z})$$

where:

$$D(\alpha_n, k_n, \boldsymbol{\theta}, z_n) = -\log\pi(\alpha_n, k_n) + \frac{1}{2}\log\det\boldsymbol{\Sigma}(\alpha_n, k_n) + \frac{1}{2}[z_n - \boldsymbol{\mu}(\alpha_n,k_n)]^T\boldsymbol{\Sigma}^{-1}[z_n - \boldsymbol{\mu}(\alpha_n,k_n)]$$

**Step 4 (Iterative minimization):**
1. Fix $\boldsymbol{\alpha}$, optimize $\mathbf{k}$ (assign pixels to GMM components)
2. Fix $\boldsymbol{\alpha}, \mathbf{k}$, optimize $\boldsymbol{\theta}$ (EM for GMM parameters)
3. Fix $\boldsymbol{\theta}, \mathbf{k}$, optimize $\boldsymbol{\alpha}$ (min-cut for segmentation)
4. Repeat until convergence.

### 1.4 Active Learning: Information Gain Derivation

**Step 1:** Define the current uncertainty about the segmentation as entropy:

$$H(\mathbf{X}) = -\sum_{\mathbf{x}} P(\mathbf{X} = \mathbf{x}|\text{data}) \log P(\mathbf{X} = \mathbf{x}|\text{data})$$

**Step 2:** The information gain from querying pixel $i$:

$$IG(i) = H(\mathbf{X}) - H(\mathbf{X}|Y_i) = H(Y_i) - H(Y_i|\mathbf{X})$$

**Step 3:** For binary segmentation with current posterior $Q_i(x_i)$:

$$H(Y_i) = -Q_i(1)\log Q_i(1) - Q_i(0)\log Q_i(0)$$

**Step 4:** The expected posterior after query (using Bayes' rule):

$$P(\mathbf{X}|Y_i = y) \propto P(Y_i = y|\mathbf{X}) \cdot P(\mathbf{X})$$

**Step 5:** Greedy query selection (approximately optimal due to submodularity):

$$i^* = \arg\max_i IG(i) = \arg\max_i H(Q_i)$$

This selects the most uncertain pixel (maximum entropy sampling).

**Step 6:** For batch queries (selecting $B$ pixels), the batch information gain:

$$IG(\mathcal{B}) = H(\mathbf{X}) - H(\mathbf{X}|\mathbf{Y}_\mathcal{B})$$

is submodular, guaranteeing that greedy selection achieves $(1 - 1/e)$ of optimal. $\blacksquare$

---

## 2. Algorithm / Method

### 2.1 Pseudocode

```
Algorithm: RL Interactive Segmentation Agent
─────────────────────────────────────────────
Input: Image I, oracle (user or ground truth)
Output: Segmentation mask with minimal queries

State: s = (image_features, current_segmentation, 
            uncertainty_map, query_history, budget_remaining)
Action: a = pixel_to_query ∈ {1, ..., N}

Initialize policy π_θ (query selection network)
Initialize segmentation model (GrabCut backend)

for episode = 1 to M do
    Initialize trimap with bounding box
    s₀ ← (features, initial_seg, entropy_map, [], B)
    for t = 0 to B (budget) do
        iₜ ← π_θ(sₜ)              // Select pixel to query
        yₜ ← oracle(iₜ)            // Get true label
        Update GMM and run graph cut
        rₜ ← ΔIoU                  // Improvement from query
        s_{t+1} ← update_state()
    end for
    Update π_θ via policy gradient
end for
```

### 2.2 Complexity Analysis

- **Information gain computation:** $O(N)$ per pixel, $O(N^2)$ for all pixels
- **Graph cut (min-cut):** $O(N \cdot E \cdot \text{max\_flow})$ ≈ $O(N^2)$ for grid graphs
- **GMM update (EM):** $O(N \cdot K \cdot d^2)$ per iteration
- **Query selection (RL):** $O(|\theta|)$ per query
- **Total per interaction round:** $O(N^2)$ dominated by graph cut

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

- **State:** Current segmentation confidence map + spatial features + remaining query budget
- **Action:** Select which pixel to query next
- **Reward:** IoU improvement after incorporating the query result
- **Transition:** Stochastic — depends on oracle response and graph cut result
- **Horizon:** Fixed budget $B$ (number of allowed queries)

### 3.2 Why RL?

1. **Sequential dependency:** The value of querying a pixel depends on what was previously queried
2. **Long-horizon planning:** Early queries can disambiguate large regions, providing more information than myopic strategies
3. **Beyond information gain:** RL can learn that queries near uncertain boundaries are more valuable than isolated uncertain pixels
4. **Personalization:** Agent adapts to user behavior patterns (click accuracy, typical object shapes)

---

## 4. Dataset

| Dataset | Size | Type | Description |
|---------|------|------|-------------|
| GrabCut | 50 images | Interactive | Standard benchmark |
| Berkeley | 100 images | Interactive | User study data |
| DAVIS | 50 videos | Video interaction | Temporal segmentation |
| SBD | 11,355 | Semantic boundaries | Boundary annotations |

---

## 5. Key Equations Summary

| Equation | Description |
|----------|-------------|
| $\max_f\text{val}(f) = \min_{(S,T)}\text{cost}(S,T)$ | Max-flow min-cut theorem |
| $p(\mathbf{z}) = \sum_k\pi_k\mathcal{N}(\mathbf{z};\mu_k,\Sigma_k)$ | Gaussian Mixture Model |
| $r_{nk} = \frac{\pi_k\mathcal{N}(z_n;\mu_k,\Sigma_k)}{\sum_j\pi_j\mathcal{N}(z_n;\mu_j,\Sigma_j)}$ | EM responsibilities |
| $IG(i) = H(X) - H(X|Y_i)$ | Information gain |
| $E(\mathbf{x}) = \sum_v D_v(x_v) + \sum_{(u,v)}V_{uv}(x_u,x_v)$ | Graph cut energy |

---

## 6. References

1. Boykov, Y., & Jolly, M.-P. (2001). Interactive graph cuts for optimal boundary & region segmentation of objects in N-D images. *ICCV*.
2. Rother, C., Kolmogorov, V., & Blake, A. (2004). GrabCut: Interactive foreground extraction using iterated graph cuts. *ACM TOG (SIGGRAPH)*.
3. Ford, L. R., & Fulkerson, D. R. (1956). Maximal flow through a network. *Canadian Journal of Mathematics*, 8, 399-404.
4. Settles, B. (2009). Active learning literature survey. *University of Wisconsin-Madison Computer Sciences Technical Report*.
5. Xu, N., et al. (2016). Deep interactive object selection. *CVPR*.
