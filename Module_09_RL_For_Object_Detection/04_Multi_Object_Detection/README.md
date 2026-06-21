![Module Logo](../logo.png)

# Multi-Object Detection via Reinforcement Learning

## Overview

This module addresses multi-object detection through reinforcement learning, tackling the challenges of detecting multiple objects simultaneously. The mathematical framework covers Non-Maximum Suppression as a set cover approximation, the Hungarian algorithm for optimal assignment with complexity proofs, focal loss derivation for class imbalance, and multi-agent RL formulations for parallel cooperative detection.

## Prerequisites

- Combinatorial optimization (assignment problems, set cover)
- Algorithm analysis (time complexity, approximation ratios)
- Probability theory (class imbalance, calibration)
- Game theory (multi-agent systems, cooperation)
- Reinforcement learning (multi-agent RL, communication)

---

## 1. Mathematical Foundations

### 1.1 NMS as Set Cover Approximation

**Problem formulation:** Given a set of detection boxes $\mathcal{B} = \{b_1, \ldots, b_n\}$ with confidence scores $\{s_i\}$, select a subset $\mathcal{B}^* \subseteq \mathcal{B}$ that covers all true objects without redundancy.

**Step 1 (Set cover connection):** Define "coverage" — box $b_i$ covers true object $o_j$ if $\text{IoU}(b_i, o_j) > \tau$. Define cover sets:

$$C_i = \{o_j : \text{IoU}(b_i, o_j) > \tau\}$$

**Step 2:** The minimum set cover problem:

$$\min |\mathcal{B}^*| \quad \text{s.t.} \quad \bigcup_{b_i \in \mathcal{B}^*} C_i = \mathcal{O}$$

This is NP-hard. NMS provides a greedy approximation.

**Step 3 (Greedy NMS):** Iteratively select the highest-scoring box, remove all boxes with IoU $> \tau$ overlap with it.

**Step 4 (Approximation guarantee):** The greedy algorithm for set cover achieves:

$$|\mathcal{B}^*_{greedy}| \leq \ln(|\mathcal{O}|) \cdot |\mathcal{B}^*_{optimal}|$$

**Step 5:** In the detection setting, NMS with proper threshold $\tau$ produces at most one detection per object with high probability when detector scores are well-calibrated.

### 1.2 Hungarian Algorithm for Assignment

**Problem:** Given $n$ detections and $m$ ground truth objects, find the optimal one-to-one assignment minimizing total cost:

$$\min_{\sigma \in \Pi_n} \sum_{i=1}^n c(i, \sigma(i))$$

where $c(i, j)$ is the cost of assigning detection $i$ to ground truth $j$ and $\sigma$ is a permutation.

**Step 1 (Reduction):** Subtract the row minimum from each row of the cost matrix $C$:

$$C'_{ij} = C_{ij} - \min_j C_{ij}$$

**Step 2:** Subtract the column minimum from each column:

$$C''_{ij} = C'_{ij} - \min_i C'_{ij}$$

**Step 3:** Find a maximum matching in the bipartite graph of zero entries. If it's a perfect matching, we're done.

**Step 4:** If not, find the minimum vertex cover of zero entries (König's theorem: in bipartite graphs, min vertex cover = max matching).

**Step 5:** Subtract the minimum uncovered element from all uncovered elements, add it to doubly-covered elements.

**Step 6:** Repeat until a perfect matching of zeros is found.

**Complexity Proof:**

**Theorem:** The Hungarian algorithm runs in $O(n^3)$.

**Proof:** Each iteration either finds a perfect matching or strictly increases the size of the maximum matching by 1. Since the matching can increase at most $n$ times, and each iteration involves finding augmenting paths ($O(n^2)$ via BFS), the total complexity is $O(n \cdot n^2) = O(n^3)$. $\blacksquare$

### 1.3 Focal Loss Derivation

**Motivation:** In object detection, the ratio of background to foreground examples is approximately 10,000:1, overwhelming standard cross-entropy.

**Step 1 (Standard cross-entropy):**

$$\text{CE}(p_t) = -\log(p_t)$$

where $p_t = p$ if $y = 1$ and $p_t = 1-p$ if $y = 0$.

**Step 2 (Class-weighted CE):**

$$\text{CE}_\alpha(p_t) = -\alpha_t \log(p_t)$$

This reweights but doesn't address the issue that easy negatives ($p_t \approx 1$) dominate the gradient.

**Step 3 (Focal loss definition):**

$$\text{FL}(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)$$

**Step 4 (Analysis of the modulating factor):** $(1-p_t)^\gamma$:
- When $p_t \to 1$ (easy example): $(1-p_t)^\gamma \to 0$, loss $\to 0$ (down-weighted)
- When $p_t \to 0$ (hard example): $(1-p_t)^\gamma \to 1$, loss $\approx -\alpha_t\log(p_t)$ (unchanged)

**Step 5 (Gradient computation):**

$$\frac{\partial \text{FL}}{\partial p} = -\alpha_t\left[\gamma(1-p_t)^{\gamma-1}\log(p_t) + \frac{(1-p_t)^\gamma}{p_t}\right] \cdot \frac{\partial p_t}{\partial p}$$

**Step 6 (Effective sample count):** With $\gamma = 2$, an easy example with $p_t = 0.9$ contributes:

$$(1-0.9)^2 = 0.01$$

of its original loss — 100× reduction compared to standard CE.

**Step 7 (Connection to curriculum learning):** Focal loss implicitly creates a curriculum — as training progresses and easy examples become confident, the loss automatically focuses on the remaining hard cases.

### 1.4 Multi-Agent RL for Parallel Detection

**Stochastic game formulation:**

$$\mathcal{G} = (N, \mathcal{S}, \{\mathcal{A}_i\}_{i=1}^N, P, \{R_i\}_{i=1}^N, \gamma)$$

where $N$ agents each detect a different object.

**Step 1 (State sharing):** All agents share the image features: $s_t = \text{CNN}(I)$.

**Step 2 (Action):** Agent $i$ selects box transformation: $a_t^i \in \mathcal{A}$.

**Step 3 (Reward with cooperation):**

$$R_i = r_i^{individual} + \lambda \cdot r^{team}$$

$$r_i^{individual} = \text{IoU}(b_i, o_{\sigma(i)})$$

$$r^{team} = \text{mAP}(\{b_i\}, \{o_j\}) - \text{overlap\_penalty}(\{b_i\})$$

**Step 4 (Overlap penalty):** Discourages multiple agents from detecting the same object:

$$\text{overlap\_penalty} = \sum_{i \neq j} \max(0, \text{IoU}(b_i, b_j) - \tau)$$

---

## 2. Algorithm / Method

### 2.1 Pseudocode

```
Algorithm: Multi-Agent RL Object Detection
─────────────────────────────────────────────
Input: Image I, N agent slots
Output: Set of detections {b₁, ..., bₖ}

Initialize N agents with shared CNN backbone
Each agent has policy π_θᵢ and critic V_φᵢ

for episode = 1 to M do
    features ← shared_CNN(I)
    for i = 1 to N do
        b₀ⁱ ← initialize_box_i(features)
    end for
    for t = 0 to T_max do
        for i = 1 to N (in parallel) do
            aₜⁱ ~ π_θᵢ(·|sₜ, communication_msg)
            bⁱ_{t+1} ← transform(bⁱₜ, aₜⁱ)
        end for
        // Compute joint reward
        assignment ← Hungarian({bⁱ_{t+1}}, ground_truth)
        rₜ ← mAP_reward(assignment) - overlap_penalty
    end for
    Update all agents (shared parameters or independent)
end for
```

### 2.2 Complexity Analysis

- **Shared CNN:** $O(HW \cdot C^2 \cdot K^2 \cdot L)$ (computed once)
- **Per-agent decision:** $O(|\theta_i|)$ per step
- **Hungarian assignment:** $O(\max(N, M)^3)$ per step
- **Communication:** $O(N^2 \cdot d_{msg})$ for pairwise messages
- **Total:** $O(HWC^2K^2L + T \cdot N \cdot |\theta| + T \cdot N^3)$

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}^1 \times \cdots \times \mathcal{A}^N, P, R, \gamma)$$

- **Joint state:** Shared image features + individual box states + communication messages
- **Joint action:** Each agent independently selects a transformation
- **Reward:** Team mAP + individual IoU - overlap penalty
- **Transition:** Deterministic per-agent box transforms

### 3.2 Why RL?

1. **Variable number of objects:** RL agents can learn to "deactivate" when no more objects remain
2. **Cooperative behavior:** Agents learn to divide the scene and avoid redundant detections
3. **End-to-end mAP optimization:** Directly optimizes the non-differentiable detection metric
4. **Scalability:** Adding more agents handles denser scenes without architectural changes

---

## 4. Dataset

| Dataset | Avg Objects | Size | Description |
|---------|------------|------|-------------|
| MS COCO | 7.7 | 330K images | Dense multi-object |
| PASCAL VOC | 2.5 | 11,530 | Moderate density |
| Objects365 | ~15 | 2M images | Large-scale, dense |
| CrowdHuman | ~22.6 | 15K images | Heavily crowded scenes |

---

## 5. Key Equations Summary

| Equation | Description |
|----------|-------------|
| $\text{FL}(p_t) = -\alpha_t(1-p_t)^\gamma\log(p_t)$ | Focal loss |
| $\min_\sigma\sum_i c(i,\sigma(i))$ | Assignment problem |
| $\|\mathcal{B}^*_{greedy}\| \leq \ln(\|\mathcal{O}\|)\|\mathcal{B}^*_{opt}\|$ | Set cover approximation |
| $R_i = r_i^{ind} + \lambda r^{team}$ | Multi-agent reward |
| Overlap penalty: $\sum_{i\neq j}\max(0, IoU(b_i,b_j)-\tau)$ | Redundancy penalty |

---

## 6. References

1. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. *ICCV*.
2. Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly*, 2(1-2), 83-97.
3. Carion, N., et al. (2020). End-to-end object detection with transformers (DETR). *ECCV*.
4. Lowe, R., et al. (2017). Multi-agent actor-critic for mixed cooperative-competitive environments. *NeurIPS*.
