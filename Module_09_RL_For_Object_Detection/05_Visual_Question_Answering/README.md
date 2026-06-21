![Module Logo](../logo.png)

# Visual Question Answering via Reinforcement Learning

## Overview

This module develops an RL-based Visual Question Answering (VQA) system that learns to ground natural language questions in visual content. The mathematical framework covers multimodal fusion theory via tensor products, cross-modal attention derivation, the formalization of visual grounding as an RL pointing game, and derivations of evaluation metrics (BLEU, CIDEr) used to measure answer quality.

## Prerequisites

- Multilinear algebra (tensor products, outer products, bilinear forms)
- Natural language processing (embeddings, sequence models)
- Attention mechanisms (cross-attention, self-attention)
- Evaluation metrics (n-gram precision, TF-IDF)
- Reinforcement learning (policy gradient for text generation)

---

## 1. Mathematical Foundations

### 1.1 Multimodal Fusion: Tensor Product

**Definition:** Given visual features $\mathbf{v} \in \mathbb{R}^{d_v}$ and language features $\mathbf{q} \in \mathbb{R}^{d_q}$, the tensor product fusion:

$$\mathbf{z} = \mathbf{v} \otimes \mathbf{q} \in \mathbb{R}^{d_v \times d_q}$$

with element $(i,j)$: $z_{ij} = v_i \cdot q_j$.

**Step 1 (Bilinear interaction):** A bilinear model computes:

$$y = \mathbf{v}^T \mathbf{W} \mathbf{q} + b$$

where $\mathbf{W} \in \mathbb{R}^{d_v \times d_q}$ captures all pairwise interactions. Parameters: $d_v \cdot d_q$.

**Step 2 (Low-rank factorization):** To reduce parameters, factor $\mathbf{W} = \mathbf{U}\mathbf{V}^T$ with $\mathbf{U} \in \mathbb{R}^{d_v \times k}$, $\mathbf{V} \in \mathbb{R}^{d_q \times k}$:

$$y = (\mathbf{U}^T\mathbf{v})^T(\mathbf{V}^T\mathbf{q}) = \sum_{i=1}^k (\mathbf{u}_i^T\mathbf{v})(\mathbf{v}_i^T\mathbf{q})$$

Parameters: $(d_v + d_q) \cdot k$ instead of $d_v \cdot d_q$.

**Step 3 (MLB — Multimodal Low-rank Bilinear):**

$$\mathbf{z} = (\mathbf{U}^T\mathbf{v}) \circ (\mathbf{V}^T\mathbf{q})$$

where $\circ$ is the Hadamard (element-wise) product. Output $\mathbf{z} \in \mathbb{R}^k$.

**Step 4 (MCB — Multimodal Compact Bilinear):** Approximate the full outer product using Count Sketch:

$$\Psi(\mathbf{v} \otimes \mathbf{q}) \approx \text{FFT}^{-1}(\text{FFT}(\Psi(\mathbf{v})) \circ \text{FFT}(\Psi(\mathbf{q})))$$

where $\Psi$ is the Count Sketch projection. This computes the outer product in $O(d\log d)$ instead of $O(d^2)$.

### 1.2 Cross-Modal Attention Derivation

**Step 1:** Given image regions $\mathbf{V} = [\mathbf{v}_1, \ldots, \mathbf{v}_R] \in \mathbb{R}^{R \times d_v}$ and question words $\mathbf{Q} = [\mathbf{q}_1, \ldots, \mathbf{q}_L] \in \mathbb{R}^{L \times d_q}$:

**Step 2 (Affinity matrix):** Compute cross-modal similarity:

$$\mathbf{A} = \tanh(\mathbf{V}\mathbf{W}_v + (\mathbf{Q}\mathbf{W}_q)\mathbf{1}^T) \in \mathbb{R}^{R \times L}$$

or simply $A_{ij} = \mathbf{v}_i^T\mathbf{W}\mathbf{q}_j$.

**Step 3 (Question-guided visual attention):**

$$\alpha_i = \text{softmax}_i\left(\mathbf{w}^T\tanh(\mathbf{W}_v\mathbf{v}_i + \mathbf{W}_q\bar{\mathbf{q}})\right)$$

where $\bar{\mathbf{q}} = \frac{1}{L}\sum_j \mathbf{q}_j$ (or last hidden state of question LSTM).

**Step 4 (Attended visual feature):**

$$\hat{\mathbf{v}} = \sum_{i=1}^R \alpha_i \mathbf{v}_i$$

**Step 5 (Stacked attention):** Apply attention multiple times:

$$\hat{\mathbf{v}}^{(k)} = \text{Attention}(\mathbf{V}, \hat{\mathbf{q}}^{(k-1)})$$

$$\hat{\mathbf{q}}^{(k)} = \hat{\mathbf{q}}^{(k-1)} + \hat{\mathbf{v}}^{(k)}$$

This iteratively refines the query representation using visual evidence.

### 1.3 Visual Grounding as RL (Pointing Game)

**Formulation:** The agent must point to the image region most relevant to the question.

**Step 1 (State):** $s_t = (\mathbf{V}, \mathbf{q}, \text{history of pointed regions})$

**Step 2 (Action):** $a_t \in \{1, \ldots, R\}$ — select a spatial region.

**Step 3 (Reward):** 

$$r_t = \begin{cases} +1 & \text{if pointed region overlaps ground truth region} \\ -1 & \text{otherwise} \end{cases}$$

**Step 4 (Policy):** The pointing policy uses cross-modal attention scores:

$$\pi_\theta(a = i | s) = \text{softmax}(\mathbf{w}^T\tanh(\mathbf{W}_v\mathbf{v}_i + \mathbf{W}_q\mathbf{q}))_i$$

**Step 5 (REINFORCE gradient):**

$$\nabla_\theta J = E\left[\nabla_\theta\log\pi_\theta(a|s)(r - b)\right]$$

### 1.4 BLEU Score Derivation

**Step 1 (Modified n-gram precision):** For candidate $c$ and reference set $\{r_j\}$:

$$p_n = \frac{\sum_{\text{n-gram} \in c} \min(\text{Count}(\text{n-gram}, c), \max_j\text{Count}(\text{n-gram}, r_j))}{\sum_{\text{n-gram} \in c} \text{Count}(\text{n-gram}, c)}$$

**Step 2 (Brevity penalty):** Let $c\_len = |c|$, $r\_len$ = closest reference length:

$$BP = \begin{cases} 1 & c\_len > r\_len \\ e^{1 - r\_len/c\_len} & c\_len \leq r\_len \end{cases}$$

**Step 3 (BLEU-N):**

$$\text{BLEU-N} = BP \cdot \exp\left(\sum_{n=1}^N w_n \log p_n\right)$$

with uniform weights $w_n = 1/N$.

### 1.5 CIDEr Score Derivation

**Step 1 (TF-IDF weighting):** For n-gram $\omega_k$ in candidate sentence $c_i$:

$$g_k(c_i) = \frac{h_k(c_i)}{\sum_{\omega_l \in c_i} h_l(c_i)} \cdot \log\frac{|I|}{\sum_{I_p \in I}\min(1, \sum_q h_k(r_{pq}))}$$

where $h_k(s)$ is the count of $\omega_k$ in sentence $s$.

**Step 2 (CIDEr_n for n-grams of length $n$):**

$$\text{CIDEr}_n(c_i, R_i) = \frac{1}{|R_i|}\sum_j \frac{\mathbf{g}^n(c_i) \cdot \mathbf{g}^n(r_{ij})}{\|\mathbf{g}^n(c_i)\| \cdot \|\mathbf{g}^n(r_{ij})\|}$$

This is the average cosine similarity of TF-IDF weighted n-gram vectors.

**Step 3 (Final CIDEr score):**

$$\text{CIDEr}(c_i, R_i) = \sum_{n=1}^N w_n \cdot \text{CIDEr}_n(c_i, R_i)$$

---

## 2. Algorithm / Method

### 2.1 Pseudocode

```
Algorithm: RL-based Visual Question Answering
──────────────────────────────────────────────
Input: Image I, Question q
Output: Answer a (or region pointing)

Encode: v = CNN(I), q_enc = LSTM(q)
Fuse: z = MLB(v, q_enc)  // Multimodal fusion

// For pointing/grounding (RL):
for t = 0 to T do
    aₜ ~ π_θ(·|z, history)  // Point to region
    oₜ ← extract_info(V[aₜ])
    z ← update(z, oₜ)
end for
answer ← classifier(z)

// For answer generation (RL with CIDEr reward):
Generate answer tokens a₁, ..., aₜ ~ π_θ
R ← CIDEr(generated_answer, references)
∇θ ← REINFORCE with self-critical baseline
```

### 2.2 Complexity Analysis

- **Image encoding:** $O(HW \cdot C^2 \cdot K^2 \cdot L_{CNN})$
- **Question encoding:** $O(L_q \cdot d_{LSTM}^2)$
- **Bilinear fusion:** $O(d_v \cdot d_q)$ (full) or $O((d_v + d_q)k)$ (low-rank)
- **Cross-modal attention:** $O(R \cdot L \cdot d)$
- **CIDEr computation:** $O(|V|^2 \cdot N_{max})$ for vocabulary $V$

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

- **State:** Multimodal fused representation + attention history
- **Action:** Select region to attend OR generate next answer token
- **Reward:** CIDEr/BLEU score for generation; accuracy for classification
- **Transition:** Sequential attention or autoregressive text generation

### 3.2 Why RL?

1. **Non-differentiable metrics:** CIDEr and BLEU are not differentiable; RL optimizes them directly via REINFORCE
2. **Exposure bias:** Teacher forcing at training leads to error accumulation at test time; RL trains with model-generated sequences
3. **Visual grounding:** Hard attention to image regions is a discrete non-differentiable operation requiring RL
4. **Multi-step reasoning:** Complex questions require multi-hop reasoning over image regions

---

## 4. Dataset

| Dataset | Size | Answer Type | Description |
|---------|------|-------------|-------------|
| VQA v2 | 1.1M QA pairs | Open-ended | Balanced yes/no |
| Visual Genome | 1.7M QA | Open-ended | Dense annotations |
| GQA | 22M QA | Compositional | Scene graph-based |
| VizWiz | 31K QA | Open-ended | From blind users |

---

## 5. Key Equations Summary

| Equation | Description |
|----------|-------------|
| $\mathbf{z} = (\mathbf{U}^T\mathbf{v})\circ(\mathbf{V}^T\mathbf{q})$ | Low-rank bilinear fusion |
| $\alpha_i = \text{softmax}(\mathbf{w}^T\tanh(\mathbf{W}_v\mathbf{v}_i + \mathbf{W}_q\bar{\mathbf{q}}))$ | Visual attention |
| $\text{BLEU} = BP\cdot\exp(\sum w_n\log p_n)$ | BLEU score |
| $\text{CIDEr}_n = \frac{1}{\|R\|}\sum_j\frac{\mathbf{g}^n(c)\cdot\mathbf{g}^n(r_j)}{\|\mathbf{g}^n(c)\|\|\mathbf{g}^n(r_j)\|}$ | CIDEr metric |
| $\nabla J = E[\nabla\log\pi(a\|s)(R - b)]$ | Self-critical gradient |

---

## 6. References

1. Antol, S., et al. (2015). VQA: Visual question answering. *ICCV*.
2. Kim, J.-H., et al. (2017). Hadamard product for low-rank bilinear pooling. *ICLR*.
3. Anderson, P., et al. (2018). Bottom-up and top-down attention for image captioning and VQA. *CVPR*.
4. Rennie, S. J., et al. (2017). Self-critical sequence training for image captioning. *CVPR*.
5. Vedantam, R., Zitnick, C. L., & Parikh, D. (2015). CIDEr: Consensus-based image description evaluation. *CVPR*.
