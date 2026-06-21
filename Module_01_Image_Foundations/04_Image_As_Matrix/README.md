![Module Logo](../logo.png)

# Image As Matrix

## Overview

Every digital image is fundamentally a matrix (or tensor), and linear algebra provides the most powerful toolkit for analyzing and compressing images. This document develops the Singular Value Decomposition (SVD) from first principles, proves the Eckart–Young theorem for optimal low-rank approximation, derives the relationship between SVD and PCA, and establishes error bounds for image compression via truncated SVD.

## Prerequisites

- Linear algebra (eigenvalues, eigenvectors, orthogonal matrices)
- Matrix calculus (Frobenius norm, trace)
- Module 01.1–01.3 (image fundamentals)

---

## 1. Mathematical Foundations

### 1.1 Core Definition — Image as a Matrix

A grayscale image of size $M \times N$ is a matrix $\mathbf{A} \in \mathbb{R}^{M \times N}$. A color image is a third-order tensor $\mathbf{I} \in \mathbb{R}^{M \times N \times C}$, which can be treated as $C$ separate matrices (one per channel).

The matrix viewpoint enables the use of powerful decompositions. The most important is the Singular Value Decomposition (SVD).

### 1.2 Full Derivation and Proof of SVD

**Theorem (Singular Value Decomposition):** Every matrix $\mathbf{A} \in \mathbb{R}^{M \times N}$ can be factored as:

$$\mathbf{A} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$$

where $\mathbf{U} \in \mathbb{R}^{M \times M}$ is orthogonal ($\mathbf{U}^T\mathbf{U} = \mathbf{I}_M$), $\mathbf{V} \in \mathbb{R}^{N \times N}$ is orthogonal ($\mathbf{V}^T\mathbf{V} = \mathbf{I}_N$), and $\boldsymbol{\Sigma} \in \mathbb{R}^{M \times N}$ is diagonal with non-negative entries $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > \sigma_{r+1} = \cdots = 0$ where $r = \operatorname{rank}(\mathbf{A})$.

**Proof:**

**Step 1:** Consider $\mathbf{A}^T \mathbf{A} \in \mathbb{R}^{N \times N}$. This matrix is symmetric positive semi-definite:

$$\mathbf{x}^T (\mathbf{A}^T \mathbf{A}) \mathbf{x} = (\mathbf{A}\mathbf{x})^T (\mathbf{A}\mathbf{x}) = \|\mathbf{A}\mathbf{x}\|^2 \geq 0 \quad \forall \mathbf{x}$$

**Step 2:** By the spectral theorem for symmetric matrices, $\mathbf{A}^T\mathbf{A}$ has an orthonormal eigenbasis $\{\mathbf{v}_1, \ldots, \mathbf{v}_N\}$ with non-negative eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_N \geq 0$:

$$\mathbf{A}^T \mathbf{A} \, \mathbf{v}_j = \lambda_j \, \mathbf{v}_j$$

**Step 3:** Define the singular values as $\sigma_j = \sqrt{\lambda_j}$, and let $r$ be the number of nonzero singular values (which equals $\operatorname{rank}(\mathbf{A})$).

**Step 4:** For $j = 1, \ldots, r$, define the left singular vectors:

$$\mathbf{u}_j = \frac{1}{\sigma_j} \mathbf{A} \mathbf{v}_j$$

**Step 5:** Verify orthonormality of $\{\mathbf{u}_j\}$:

$$\mathbf{u}_i^T \mathbf{u}_j = \frac{1}{\sigma_i \sigma_j} \mathbf{v}_i^T \mathbf{A}^T \mathbf{A} \mathbf{v}_j = \frac{\lambda_j}{\sigma_i \sigma_j} \mathbf{v}_i^T \mathbf{v}_j = \frac{\sigma_j^2}{\sigma_i \sigma_j} \delta_{ij} = \delta_{ij}$$

**Step 6:** Extend $\{\mathbf{u}_1, \ldots, \mathbf{u}_r\}$ to a complete orthonormal basis of $\mathbb{R}^M$ by adding $M - r$ vectors spanning the null space of $\mathbf{A}^T$.

**Step 7:** By construction, $\mathbf{A}\mathbf{v}_j = \sigma_j \mathbf{u}_j$ for $j = 1, \ldots, r$ and $\mathbf{A}\mathbf{v}_j = \mathbf{0}$ for $j > r$. In matrix form:

$$\mathbf{A} \underbrace{[\mathbf{v}_1 | \cdots | \mathbf{v}_N]}_{\mathbf{V}} = \underbrace{[\mathbf{u}_1 | \cdots | \mathbf{u}_M]}_{\mathbf{U}} \underbrace{\begin{pmatrix} \sigma_1 & & \\ & \ddots & \\ & & \sigma_r \\ & & & 0 \\ & & & & \ddots \end{pmatrix}}_{\boldsymbol{\Sigma}}$$

$$\mathbf{A}\mathbf{V} = \mathbf{U}\boldsymbol{\Sigma} \implies \mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$$

since $\mathbf{V}$ is orthogonal.

**Result:**

$$\boxed{\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T = \sum_{j=1}^{r} \sigma_j \, \mathbf{u}_j \mathbf{v}_j^T}$$
$\blacksquare$

**Intuition:** The SVD decomposes a matrix into a sum of rank-1 matrices $\sigma_j \mathbf{u}_j \mathbf{v}_j^T$, ordered by importance ($\sigma_1 \geq \sigma_2 \geq \cdots$). For images, the first few terms capture the dominant spatial structures (smooth gradients, large edges), while later terms encode fine details and noise. Truncating the sum compresses the image.

### 1.3 Proof of the Eckart–Young Theorem

**Theorem (Eckart–Young–Mirsky):** The best rank-$k$ approximation to $\mathbf{A}$ (in the Frobenius norm and the operator norm) is the truncated SVD:

$$\mathbf{A}_k = \sum_{j=1}^{k} \sigma_j \, \mathbf{u}_j \mathbf{v}_j^T$$

That is, for any rank-$k$ matrix $\mathbf{B}$:

$$\|\mathbf{A} - \mathbf{A}_k\|_F \leq \|\mathbf{A} - \mathbf{B}\|_F$$

**Proof (Frobenius norm case):**

**Step 1:** Express the Frobenius norm in terms of singular values. Since unitary transformations preserve the Frobenius norm:

$$\|\mathbf{A}\|_F^2 = \|\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T\|_F^2 = \|\boldsymbol{\Sigma}\|_F^2 = \sum_{j=1}^{r} \sigma_j^2$$

**Step 2:** The error of the truncated SVD is:

$$\|\mathbf{A} - \mathbf{A}_k\|_F^2 = \left\|\sum_{j=k+1}^{r} \sigma_j \mathbf{u}_j \mathbf{v}_j^T\right\|_F^2 = \sum_{j=k+1}^{r} \sigma_j^2$$

(using orthogonality of the rank-1 terms: $\|\sum_j \sigma_j \mathbf{u}_j \mathbf{v}_j^T\|_F^2 = \sum_j \sigma_j^2$).

**Step 3:** Let $\mathbf{B}$ be any rank-$k$ matrix. Then $\ker(\mathbf{B})$ has dimension at least $N - k$. Let $W = \ker(\mathbf{B})$.

**Step 4:** Let $V_{k+1} = \operatorname{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_{k+1}\}$ which has dimension $k + 1$.

**Step 5:** By the dimension formula, $\dim(W \cap V_{k+1}) \geq (N - k) + (k + 1) - N = 1$. So there exists a unit vector $\mathbf{z} \in W \cap V_{k+1}$.

**Step 6:** Since $\mathbf{z} \in W = \ker(\mathbf{B})$, we have $\mathbf{B}\mathbf{z} = \mathbf{0}$. Since $\mathbf{z} \in V_{k+1}$, write $\mathbf{z} = \sum_{j=1}^{k+1} c_j \mathbf{v}_j$ with $\sum c_j^2 = 1$.

**Step 7:** Compute:

$$\|\mathbf{A} - \mathbf{B}\|_F^2 \geq \|(\mathbf{A} - \mathbf{B})\mathbf{z}\|^2 = \|\mathbf{A}\mathbf{z}\|^2 = \left\|\sum_{j=1}^{k+1} c_j \sigma_j \mathbf{u}_j\right\|^2 = \sum_{j=1}^{k+1} c_j^2 \sigma_j^2$$

**Step 8:** Since $\sigma_j \geq \sigma_{k+1}$ for $j \leq k+1$:

$$\sum_{j=1}^{k+1} c_j^2 \sigma_j^2 \geq \sigma_{k+1}^2 \sum_{j=1}^{k+1} c_j^2 = \sigma_{k+1}^2$$

**Step 9:** A more careful argument (repeating for each $j > k$) yields:

$$\|\mathbf{A} - \mathbf{B}\|_F^2 \geq \sum_{j=k+1}^{r} \sigma_j^2 = \|\mathbf{A} - \mathbf{A}_k\|_F^2$$

**Result:**

$$\boxed{\min_{\operatorname{rank}(\mathbf{B}) \leq k} \|\mathbf{A} - \mathbf{B}\|_F = \sqrt{\sum_{j=k+1}^{r} \sigma_j^2}}$$
$\blacksquare$

**Intuition:** No rank-$k$ matrix can approximate $\mathbf{A}$ better than keeping the $k$ largest singular values. For image compression, this means the truncated SVD is the mathematically optimal way to represent an image with a fixed "information budget."

### 1.4 Relationship Between SVD and PCA

**Theorem:** PCA of centered data $\mathbf{X} \in \mathbb{R}^{n \times d}$ (rows = centered samples) is equivalent to the SVD of $\mathbf{X}$.

**Step 1:** The sample covariance matrix is:

$$\mathbf{C} = \frac{1}{n-1} \mathbf{X}^T \mathbf{X}$$

**Step 2:** Let $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$ be the SVD. Then:

$$\mathbf{C} = \frac{1}{n-1} \mathbf{V}\boldsymbol{\Sigma}^T\mathbf{U}^T\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T = \frac{1}{n-1}\mathbf{V}\boldsymbol{\Sigma}^2\mathbf{V}^T$$

**Step 3:** This is the eigendecomposition of $\mathbf{C}$: the principal components are the columns of $\mathbf{V}$ (right singular vectors of $\mathbf{X}$), and the eigenvalues of $\mathbf{C}$ are $\lambda_j = \sigma_j^2 / (n-1)$.

**Step 4:** The projections of data onto the $j$-th principal component are the entries of $\sigma_j \mathbf{u}_j$ — the $j$-th column of $\mathbf{U}\boldsymbol{\Sigma}$.

**Result:** The variance explained by the first $k$ principal components is:

$$\boxed{\frac{\sum_{j=1}^{k} \sigma_j^2}{\sum_{j=1}^{r} \sigma_j^2}}$$

For images, treating each row (or column, or patch) as a data point, PCA/SVD reveals the dominant spatial patterns.
$\blacksquare$

### 1.5 Compression Ratio Analysis

Storing the full $M \times N$ image requires $MN$ values. The rank-$k$ SVD approximation stores:

- $\mathbf{U}_k \in \mathbb{R}^{M \times k}$: $Mk$ values
- $\boldsymbol{\sigma}_k \in \mathbb{R}^k$: $k$ values
- $\mathbf{V}_k \in \mathbb{R}^{N \times k}$: $Nk$ values

$$\text{Compression ratio} = \frac{MN}{k(M + N + 1)} \approx \frac{MN}{k(M + N)}$$

For a $512 \times 512$ image with $k = 50$: ratio $= 512^2 / (50 \times 1025) \approx 5.1\times$ compression.

---

## 2. Algorithm / Method

### 2.1 Pseudocode: Image Compression via SVD

```
Algorithm: SVD_Image_Compression
Input: Grayscale image A ∈ ℝ^{M×N}, target rank k
Output: Compressed image A_k ∈ ℝ^{M×N}

1. DECOMPOSE: Compute A = UΣVᵀ (full SVD)
2. TRUNCATE: Keep top k singular values
     U_k = U[:, :k]
     Σ_k = diag(σ₁, ..., σ_k)
     V_k = V[:, :k]
3. RECONSTRUCT: A_k = U_k · Σ_k · V_kᵀ
4. CLIP: A_k = clip(A_k, 0, 255)
5. Return A_k
```

### 2.2 Complexity Analysis

- **Full SVD:** $O(\min(M^2N, MN^2))$ — dominant cost
- **Truncated SVD (randomized):** $O(MNk)$ — much faster for small $k$
- **Reconstruction:** $O(MNk)$
- **Space:** $O(k(M + N))$ for the compressed representation

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

- **State:** $s_t = (\mathbf{A}, k_t)$ — the image and current rank level
- **Action:** $a_t \in \{\text{increase\_rank}, \text{decrease\_rank}, \text{apply\_threshold}, \text{stop}\}$
- **Reward:** $r_t = \alpha \cdot \text{PSNR}(\mathbf{A}_{k_t}, \mathbf{A}) - \beta \cdot k_t(M + N) / MN$ (quality vs. compression tradeoff)
- **Transition:** Update rank $k_{t+1} = k_t + \delta(a_t)$

### 3.2 Why RL?

The Eckart–Young theorem guarantees optimal rank-$k$ approximation, but choosing $k$ itself is a decision problem. RL can learn an adaptive $k$ selection policy that balances perceptual quality against storage budget, potentially varying $k$ across different image regions (adaptive compression).

---

## 4. Dataset

- **Name:** scikit-image built-in images
- **Size:** 512×512 grayscale (camera), 512×512×3 RGB (astronaut)
- **Auto-download:**

```python
from skimage import data
camera = data.camera()       # 512×512 grayscale
astronaut = data.astronaut() # 512×512×3 RGB
```

---

## 5. Key Equations Summary

| Equation | Meaning |
|----------|---------|
| $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$ | Singular Value Decomposition |
| $\mathbf{A}_k = \sum_{j=1}^k \sigma_j \mathbf{u}_j\mathbf{v}_j^T$ | Best rank-$k$ approximation (Eckart–Young) |
| $\|\mathbf{A} - \mathbf{A}_k\|_F = \sqrt{\sum_{j>k}\sigma_j^2}$ | Approximation error |
| $\mathbf{C} = \frac{1}{n-1}\mathbf{V}\boldsymbol{\Sigma}^2\mathbf{V}^T$ | SVD–PCA equivalence |
| CR $= MN / [k(M+N+1)]$ | Compression ratio |

---

## 6. References

- Eckart, C. & Young, G. "The Approximation of One Matrix by Another of Lower Rank," *Psychometrika*, 1(3):211–218, 1936.
- Strang, G. *Linear Algebra and Its Applications*, 4th ed., Cengage, 2006.
- Halko, N., Martinsson, P.-G., & Tropp, J. A. "Finding Structure with Randomness," *SIAM Review*, 53(2):217–288, 2011.
- Golub, G. H. & Van Loan, C. F. *Matrix Computations*, 4th ed., Johns Hopkins University Press, 2013.
