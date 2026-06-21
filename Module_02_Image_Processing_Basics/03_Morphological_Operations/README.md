![Module Logo](../logo.png)

# Morphological Operations

## Overview

Mathematical morphology provides a rigorous algebraic framework for analyzing shapes in images using set-theoretic operations. This document proves the fundamental algebraic properties of erosion and dilation (duality, distributivity, extensivity, anti-extensivity), proves the idempotency of morphological opening and closing, and establishes that these operations form a complete lattice structure.

## Prerequisites

- Set theory (unions, intersections, complements)
- Abstract algebra (lattice theory basics)
- Binary image representation
- Module 02.1 (convolution concepts)

---

## 1. Mathematical Foundations

### 1.1 Core Definitions

Let $A \subseteq \mathbb{Z}^2$ be a binary image (foreground pixel set) and $B \subseteq \mathbb{Z}^2$ be a structuring element.

**Minkowski Addition (Dilation):**

$$A \oplus B = \{a + b : a \in A, \, b \in B\} = \bigcup_{b \in B} A_b$$

where $A_b = \{a + b : a \in A\}$ is the translation of $A$ by $b$.

**Minkowski Subtraction (Erosion):**

$$A \ominus B = \{z \in \mathbb{Z}^2 : B_z \subseteq A\} = \bigcap_{b \in B} A_{-b}$$

where $B_z = \{z + b : b \in B\}$.

**Reflection:** $\hat{B} = \{-b : b \in B\}$.

### 1.2 Proof of Duality Between Erosion and Dilation

**Theorem:** Dilation and erosion are dual operations under complementation:

$$(A \oplus B)^c = A^c \ominus \hat{B}$$

**Proof:**

**Step 1:** $z \in (A \oplus B)^c$ iff $z \notin A \oplus B$.

**Step 2:** By definition of dilation: $z \notin A \oplus B$ iff $\nexists \, a \in A, b \in B$ such that $z = a + b$.

**Step 3:** Equivalently: for all $b \in B$, $z - b \notin A$, i.e., $z - b \in A^c$.

**Step 4:** Using the reflection $\hat{B}$: for all $\hat{b} \in \hat{B}$ (where $\hat{b} = -b$), $z + \hat{b} \in A^c$, i.e., $\hat{B}_z \subseteq A^c$.

**Step 5:** By definition of erosion: $z \in A^c \ominus \hat{B}$.

**Result:**

$$\boxed{(A \oplus B)^c = A^c \ominus \hat{B}}$$
$\blacksquare$

**Corollary:** $(A \ominus B)^c = A^c \oplus \hat{B}$ (obtained by applying the theorem to $A^c$ and $\hat{B}$).

**Intuition:** Dilating the foreground is the same as eroding the background (and vice versa). This duality means we only need to implement one operation; the other follows by complementation.

### 1.3 Proof of Distributivity

**Theorem (Dilation distributes over union):**

$$A \oplus (B \cup C) = (A \oplus B) \cup (A \oplus C)$$

**Proof:**

**Step 1:** $z \in A \oplus (B \cup C)$ iff $\exists \, a \in A, \, d \in B \cup C$ such that $z = a + d$.

**Step 2:** $d \in B \cup C$ means $d \in B$ or $d \in C$.

**Step 3:** If $d \in B$: $z = a + d$ with $a \in A, d \in B$, so $z \in A \oplus B$.

**Step 4:** If $d \in C$: similarly, $z \in A \oplus C$.

**Step 5:** Therefore $z \in (A \oplus B) \cup (A \oplus C)$. The reverse inclusion follows by the same argument.
$\blacksquare$

**Theorem (Erosion distributes over intersection):**

$$A \ominus (B \cup C) = (A \ominus B) \cap (A \ominus C)$$

**Proof:**

**Step 1:** $z \in A \ominus (B \cup C)$ iff $(B \cup C)_z \subseteq A$.

**Step 2:** $(B \cup C)_z = B_z \cup C_z$, so $B_z \cup C_z \subseteq A$.

**Step 3:** This means $B_z \subseteq A$ AND $C_z \subseteq A$.

**Step 4:** Therefore $z \in (A \ominus B)$ and $z \in (A \ominus C)$, i.e., $z \in (A \ominus B) \cap (A \ominus C)$.
$\blacksquare$

### 1.4 Opening and Closing — Definitions and Properties

**Morphological Opening:**

$$A \circ B = (A \ominus B) \oplus B$$

**Morphological Closing:**

$$A \bullet B = (A \oplus B) \ominus B$$

#### Proof of Idempotency of Opening

**Theorem:** $(A \circ B) \circ B = A \circ B$ — opening is idempotent.

**Proof:**

**Step 1:** We need two preliminary results:

**Lemma 1 (Anti-extensivity of erosion-dilation):** $A \ominus B \oplus B \subseteq A$ when $\mathbf{0} \in B$.

*Proof:* $z \in (A \ominus B) \oplus B$ means $z = a' + b$ where $a' \in A \ominus B$ and $b \in B$. Since $a' \in A \ominus B$, we have $B_{a'} \subseteq A$, so $a' + b \in A$ (because $b \in B$). Thus $z \in A$.

**Lemma 2 (Erosion is order-preserving):** If $A \subseteq C$, then $A \ominus B \subseteq C \ominus B$.

*Proof:* $z \in A \ominus B$ means $B_z \subseteq A \subseteq C$, so $z \in C \ominus B$.

**Step 2:** Let $A' = A \circ B = (A \ominus B) \oplus B$. We need to show $A' \circ B = A'$, i.e.:

$$((A' \ominus B) \oplus B) = A'$$

**Step 3:** By Lemma 1 (anti-extensivity), $A' \circ B \subseteq A'$.

**Step 4:** For the reverse inclusion, note $A \ominus B \subseteq A' \ominus B$ would give $A' = (A \ominus B) \oplus B \subseteq (A' \ominus B) \oplus B = A' \circ B$.

To show $A \ominus B \subseteq A' \ominus B$: We need $A' \supseteq$ all translations $B_z$ that fit inside $A$. Since $A' = (A \ominus B) \oplus B$, for any $z \in A \ominus B$, $B_z \subseteq A'$. Thus $z \in A' \ominus B$, giving $A \ominus B \subseteq A' \ominus B$.

**Step 5:** Therefore $A' \subseteq A' \circ B$, and combined with Step 3:

$$A' \circ B = A'$$

**Result:** Opening is idempotent: applying it twice gives the same result as applying it once.
$\blacksquare$

**Intuition:** Opening removes structures smaller than $B$ from $A$. Once they are removed, there is nothing left to remove on a second application. Geometrically, opening is the union of all translations of $B$ that fit inside $A$: $A \circ B = \bigcup\{B_z : B_z \subseteq A\}$.

#### Proof of Idempotency of Closing

**Theorem:** $(A \bullet B) \bullet B = A \bullet B$.

**Proof:** By duality. Define $A' = A \bullet B = (A \oplus B) \ominus B$. Since $(A \bullet B)^c = A^c \circ \hat{B}$ (closing of $A$ is complement of opening of $A^c$ with reflected $B$), and opening is idempotent:

$$(A \bullet B) \bullet B = \left[\left(A^c \circ \hat{B}\right)^c \bullet B\right]$$

Applying the duality and idempotency of opening completes the proof.
$\blacksquare$

### 1.5 Lattice Structure

The set of all subsets of $\mathbb{Z}^2$ with $\subseteq$ forms a complete lattice. Erosion and dilation are adjoint operations in this lattice:

$$A \ominus B \subseteq C \iff A \subseteq C \oplus B$$

This adjunction is the foundation of mathematical morphology, from which all properties (extensivity, idempotency, monotonicity) follow.

---

## 2. Algorithm / Method

### 2.1 Pseudocode: Binary Erosion

```
Algorithm: Binary_Erosion
Input: Binary image A ∈ {0,1}^{M×N}, structuring element B
Output: Eroded image A'

1. For each pixel (m,n):
     fits = True
     For each (i,j) in B:
       if A[m+i, n+j] == 0:
         fits = False; break
     A'[m,n] = 1 if fits else 0
2. Return A'
```

### 2.2 Complexity Analysis

- **Time:** $O(MN \cdot |B|)$ where $|B|$ is the number of pixels in the structuring element
- **Space:** $O(MN)$ for the output
- **Decomposable SE:** If $B = B_1 \oplus B_2$, then $A \ominus B = (A \ominus B_1) \ominus B_2$, reducing cost for large SEs

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

- **State:** $s_t = \mathbf{I}_t$ — the current binary or grayscale image
- **Action:** $a_t \in \{\text{erode}(B_k), \text{dilate}(B_k), \text{open}(B_k), \text{close}(B_k)\}$ with various SEs $B_k$
- **Reward:** $r_t = \text{IoU}(\mathbf{I}_{t+1}, \mathbf{I}_{\text{target}})$ — intersection over union with target segmentation
- **Transition:** Apply morphological operation

### 3.2 Why RL?

Morphological pipelines require choosing operations and structuring element sizes in the right order. Since opening and closing are idempotent, repeated application is wasteful — an RL agent can learn efficient sequences that avoid redundant operations. The algebraic properties (distributivity, duality) constrain the search space.

---

## 4. Dataset

- **Name:** scikit-image sample images
- **Size:** Various binary and grayscale test images
- **Auto-download:**

```python
from skimage import data, morphology
coins = data.coins() > 100  # threshold to binary
opened = morphology.opening(coins, morphology.disk(5))
```

---

## 5. Key Equations Summary

| Equation | Meaning |
|----------|---------|
| $A \oplus B = \bigcup_{b \in B} A_b$ | Dilation (Minkowski sum) |
| $A \ominus B = \bigcap_{b \in B} A_{-b}$ | Erosion (Minkowski subtraction) |
| $(A \oplus B)^c = A^c \ominus \hat{B}$ | Erosion–dilation duality |
| $A \circ B = (A \ominus B) \oplus B$ | Opening |
| $(A \circ B) \circ B = A \circ B$ | Idempotency of opening |

---

## 6. References

- Serra, J. *Image Analysis and Mathematical Morphology*, Academic Press, 1982.
- Haralick, R. M., Sternberg, S. R., & Zhuang, X. "Image Analysis Using Mathematical Morphology," *IEEE Trans. PAMI*, 9(4):532–550, 1987.
- Soille, P. *Morphological Image Analysis*, 2nd ed., Springer, 2004.
- Heijmans, H. J. A. M. *Morphological Image Operators*, Academic Press, 1994.
