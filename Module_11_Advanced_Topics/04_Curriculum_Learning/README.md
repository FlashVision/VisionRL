![Module Logo](../logo.png)

# Curriculum Learning for Vision with Reinforcement Learning

## Overview

This module applies curriculum learning principles to vision-based reinforcement learning, where training examples or tasks are presented in a meaningful order to improve learning efficiency. The mathematical framework covers task difficulty ordering theory, the self-paced learning optimization objective, optimal curriculum theorems, and RL-based automatic curriculum generation that adapts to the learner's current capabilities.

## Prerequisites

- Optimization theory (biconvex optimization, alternating minimization)
- Learning theory (sample complexity, convergence rates)
- Scheduling theory (optimal ordering, priority scheduling)
- Information theory (learning progress, compression)
- Reinforcement learning (bandit algorithms, teacher-student framework)

---

## 1. Mathematical Foundations

### 1.1 Task Difficulty Ordering Theory

**Definition:** A curriculum is an ordering $\sigma: \{1, \ldots, N\} \to \{1, \ldots, N\}$ over training examples, where $\sigma(k)$ is the index of the $k$-th example presented.

**Step 1 (Difficulty measure):** Define task difficulty for example $i$:

$$d_i = \mathcal{L}(f_\theta(x_i), y_i)$$

measured as the loss under the current model (or a reference model).

**Step 2 (Easy-to-hard principle):** Present examples in increasing difficulty:

$$d_{\sigma(1)} \leq d_{\sigma(2)} \leq \cdots \leq d_{\sigma(N)}$$

**Step 3 (Theoretical motivation):** Under convex loss functions, the gradient from easy examples points toward the optimum more reliably:

$$\text{Var}[\nabla\mathcal{L}_i] \propto d_i$$

Easy examples have lower gradient variance, providing more stable initial learning.

**Step 4 (Anti-curriculum):** In some settings (e.g., hard example mining), the opposite ordering is beneficial:

$$d_{\sigma(1)} \geq d_{\sigma(2)} \geq \cdots \geq d_{\sigma(N)}$$

This is optimal when the model is already close to convergence and needs to focus on difficult boundary cases.

### 1.2 Self-Paced Learning Objective

**Definition (Kumar et al. 2010):** Self-paced learning jointly optimizes the model $w$ and a weight vector $v \in [0,1]^N$:

$$\min_{w, v} E(w, v; \lambda) = \sum_{i=1}^N v_i \mathcal{L}(y_i, f_w(x_i)) - \lambda\sum_{i=1}^N v_i$$

subject to $v_i \in [0, 1]$.

**Step 1 (Interpretation):** The objective has two competing terms:
- First term: weighted training loss (minimize loss on selected examples)
- Second term: $-\lambda\|v\|_1$ encourages including more examples (regularization)

**Step 2 (Optimal $v$ for fixed $w$):** Taking derivative with respect to $v_i$:

$$\frac{\partial E}{\partial v_i} = \mathcal{L}_i - \lambda = 0 \implies v_i^* = \begin{cases} 1 & \text{if } \mathcal{L}_i < \lambda \\ 0 & \text{if } \mathcal{L}_i > \lambda \end{cases}$$

Examples with loss below threshold $\lambda$ are included (easy examples first).

**Step 3 (Curriculum emerges):** As training progresses, increase $\lambda$:
- Small $\lambda$: only easiest examples selected
- Large $\lambda$: all examples included
- This automatically creates an easy-to-hard curriculum

**Step 4 (Optimal $w$ for fixed $v$):** Standard weighted ERM:

$$w^* = \arg\min_w \sum_i v_i\mathcal{L}(y_i, f_w(x_i))$$

**Step 5 (Alternating optimization):**
1. Fix $w$, solve for $v^*$ (threshold at $\lambda$)
2. Fix $v$, solve for $w^*$ (weighted SGD)
3. Increase $\lambda$ (anneal pace parameter)
4. Repeat until convergence

**Step 6 (Generalized self-paced regularizer):** Replace $-\lambda\|v\|_1$ with:

$$g(v; \lambda) = -\lambda\left(\sum_i\log(v_i + \epsilon)\right) \quad \text{(soft weighting)}$$

or mixture models:

$$g(v; \lambda) = \lambda\sum_i(v_i\log v_i + (1-v_i)\log(1-v_i))$$

### 1.3 Optimal Curriculum Theorem

**Theorem (Weinshall et al. 2018, building on Bengio et al. 2009):** Under a linear model $f_w(x) = w^Tx$ with squared loss, training with curriculum (easy-first) converges faster than uniform random sampling.

**Proof sketch:**

**Step 1:** The convergence rate of SGD depends on the condition number $\kappa = \lambda_{max}/\lambda_{min}$ of the data covariance matrix.

**Step 2:** Easy examples (small loss) correspond to data points near the current decision boundary, which have smaller projection on the "hard" directions (high-eigenvalue components).

**Step 3:** By first training on easy examples, the model quickly aligns with the dominant eigenvectors. The effective condition number for the remaining hard examples is reduced.

**Step 4:** This reduces the total number of iterations needed by a factor proportional to:

$$\text{Speedup} \approx \frac{\kappa_{uniform}}{\kappa_{curriculum}} = O(\sqrt{\kappa})$$

**Step 5 (Formal bound):** Under appropriate regularity conditions:

$$E[\mathcal{L}(w_T^{curriculum})] \leq E[\mathcal{L}(w_T^{random})] - \Omega\left(\frac{1}{\sqrt{T}}\right)$$

### 1.4 RL for Automatic Curriculum Generation

**Step 1 (Teacher-Student framework):** A teacher agent (RL policy) selects tasks for a student learner:

- Teacher state: $s_T = (\text{student\_performance\_history}, \text{task\_statistics})$
- Teacher action: $a_T = \text{task\_parameters}$ (difficulty level, domain)
- Teacher reward: $R_T = \Delta\text{performance}_{student}$ (student's learning progress)

**Step 2 (Learning progress as reward):**

$$R_T = |\mathcal{L}_{student}^{before}(T_k) - \mathcal{L}_{student}^{after}(T_k)|$$

Tasks with maximum learning progress are selected (zone of proximal development).

**Step 3 (Multi-armed bandit formulation):** Each difficulty level is an arm:

$$\text{reward}(d) = E[\text{learning\_progress} | \text{difficulty} = d]$$

The UCB1 strategy for task selection:

$$d^* = \arg\max_d \left[\hat{\mu}_d + c\sqrt{\frac{\log t}{n_d}}\right]$$

---

## 2. Algorithm / Method

### 2.1 Pseudocode

```
Algorithm: RL Curriculum Generator for Vision
──────────────────────────────────────────────
Input: Student model f_θ, task space T, difficulty range [d_min, d_max]
Output: Trained student + curriculum policy

Initialize: Teacher policy π_T (curriculum selector)
            Student model f_θ

for meta_episode = 1 to M do
    // Teacher selects curriculum
    s_T ← (student_performance_vector, training_progress)
    
    for curriculum_step = 1 to L do
        // Teacher selects task difficulty
        d_t ~ π_T(·|s_T)
        task_t ← sample_task(difficulty=d_t)
        
        // Student trains on selected task
        performance_before ← evaluate(f_θ, val_set)
        Train f_θ on task_t for K steps
        performance_after ← evaluate(f_θ, val_set)
        
        // Teacher reward: learning progress
        r_T ← performance_after - performance_before
        s_T ← update(s_T, performance_after, d_t)
    end for
    
    // Update teacher via policy gradient
    Update π_T via REINFORCE with learning progress reward
end for
```

### 2.2 Complexity Analysis

- **Student training (per task):** $O(K \cdot |\theta_{student}|)$ for $K$ SGD steps
- **Teacher decision:** $O(|\theta_{teacher}|)$ per task selection
- **Evaluation:** $O(|V| \cdot |\theta_{student}|)$ on validation set $V$
- **Total:** $O(M \cdot L \cdot (K|\theta_s| + |V||\theta_s|))$
- **Overhead vs. random curriculum:** Only the teacher network ($O(|\theta_T|)$ per step)

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

$$\mathcal{M}_{teacher} = (\mathcal{S}_T, \mathcal{A}_T, P_T, R_T, \gamma)$$

- **State:** Student's performance profile + training history + current difficulty level
- **Action:** Next task/difficulty level to present to student
- **Reward:** Student's learning progress (performance improvement)
- **Transition:** Stochastic — depends on student's learning dynamics
- **Discount:** $\gamma \approx 0.99$ (long-term student improvement matters)

### 3.2 Why RL for Curriculum?

1. **Adaptive difficulty:** RL teacher adapts to the student's current level, unlike fixed schedules
2. **Non-stationary optimal curriculum:** As the student improves, the optimal task changes — RL handles this naturally
3. **Multi-dimensional difficulty:** Tasks vary in many ways (size, noise, occlusion); RL explores this space
4. **Maximizing learning signal:** Not all easy-to-hard curricula are equal; RL finds the one that maximizes overall learning

---

## 4. Dataset

| Dataset | Task | Curriculum Dimension | Description |
|---------|------|---------------------|-------------|
| CIFAR-10 | Classification | Label noise level | Noisy labels |
| ImageNet | Classification | Image complexity | Varied difficulty |
| CurriculumNet | Classification | Data difficulty scores | Pre-computed scores |
| CLEVR | Reasoning | Question complexity | Multi-hop visual QA |

---

## 5. Key Equations Summary

| Equation | Description |
|----------|-------------|
| $E(w,v;\lambda) = \sum_i v_i\mathcal{L}_i - \lambda\sum_i v_i$ | Self-paced learning |
| $v_i^* = \mathbb{1}[\mathcal{L}_i < \lambda]$ | Optimal example selection |
| $R_T = \|\mathcal{L}^{before}-\mathcal{L}^{after}\|$ | Learning progress reward |
| $d^* = \arg\max_d[\hat{\mu}_d + c\sqrt{\log t/n_d}]$ | UCB task selection |
| Speedup $\approx O(\sqrt{\kappa})$ | Curriculum convergence gain |

---

## 6. References

1. Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. *ICML*.
2. Kumar, M. P., Packer, B., & Koller, D. (2010). Self-paced learning for latent variable models. *NeurIPS*.
3. Graves, A., et al. (2017). Automated curriculum learning for neural networks. *ICML*.
4. Narvekar, S., et al. (2020). Curriculum learning for reinforcement learning domains: A framework and survey. *JMLR*, 21, 1-50.
5. Weinshall, D., Cohen, G., & Amir, D. (2018). Curriculum learning by transfer learning. *arXiv:1802.03796*.
