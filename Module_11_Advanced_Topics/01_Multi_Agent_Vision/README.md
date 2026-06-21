![Module Logo](../logo.png)

# Multi-Agent Reinforcement Learning for Vision

## Overview

This module explores multi-agent reinforcement learning (MARL) applied to vision tasks, where multiple agents cooperate or compete to solve complex visual problems. The mathematical framework covers stochastic games, proves the existence of Nash equilibria via Brouwer's fixed-point theorem, derives the QMIX value decomposition with monotonicity guarantees, and formulates communication protocol learning through the information bottleneck.

## Prerequisites

- Game theory (normal form, extensive form, Nash equilibrium)
- Fixed-point theory (Brouwer, Kakutani theorems)
- Multi-agent systems (cooperation, competition, communication)
- Information theory (mutual information, bottleneck principle)
- Reinforcement learning (value decomposition, CTDE paradigm)

---

## 1. Mathematical Foundations

### 1.1 Stochastic (Markov) Game Definition

**Definition:** A stochastic game (or Markov game) is a tuple:

$$\mathcal{G} = (N, \mathcal{S}, \{\mathcal{A}_i\}_{i=1}^N, P, \{R_i\}_{i=1}^N, \gamma)$$

where:
- $N$: Number of agents
- $\mathcal{S}$: State space (shared environment state)
- $\mathcal{A}_i$: Action space for agent $i$; joint action $\mathbf{a} = (a_1, \ldots, a_N) \in \mathcal{A}_1 \times \cdots \times \mathcal{A}_N$
- $P: \mathcal{S} \times \mathcal{A}_1 \times \cdots \times \mathcal{A}_N \to \Delta(\mathcal{S})$: Transition function
- $R_i: \mathcal{S} \times \mathcal{A}_1 \times \cdots \times \mathcal{A}_N \to \mathbb{R}$: Reward for agent $i$
- $\gamma \in [0, 1)$: Discount factor

**Special cases:**
- **Cooperative:** $R_1 = R_2 = \cdots = R_N$ (team reward)
- **Zero-sum (2 players):** $R_1 = -R_2$ (competitive)
- **General-sum:** Arbitrary reward structure

**Joint policy:** $\boldsymbol{\pi} = (\pi_1, \ldots, \pi_N)$ where $\pi_i: \mathcal{S} \to \Delta(\mathcal{A}_i)$.

**Value function for agent $i$:**

$$V_i^{\boldsymbol{\pi}}(s) = E_{\boldsymbol{\pi}}\left[\sum_{t=0}^\infty \gamma^t R_i(s_t, \mathbf{a}_t) \mid s_0 = s\right]$$

### 1.2 Nash Equilibrium: Existence Proof via Brouwer Fixed Point

**Definition:** A joint policy $\boldsymbol{\pi}^* = (\pi_1^*, \ldots, \pi_N^*)$ is a Nash equilibrium if:

$$V_i^{(\pi_i^*, \boldsymbol{\pi}_{-i}^*)}(s) \geq V_i^{(\pi_i, \boldsymbol{\pi}_{-i}^*)}(s) \quad \forall \pi_i, \forall i, \forall s$$

No agent can improve its value by unilaterally changing its policy.

**Theorem (Nash 1950):** Every finite stochastic game has at least one Nash equilibrium in mixed strategies.

**Proof sketch via Brouwer's Fixed-Point Theorem:**

**Step 1 (Brouwer's theorem):** Every continuous function $f: K \to K$ from a compact convex set $K \subseteq \mathbb{R}^n$ to itself has a fixed point $x^*$ with $f(x^*) = x^*$.

**Step 2:** Define the strategy space for each agent $i$ as the simplex:

$$\Sigma_i = \Delta(\mathcal{A}_i) = \{\sigma_i \in \mathbb{R}^{|\mathcal{A}_i|} : \sigma_i \geq 0, \sum_a \sigma_i(a) = 1\}$$

**Step 3:** The joint strategy space $\Sigma = \Sigma_1 \times \cdots \times \Sigma_N$ is compact and convex.

**Step 4:** Define the best-response correspondence. For each agent $i$, define the "regret" for action $a$:

$$r_i(a; \boldsymbol{\sigma}) = \max(0, V_i((a, \boldsymbol{\sigma}_{-i})) - V_i(\boldsymbol{\sigma}))$$

**Step 5:** Define the fixed-point map $f: \Sigma \to \Sigma$:

$$f_i(a; \boldsymbol{\sigma}) = \frac{\sigma_i(a) + r_i(a; \boldsymbol{\sigma})}{1 + \sum_{a'} r_i(a'; \boldsymbol{\sigma})}$$

**Step 6:** $f$ is continuous (as composition of continuous functions) and maps $\Sigma$ to $\Sigma$ (verify normalization).

**Step 7:** By Brouwer's theorem, $\exists \boldsymbol{\sigma}^*: f(\boldsymbol{\sigma}^*) = \boldsymbol{\sigma}^*$.

**Step 8:** At the fixed point, $r_i(a; \boldsymbol{\sigma}^*) = 0$ for all $a$ with $\sigma_i^*(a) > 0$ (otherwise the normalizer would differ). This means no agent has positive regret — exactly the Nash condition. $\blacksquare$

### 1.3 QMIX Decomposition: Monotonicity Guarantee

**Definition:** QMIX represents the joint Q-function $Q_{tot}$ as a monotonic combination of individual Q-functions $Q_i(s, a_i)$:

$$Q_{tot}(\boldsymbol{\tau}, \mathbf{a}) = f_\theta(Q_1(\tau_1, a_1), \ldots, Q_N(\tau_N, a_N), s)$$

where $\tau_i$ is agent $i$'s action-observation history and $s$ is the global state.

**Monotonicity constraint:**

$$\frac{\partial Q_{tot}}{\partial Q_i} \geq 0 \quad \forall i$$

**Proof that monotonicity enables decentralized execution:**

**Theorem:** Under the monotonicity constraint, $\arg\max_\mathbf{a} Q_{tot} = (\arg\max_{a_1} Q_1, \ldots, \arg\max_{a_N} Q_N)$.

**Proof:**

**Step 1:** We need to show that maximizing $Q_{tot}$ jointly is equivalent to each agent maximizing independently.

**Step 2:** Since $\frac{\partial Q_{tot}}{\partial Q_i} \geq 0$ (monotone non-decreasing), increasing any $Q_i$ cannot decrease $Q_{tot}$.

**Step 3:** For any $a_i^* = \arg\max_{a_i} Q_i(\tau_i, a_i)$, we have $Q_i(\tau_i, a_i^*) \geq Q_i(\tau_i, a_i)$ for all $a_i$.

**Step 4:** By monotonicity:

$$Q_{tot}(\ldots, Q_i(\tau_i, a_i^*), \ldots) \geq Q_{tot}(\ldots, Q_i(\tau_i, a_i), \ldots) \quad \forall a_i$$

**Step 5:** This holds for all agents simultaneously, so:

$$Q_{tot}(\boldsymbol{\tau}, \mathbf{a}^*) \geq Q_{tot}(\boldsymbol{\tau}, \mathbf{a}) \quad \forall \mathbf{a}$$

where $\mathbf{a}^* = (a_1^*, \ldots, a_N^*)$. $\blacksquare$

**QMIX mixing network:** The function $f_\theta$ is implemented as a feedforward network with non-negative weights (enforced via absolute value or hypernetwork with non-negative output).

### 1.4 Communication: Information Bottleneck

**Step 1:** Agent $i$ must compress its observation $o_i$ into a message $m_i$ of limited capacity:

$$I(m_i; o_i) \leq C$$

while maximizing information about the team objective:

$$\max_{p(m_i|o_i)} I(m_i; \text{team\_objective}) \quad \text{s.t.} \quad I(m_i; o_i) \leq C$$

**Step 2 (Information Bottleneck Lagrangian):**

$$\mathcal{L}_{IB} = I(m; \text{task}) - \beta I(m; o)$$

**Step 3:** The optimal message distribution satisfies:

$$p(m|o) = \frac{p(m)}{Z(o, \beta)}\exp\left(\frac{1}{\beta}\text{KL}(p(\text{task}|o) \| p(\text{task}|m))\right)$$

**Step 4:** In practice, parameterize the encoder: $m_i = g_\phi(o_i)$ with communication channel, and add KL regularization:

$$\mathcal{L} = -R_{team} + \beta\sum_i \text{KL}(q_\phi(m_i|o_i) \| p(m_i))$$

---

## 2. Algorithm / Method

### 2.1 Pseudocode

```
Algorithm: QMIX Multi-Agent Vision
─────────────────────────────────────────
Input: Vision environment with N agents
Output: Cooperative policy for all agents

Initialize: Individual Q-networks {Q_θᵢ}, mixing network f_w
            Replay buffer D, target networks

for episode = 1 to M do
    s₀ ← env.reset()
    for t = 0 to T do
        // Each agent selects action from local observation
        for i = 1 to N do
            oᵢₜ ← observe(sₜ, agent_i)
            aᵢₜ ← ε-greedy(Qᵢ(τᵢ, ·))
            mᵢₜ ← communicate(oᵢₜ)  // Optional
        end for
        
        // Environment step
        s_{t+1}, r_team ← env.step(a₁ₜ, ..., aₙₜ)
        Store (sₜ, {oᵢₜ}, {aᵢₜ}, r_team, s_{t+1}) in D
    end for
    
    // QMIX update
    Sample batch from D
    Q_tot = f_w(Q₁(τ₁,a₁), ..., Qₙ(τₙ,aₙ), s)
    y = r + γ · Q_tot_target(s', argmax Q_i)
    Minimize (y - Q_tot)²
    Enforce ∂f_w/∂Qᵢ ≥ 0 (non-negative weights)
end for
```

### 2.2 Complexity Analysis

- **Per-agent Q computation:** $O(|\theta_i|)$
- **Mixing network:** $O(N \cdot d_{mix}^2)$
- **Communication:** $O(N^2 \cdot d_{msg})$ for broadcast
- **Total per step:** $O(N \cdot |\theta| + N^2 \cdot d_{msg})$
- **Environment:** $O(N \cdot HW)$ for visual observations

---

## 3. Connection to Reinforcement Learning

### 3.1 MDP Formulation

$$\mathcal{G} = (N, \mathcal{S}, \{\mathcal{A}_i\}, P, R_{team}, \gamma)$$

- **Agents:** $N$ vision agents, each with partial observation of the image/scene
- **State:** Full image or scene description
- **Observation:** Agent $i$ sees local region $o_i$ (partial observability)
- **Joint action:** Each agent proposes an action (e.g., detect object, classify region)
- **Team reward:** Global task metric (mAP, mIoU, etc.)

### 3.2 Why Multi-Agent RL?

1. **Divide and conquer:** Multiple agents can process different image regions in parallel
2. **Specialization:** Agents can learn to specialize in different object types or scales
3. **Communication:** Agents share relevant findings, mimicking human collaborative analysis
4. **Scalability:** Adding agents handles larger images without increasing per-agent complexity

---

## 4. Dataset

| Dataset | Task | Agents | Description |
|---------|------|--------|-------------|
| StarCraft (SMAC) | Unit micro | 2-27 | Cooperative combat |
| Multi-Object Tracking | Tracking | N objects | Assign trackers |
| Collaborative Detection | Detection | Variable | Divide image |
| Multi-drone Surveillance | Coverage | 3-10 | Area coverage |

---

## 5. Key Equations Summary

| Equation | Description |
|----------|-------------|
| $V_i^{\boldsymbol{\pi}}(s) = E[\sum_t\gamma^t R_i(s_t,\mathbf{a}_t)]$ | Agent value function |
| $\frac{\partial Q_{tot}}{\partial Q_i} \geq 0$ | QMIX monotonicity |
| $f(\boldsymbol{\sigma}^*) = \boldsymbol{\sigma}^*$ (Brouwer) | Nash existence |
| $\mathcal{L}_{IB} = I(m;\text{task}) - \beta I(m;o)$ | Information bottleneck |
| $Q_{tot} = f_\theta(Q_1,...,Q_N,s)$ | QMIX factorization |

---

## 6. References

1. Rashid, T., et al. (2018). QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning. *ICML*.
2. Nash, J. F. (1950). Equilibrium points in n-person games. *PNAS*, 36(1), 48-49.
3. Foerster, J., et al. (2016). Learning to communicate with deep multi-agent reinforcement learning. *NeurIPS*.
4. Tishby, N., Pereira, F. C., & Bialek, W. (2000). The information bottleneck method. *arXiv:physics/0004057*.
5. Lowe, R., et al. (2017). Multi-agent actor-critic for mixed cooperative-competitive environments. *NeurIPS*.
