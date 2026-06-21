"""Comprehensive tests for VisionRL: RL algorithms, environments, policy networks,
reward computation, and training loops.

VisionRL is primarily notebook-based, so these tests verify standalone implementations
of the key RL algorithms taught in the course modules.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Grid Environment (discrete)
# ---------------------------------------------------------------------------


class GridEnv:
    """Simple 4x4 grid environment for testing RL algorithms."""

    def __init__(self, size: int = 4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # up, down, left, right
        self.goal = (size - 1, size - 1)
        self.reset()

    def reset(self):
        self.pos = (0, 0)
        return self._state()

    def _state(self):
        return self.pos[0] * self.size + self.pos[1]

    def step(self, action):
        r, c = self.pos
        if action == 0:
            r = max(0, r - 1)
        elif action == 1:
            r = min(self.size - 1, r + 1)
        elif action == 2:
            c = max(0, c - 1)
        elif action == 3:
            c = min(self.size - 1, c + 1)
        self.pos = (r, c)
        done = self.pos == self.goal
        reward = 1.0 if done else -0.01
        return self._state(), reward, done, {}

    @property
    def observation_space_n(self):
        return self.n_states

    @property
    def action_space_n(self):
        return self.n_actions


# ---------------------------------------------------------------------------
# Image-based Environment (continuous observation)
# ---------------------------------------------------------------------------


class ImageEnv:
    """Simple image-based environment that returns pixel observations."""

    def __init__(self, img_size: int = 8, n_actions: int = 4):
        self.img_size = img_size
        self.n_actions = n_actions
        self.pos = [img_size // 2, img_size // 2]
        self.step_count = 0
        self.max_steps = 50

    def reset(self):
        self.pos = [self.img_size // 2, self.img_size // 2]
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros((1, self.img_size, self.img_size), dtype=np.float32)
        obs[0, self.pos[0], self.pos[1]] = 1.0
        return obs

    def step(self, action):
        self.step_count += 1
        if action == 0:
            self.pos[0] = max(0, self.pos[0] - 1)
        elif action == 1:
            self.pos[0] = min(self.img_size - 1, self.pos[0] + 1)
        elif action == 2:
            self.pos[1] = max(0, self.pos[1] - 1)
        elif action == 3:
            self.pos[1] = min(self.img_size - 1, self.pos[1] + 1)

        done = self.step_count >= self.max_steps
        dist = abs(self.pos[0]) + abs(self.pos[1])
        reward = -0.1 * dist / self.img_size
        return self._get_obs(), reward, done, {}


# ---------------------------------------------------------------------------
# Policy Networks
# ---------------------------------------------------------------------------


class QNetwork(nn.Module):
    """Simple Q-Network for discrete action spaces."""

    def __init__(self, n_states: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(n_states, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE algorithm."""

    def __init__(self, n_states: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(n_states, hidden)
        self.fc2 = nn.Linear(hidden, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)


class ConvQNetwork(nn.Module):
    """CNN-based Q-Network for image observations."""

    def __init__(self, in_channels: int = 1, n_actions: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(16 * 4, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class ActorCritic(nn.Module):
    """Actor-Critic network with shared feature extractor."""

    def __init__(self, n_states: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.shared = nn.Linear(n_states, hidden)
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        features = F.relu(self.shared(x))
        policy = F.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        return policy, value


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: list = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Tests: Grid Environment
# ---------------------------------------------------------------------------


class TestGridEnv:
    def test_reset(self):
        env = GridEnv(4)
        state = env.reset()
        assert state == 0
        assert env.pos == (0, 0)

    def test_step_actions(self):
        env = GridEnv(4)
        env.reset()
        s, r, done, _ = env.step(1)  # down
        assert env.pos == (1, 0)
        assert not done

        s, r, done, _ = env.step(3)  # right
        assert env.pos == (1, 1)

    def test_reach_goal(self):
        env = GridEnv(4)
        env.reset()
        for _ in range(3):
            env.step(1)  # down
        for _ in range(3):
            _, r, done, _ = env.step(3)  # right
        assert done
        assert r == 1.0

    def test_boundary_clipping(self):
        env = GridEnv(4)
        env.reset()
        env.step(0)  # up from (0,0)
        assert env.pos == (0, 0)
        env.step(2)  # left from (0,0)
        assert env.pos == (0, 0)


class TestImageEnv:
    def test_reset_shape(self):
        env = ImageEnv(img_size=8)
        obs = env.reset()
        assert obs.shape == (1, 8, 8)
        assert obs.sum() == 1.0

    def test_step(self):
        env = ImageEnv(img_size=8)
        env.reset()
        obs, reward, done, info = env.step(0)
        assert obs.shape == (1, 8, 8)
        assert isinstance(reward, float)
        assert not done

    def test_max_steps_terminates(self):
        env = ImageEnv(img_size=8)
        env.max_steps = 5
        env.reset()
        for _ in range(5):
            _, _, done, _ = env.step(0)
        assert done


# ---------------------------------------------------------------------------
# Tests: Policy Networks
# ---------------------------------------------------------------------------


class TestQNetwork:
    def test_forward_shape(self):
        net = QNetwork(n_states=16, n_actions=4)
        x = torch.randn(8, 16)
        out = net(x)
        assert out.shape == (8, 4)

    def test_gradient_flow(self):
        net = QNetwork(n_states=16, n_actions=4)
        x = torch.randn(4, 16)
        out = net(x)
        loss = out.sum()
        loss.backward()
        assert net.fc1.weight.grad is not None


class TestPolicyNetwork:
    def test_output_probabilities(self):
        net = PolicyNetwork(n_states=16, n_actions=4)
        x = torch.randn(4, 16)
        probs = net(x)
        assert probs.shape == (4, 4)
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)
        assert (probs >= 0).all()

    def test_action_sampling(self):
        net = PolicyNetwork(n_states=16, n_actions=4)
        x = torch.randn(1, 16)
        probs = net(x)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        assert 0 <= action.item() < 4


class TestConvQNetwork:
    def test_forward(self):
        net = ConvQNetwork(in_channels=1, n_actions=4)
        x = torch.randn(2, 1, 8, 8)
        out = net(x)
        assert out.shape == (2, 4)


class TestActorCritic:
    def test_forward(self):
        net = ActorCritic(n_states=16, n_actions=4)
        x = torch.randn(4, 16)
        policy, value = net(x)
        assert policy.shape == (4, 4)
        assert value.shape == (4, 1)
        assert torch.allclose(policy.sum(dim=1), torch.ones(4), atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: Replay Buffer
# ---------------------------------------------------------------------------


class TestReplayBuffer:
    def test_push_and_len(self):
        buf = ReplayBuffer(capacity=100)
        assert len(buf) == 0
        buf.push(0, 1, 0.5, 1, False)
        assert len(buf) == 1

    def test_sample(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(20):
            buf.push(i, i % 4, float(i), i + 1, i == 19)
        states, actions, rewards, next_states, dones = buf.sample(5)
        assert states.shape == (5,)
        assert actions.shape == (5,)
        assert rewards.shape == (5,)

    def test_capacity_overflow(self):
        buf = ReplayBuffer(capacity=5)
        for i in range(10):
            buf.push(i, 0, 0.0, i + 1, False)
        assert len(buf) == 5


# ---------------------------------------------------------------------------
# Tests: Reward Computation
# ---------------------------------------------------------------------------


class TestRewardComputation:
    def test_discounted_returns(self):
        rewards = [1.0, 0.0, 0.0, 1.0]
        gamma = 0.99
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        assert len(returns) == 4
        assert returns[-1] == 1.0
        assert returns[0] > returns[-1]

    def test_advantage_computation(self):
        values = torch.tensor([0.5, 0.6, 0.7, 0.8])
        rewards = torch.tensor([1.0, 0.0, 0.0, 1.0])
        next_values = torch.tensor([0.6, 0.7, 0.8, 0.0])
        gamma = 0.99
        td_targets = rewards + gamma * next_values
        advantages = td_targets - values
        assert advantages.shape == (4,)
        assert advantages[0] > 0

    def test_gae_computation(self):
        """Generalized Advantage Estimation (GAE-lambda)."""
        gamma = 0.99
        lam = 0.95
        rewards = [0.0, 0.0, 1.0]
        values = [0.1, 0.2, 0.3]
        next_values = [0.2, 0.3, 0.0]

        deltas = [r + gamma * nv - v for r, nv, v in zip(rewards, next_values, values)]
        advantages = []
        gae = 0
        for delta in reversed(deltas):
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
        assert len(advantages) == 3
        assert advantages[-1] == pytest.approx(deltas[-1], abs=1e-6)


# ---------------------------------------------------------------------------
# Tests: Q-Learning (tabular)
# ---------------------------------------------------------------------------


class TestQLearning:
    def test_q_table_update(self):
        env = GridEnv(4)
        q_table = np.zeros((env.n_states, env.n_actions))
        alpha = 0.1
        gamma = 0.99

        state = env.reset()
        action = 3  # right
        next_state, reward, done, _ = env.step(action)

        old_q = q_table[state, action]
        q_table[state, action] = old_q + alpha * (reward + gamma * np.max(q_table[next_state]) - old_q)
        assert q_table[state, action] != 0.0

    def test_convergence(self):
        env = GridEnv(3)
        q_table = np.zeros((env.n_states, env.n_actions))
        alpha, gamma, epsilon = 0.5, 0.9, 0.3

        for _ in range(500):
            state = env.reset()
            for _ in range(50):
                if np.random.random() < epsilon:
                    action = np.random.randint(env.n_actions)
                else:
                    action = np.argmax(q_table[state])
                next_state, reward, done, _ = env.step(action)
                q_table[state, action] += alpha * (
                    reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
                )
                state = next_state
                if done:
                    break

        start_state = 0
        assert np.max(q_table[start_state]) > 0


# ---------------------------------------------------------------------------
# Tests: DQN Training Loop
# ---------------------------------------------------------------------------


class TestDQNTraining:
    def test_single_update(self):
        env = GridEnv(4)
        n_states = env.n_states
        n_actions = env.n_actions
        net = QNetwork(n_states, n_actions, hidden=32)
        target_net = QNetwork(n_states, n_actions, hidden=32)
        target_net.load_state_dict(net.state_dict())
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        buf = ReplayBuffer(100)
        state = env.reset()
        for _ in range(32):
            action = np.random.randint(n_actions)
            next_state, reward, done, _ = env.step(action)
            buf.push(state, action, reward, next_state, done)
            state = next_state if not done else env.reset()

        states, actions, rewards, next_states, dones = buf.sample(16)
        states_t = F.one_hot(torch.tensor(states, dtype=torch.long), n_states).float()
        next_states_t = F.one_hot(torch.tensor(next_states, dtype=torch.long), n_states).float()
        actions_t = torch.tensor(actions, dtype=torch.long)
        rewards_t = torch.tensor(rewards)
        dones_t = torch.tensor(dones)

        q_values = net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q = target_net(next_states_t).max(1)[0]
            targets = rewards_t + 0.99 * next_q * (1 - dones_t)

        loss = F.mse_loss(q_values, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert loss.item() >= 0


# ---------------------------------------------------------------------------
# Tests: REINFORCE
# ---------------------------------------------------------------------------


class TestREINFORCE:
    def test_single_episode(self):
        env = GridEnv(3)
        net = PolicyNetwork(env.n_states, env.n_actions, hidden=32)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        state = env.reset()
        log_probs = []
        rewards = []

        for _ in range(20):
            state_t = F.one_hot(torch.tensor(state), env.n_states).float().unsqueeze(0)
            probs = net(state_t)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            next_state, reward, done, _ = env.step(action.item())
            rewards.append(reward)
            state = next_state
            if done:
                break

        gamma = 0.99
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        loss = sum(-lp * R for lp, R in zip(log_probs, returns_t))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert loss.item() != 0.0


# ---------------------------------------------------------------------------
# Tests: Actor-Critic (A2C)
# ---------------------------------------------------------------------------


class TestActorCriticTraining:
    def test_single_step(self):
        env = GridEnv(3)
        net = ActorCritic(env.n_states, env.n_actions, hidden=32)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        state = env.reset()
        state_t = F.one_hot(torch.tensor(state), env.n_states).float().unsqueeze(0)
        policy, value = net(state_t)

        dist = torch.distributions.Categorical(policy)
        action = dist.sample()
        next_state, reward, done, _ = env.step(action.item())

        next_state_t = F.one_hot(torch.tensor(next_state), env.n_states).float().unsqueeze(0)
        _, next_value = net(next_state_t)

        td_target = reward + 0.99 * next_value.detach() * (1 - int(done))
        advantage = td_target - value

        actor_loss = -dist.log_prob(action) * advantage.detach()
        critic_loss = F.mse_loss(value, td_target)
        loss = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert not torch.isnan(loss)


# ---------------------------------------------------------------------------
# Tests: PPO
# ---------------------------------------------------------------------------


class TestPPO:
    def test_clipped_objective(self):
        old_probs = torch.tensor([0.3, 0.5, 0.2])
        new_probs = torch.tensor([0.4, 0.4, 0.3])
        advantages = torch.tensor([1.0, -0.5, 0.2])
        epsilon = 0.2

        ratios = new_probs / old_probs
        clipped = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)
        loss1 = ratios * advantages
        loss2 = clipped * advantages
        ppo_loss = -torch.min(loss1, loss2).mean()

        assert ppo_loss.shape == ()
        assert not torch.isnan(ppo_loss)

    def test_value_loss(self):
        predicted_values = torch.tensor([0.5, 0.7, 0.3])
        returns = torch.tensor([0.6, 0.8, 0.4])
        value_loss = F.mse_loss(predicted_values, returns)
        assert value_loss.item() > 0


# ---------------------------------------------------------------------------
# Integration: create env → create agent → train → evaluate
# ---------------------------------------------------------------------------


class TestIntegrationRL:
    def test_full_training_loop(self):
        env = GridEnv(3)
        net = QNetwork(env.n_states, env.n_actions, hidden=32)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
        buf = ReplayBuffer(500)
        epsilon = 1.0
        gamma = 0.95

        total_rewards = []
        for episode in range(50):
            state = env.reset()
            ep_reward = 0
            for _ in range(30):
                state_t = F.one_hot(torch.tensor(state), env.n_states).float().unsqueeze(0)
                if np.random.random() < epsilon:
                    action = np.random.randint(env.n_actions)
                else:
                    with torch.no_grad():
                        action = net(state_t).argmax().item()

                next_state, reward, done, _ = env.step(action)
                buf.push(state, action, reward, next_state, done)
                ep_reward += reward
                state = next_state

                if len(buf) >= 16:
                    states, actions, rewards_b, next_states, dones = buf.sample(16)
                    s_t = F.one_hot(torch.tensor(states, dtype=torch.long), env.n_states).float()
                    ns_t = F.one_hot(torch.tensor(next_states, dtype=torch.long), env.n_states).float()
                    a_t = torch.tensor(actions, dtype=torch.long)
                    r_t = torch.tensor(rewards_b)
                    d_t = torch.tensor(dones)

                    q = net(s_t).gather(1, a_t.unsqueeze(1)).squeeze()
                    with torch.no_grad():
                        nq = net(ns_t).max(1)[0]
                        targets = r_t + gamma * nq * (1 - d_t)

                    loss = F.mse_loss(q, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if done:
                    break

            total_rewards.append(ep_reward)
            epsilon = max(0.05, epsilon * 0.95)

        assert len(total_rewards) == 50
        last_10 = np.mean(total_rewards[-10:])
        first_10 = np.mean(total_rewards[:10])
        assert last_10 >= first_10 - 1.0

    def test_evaluate_policy(self):
        env = GridEnv(3)
        net = QNetwork(env.n_states, env.n_actions, hidden=32)

        eval_rewards = []
        for _ in range(5):
            state = env.reset()
            ep_reward = 0
            for _ in range(30):
                state_t = F.one_hot(torch.tensor(state), env.n_states).float().unsqueeze(0)
                with torch.no_grad():
                    action = net(state_t).argmax().item()
                state, reward, done, _ = env.step(action)
                ep_reward += reward
                if done:
                    break
            eval_rewards.append(ep_reward)
        assert len(eval_rewards) == 5
