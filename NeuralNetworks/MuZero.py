import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque

class MCTSNode:
    def __init__(self, latent, parent=None, prior=1.0):
        self.latent = latent
        self.parent = parent
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}  # action tuple -> node

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class ContinuousMCTS:
    def __init__(self, agent, action_dim, c_puct=1.0, num_simulations=25, num_samples=5):
        self.agent = agent
        self.action_dim = action_dim
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.num_samples = num_samples

    def select_child(self, node):
        total_visits = sum(child.visit_count for child in node.children.values()) + 1
        best_score = -float('inf')
        best_action, best_child = None, None

        for action_tuple, child in node.children.items():
            u = self.c_puct * child.prior * (total_visits ** 0.5) / (1 + child.visit_count)
            q = child.value()
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action_tuple
                best_child = child
        return best_action, best_child

    def expand_node(self, node):
        mean, std, value = self.agent.prediction(node.latent)
        for _ in range(self.num_samples):
            action = torch.normal(mean, std)
            action_tuple = tuple(action.tolist())
            if action_tuple not in node.children:
                next_latent, reward, _, _, _ = self.agent.recurrent_inference(node.latent, action)
                node.children[action_tuple] = MCTSNode(next_latent, parent=node, prior=1.0/self.num_samples)
        return value

    def backpropagate(self, path, value):
        for node in reversed(path):
            node.value_sum += value
            node.visit_count += 1

    def run(self, root_latent):
        root = MCTSNode(root_latent)
        for _ in range(self.num_simulations):
            node = root
            path = [node]

            while node.children:
                action, node = self.select_child(node)
                path.append(node)

            value = self.expand_node(node)
            self.backpropagate(path, value)

        best_action = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
        return torch.tensor(best_action)

# ----------------------------
# Replay Buffer
# ----------------------------
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(torch.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# ----------------------------
# MuZero Network
# ----------------------------
class RepresentationNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.fc = nn.Linear(state_dim, hidden_dim)
    def forward(self, state):
        return F.relu(self.fc(state))

class DynamicsNetwork(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
    def forward(self, latent, action):
        x = torch.cat([latent, action], dim=-1)
        x = F.relu(self.fc1(x))
        next_latent = F.relu(self.fc2(x))
        reward = self.reward_head(x)
        return next_latent, reward

class PredictionNetwork(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
    def forward(self, latent):
        x = F.relu(self.fc1(latent))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(-5,2)
        std = torch.exp(log_std)
        value = self.value_head(x)
        return mean, std, value

class BehaviorMuZero:
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.representation = RepresentationNetwork(state_dim, hidden_dim)
        self.dynamics = DynamicsNetwork(hidden_dim, action_dim, hidden_dim)
        self.prediction = PredictionNetwork(hidden_dim, action_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(
            list(self.representation.parameters()) +
            list(self.dynamics.parameters()) +
            list(self.prediction.parameters()),
            lr=1e-3
        )

    def initial_inference(self, state):
        latent = self.representation(state)
        mean, std, value = self.prediction(latent)
        return latent, mean, std, value

    def recurrent_inference(self, latent, action):
        next_latent, reward = self.dynamics(latent, action)
        mean, std, value = self.prediction(next_latent)
        return next_latent, reward, mean, std, value

    def update(self, batch):
        state, action, reward, next_state, done = batch
        latent = self.representation(state)
        next_latent, pred_reward, mean, std, value = self.recurrent_inference(latent, action)
        
        # Loss: reward + value + policy (negative log-likelihood)
        reward_loss = F.mse_loss(pred_reward, reward)
        value_loss = F.mse_loss(value, reward)  # simplified
        dist = torch.distributions.Normal(mean, std)
        policy_loss = -dist.log_prob(action).mean()
        
        loss = reward_loss + value_loss + policy_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

def train_behavior_muzero(agent, simulation, episodes=300, max_steps=50,
                          buffer_capacity=5000, batch_size=32):
    replay = ReplayBuffer(capacity=buffer_capacity)
    mcts = ContinuousMCTS(agent, action_dim=agent.action_dim, num_simulations=20, num_samples=5)
    
    for ep in range(episodes):
        state = torch.tensor(simulation.reset(), dtype=torch.float32)
        total_reward = 0

        for step in range(max_steps):
            # Plan next action using latent-space MCTS
            latent, _, _, _ = agent.initial_inference(state)
            action = mcts.run(latent)
            
            # Step simulation
            next_state, reward, done = simulation.step(action.numpy())
            next_state = torch.tensor(next_state, dtype=torch.float32)
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            
            # Store experience
            replay.push(state, action, reward_tensor, next_state, done)
            total_reward += reward
            state = next_state
            
            # Update agent from replay buffer
            if len(replay) > batch_size:
                batch = replay.sample(batch_size)
                loss = agent.update(batch)
            
            if done:
                break

        # Logging
        print(f"Episode {ep+1}/{episodes}, Total Reward: {total_reward:.2f}")

    print("Training complete.")


class PointMass2D:
    def __init__(self, dt=0.1, max_steps=50, target=None):
        self.dt = dt
        self.max_steps = max_steps
        self.state = None
        self.step_count = 0
        self.target = target if target is not None else np.array([5.0, 5.0])

    def reset(self):
        self.state = np.array([0.0, 0.0, 0.0, 0.0])  # x, y, vx, vy
        self.step_count = 0
        return self.state.copy()

    def step(self, action):
        """
        action: np.array([ax, ay]) accelerations in x and y
        """
        ax, ay = action
        x, y, vx, vy = self.state
        vx += ax * self.dt
        vy += ay * self.dt
        x += vx * self.dt
        y += vy * self.dt
        self.state = np.array([x, y, vx, vy])
        self.step_count += 1

        # Reward = negative distance to target
        distance = np.linalg.norm(self.target - np.array([x, y]))
        reward = -distance

        done = self.step_count >= self.max_steps or distance < 0.1
        return self.state.copy(), reward, done
    
# Simulation and agent parameters
state_dim = 4  # x, y, vx, vy
action_dim = 2  # ax, ay

simulation = PointMass2D()
agent = BehaviorMuZero(state_dim, action_dim)
mcts = ContinuousMCTS(agent, action_dim, num_simulations=20, num_samples=5)

# Run one episode
state = torch.tensor(simulation.reset(), dtype=torch.float32)
total_reward = 0

for step in range(simulation.max_steps):
    latent, _, _, _ = agent.initial_inference(state)
    action = mcts.run(latent)
    next_state, reward, done = simulation.step(action.numpy())
    state = torch.tensor(next_state, dtype=torch.float32)
    total_reward += reward
    print(f"Step {step+1}: State={next_state}, Action={action.numpy()}, Reward={reward:.2f}")
    if done:
        break

print(f"Total Reward: {total_reward:.2f}")