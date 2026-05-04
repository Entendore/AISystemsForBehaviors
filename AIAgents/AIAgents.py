import json
import random

# ============================
# Environment
# ============================
class Environment:
    def __init__(self, config_file="config.json"):
        with open(config_file, "r") as f:
            config = json.load(f)

        self.state = 0
        self.goal = config["goal"]
        self.actions = config["actions"]
        self.reward_rules = config["rewards"]  # list of rules

    def reset(self):
        self.state = 0
        return self.state

    def get_state(self):
        return self.state

    def apply_action(self, action):
        if action in self.actions:
            self.state += self.actions[action]

        reward = self._compute_reward()
        done = self.state == self.goal
        return self.state, reward, done

    def _compute_reward(self):
        # Safely evaluate conditions
        context = {"state": self.state, "goal": self.goal}
        for rule in self.reward_rules:
            try:
                if eval(rule["condition"], {}, context):
                    return rule["value"]
            except Exception:
                continue
        return 0  # fallback


# ============================
# Rule-based / Random Agent
# ============================
class RuleBasedAgent:
    def __init__(self, actions):
        self.actions = actions

    def choose_action(self, state):
        return random.choice(self.actions)


# ============================
# Q-Learning Agent
# ============================
class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = {}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}
        return self.q_table[state]

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = self.get_q_values(state)
        return max(q_values, key=q_values.get)

    def learn(self, state, action, reward, next_state):
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)
        q_values[action] += self.alpha * (
            reward + self.gamma * max(next_q_values.values()) - q_values[action]
        )


# ============================
# Simulation
# ============================
def run_simulation(agent_type="rule", episodes=10, config_file="config.json"):
    env = Environment(config_file)
    if agent_type == "rule":
        agent = RuleBasedAgent(actions=list(env.actions.keys()))
    elif agent_type == "qlearning":
        agent = QLearningAgent(actions=list(env.actions.keys()))
    else:
        raise ValueError("Invalid agent_type")

    for episode in range(episodes):
        state = env.reset()
        done, steps, total_reward = False, 0, 0

        while not done and steps < 50:
            action = agent.choose_action(state)
            next_state, reward, done = env.apply_action(action)
            if agent_type == "qlearning":
                agent.learn(state, action, reward, next_state)
            total_reward += reward
            state = next_state
            steps += 1

        print(f"Episode {episode}: finished in {steps} steps, Total Reward={total_reward}")

    if agent_type == "qlearning":
        print("\nLearned Q-Table:")
        for s, actions in agent.q_table.items():
            print(f"State {s}: {actions}")


if __name__ == "__main__":
    print("=== Rule-Based Agent ===")
    run_simulation(agent_type="rule", episodes=3)
    print("\n=== Q-Learning Agent ===")
    run_simulation(agent_type="qlearning", episodes=10)
