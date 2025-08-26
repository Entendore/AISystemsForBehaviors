import numpy as np
import random

class MDP:
    def __init__(self, states, actions, transition_probabilities, rewards, gamma=0.9, terminal_states=None):
        self.states = states
        self.actions = actions
        self.P = transition_probabilities
        self.R = rewards
        self.gamma = gamma
        self.terminal_states = terminal_states or set()

    def policy_evaluation(self, policy, theta=1e-6):
        V = {s: 0 for s in self.states}
        while True:
            delta = 0
            for s in self.states:
                if s in self.terminal_states:
                    continue
                a = policy[s]
                v = sum(prob * (self.R[s][a] + self.gamma * V[next_state])
                        for prob, next_state in self.P[s][a])
                delta = max(delta, abs(V[s] - v))
                V[s] = v
            if delta < theta:
                break
        return V

    def value_iteration(self, theta=1e-6):
        V = {s: 0 for s in self.states}
        while True:
            delta = 0
            for s in self.states:
                if s in self.terminal_states:
                    continue
                v = max(
                    sum(prob * (self.R[s][a] + self.gamma * V[next_state])
                        for prob, next_state in self.P[s][a])
                    for a in self.actions
                )
                delta = max(delta, abs(V[s] - v))
                V[s] = v
            if delta < theta:
                break
        policy = {}
        for s in self.states:
            if s in self.terminal_states:
                policy[s] = None
            else:
                best_a = max(
                    self.actions,
                    key=lambda a: sum(prob * (self.R[s][a] + self.gamma * V[next_state])
                                      for prob, next_state in self.P[s][a])
                )
                policy[s] = best_a
        return V, policy

    def policy_iteration(self, theta=1e-6):
        policy = {s: random.choice(self.actions) for s in self.states if s not in self.terminal_states}
        policy.update({s: None for s in self.terminal_states})
        while True:
            V = self.policy_evaluation(policy, theta)
            policy_stable = True
            for s in self.states:
                if s in self.terminal_states:
                    continue
                old_action = policy[s]
                best_action = max(
                    self.actions,
                    key=lambda a: sum(prob * (self.R[s][a] + self.gamma * V[next_state])
                                      for prob, next_state in self.P[s][a])
                )
                policy[s] = best_action
                if old_action != best_action:
                    policy_stable = False
            if policy_stable:
                break
        return V, policy

    def step(self, state, action):
        if state in self.terminal_states:
            return state, 0
        transitions = self.P[state][action]
        probs, next_states = zip(*transitions)
        next_state = random.choices(next_states, weights=probs)[0]
        reward = self.R[state][action]
        return next_state, reward

    def q_learning(self, episodes=1000, alpha=0.1, epsilon=0.1):
        Q = {s: {a: 0 for a in self.actions} for s in self.states}
        for _ in range(episodes):
            state = random.choice([s for s in self.states if s not in self.terminal_states])
            while state not in self.terminal_states:
                if random.random() < epsilon:
                    action = random.choice(self.actions)
                else:
                    action = max(Q[state], key=Q[state].get)
                next_state, reward = self.step(state, action)
                Q[state][action] += alpha * (reward + self.gamma * max(Q[next_state].values()) - Q[state][action])
                state = next_state
        policy = {s: (max(Q[s], key=Q[s].get) if s not in self.terminal_states else None) for s in self.states}
        return Q, policy

def main():
    states = ['s0', 's1', 's2']
    actions = ['a0', 'a1']
    terminal_states = {'s2'}

    P = {
        's0': {'a0': [(0.8, 's0'), (0.2, 's1')], 'a1': [(0.5, 's1'), (0.5, 's2')]},
        's1': {'a0': [(1.0, 's0')], 'a1': [(1.0, 's2')]},
        's2': {'a0': [(1.0, 's2')], 'a1': [(1.0, 's2')]}
    }

    R = {
        's0': {'a0': 0, 'a1': 1},
        's1': {'a0': 0, 'a1': 2},
        's2': {'a0': 0, 'a1': 0}
    }

    mdp = MDP(states, actions, P, R, gamma=0.9, terminal_states=terminal_states)

    # Value Iteration
    V_vi, policy_vi = mdp.value_iteration()
    print("Value Iteration Policy:", policy_vi)

    # Policy Iteration
    V_pi, policy_pi = mdp.policy_iteration()
    print("Policy Iteration Policy:", policy_pi)

    # Q-learning
    Q, policy_q = mdp.q_learning(episodes=5000, alpha=0.1, epsilon=0.1)
    print("Q-learning Policy:", policy_q)

    # Simulate a trajectory using learned policy
    state = 's0'
    print("\nSimulating trajectory using Q-learning policy:")
    for _ in range(5):
        action = policy_q[state]
        next_state, reward = mdp.step(state, action)
        print(f"{state} --{action}/{reward}--> {next_state}")
        state = next_state

if __name__ == "__main__":
    main()
