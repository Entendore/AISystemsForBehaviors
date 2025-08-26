import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------
# Parameters
# ----------------------------
POPULATION_SIZE = 20
INPUT_SIZE = 10  # pos(x,y) + vel(x,y) + nearest obstacle(dx,dy) + nearest goal(dx,dy) + nearest agent(dx,dy) + avg agent vector(dx,dy)
HIDDEN_SIZE = 16
OUTPUT_SIZE = 2  # continuous acceleration vector
GENERATIONS = 20
MUTATION_RATE = 0.1
STEPS = 50
LR = 0.05  # RL fine-tuning

OBSTACLE_RADIUS = 0.5

# ----------------------------
# Physics-Based Neural Agent
# ----------------------------
class PhysicsAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        acc = torch.tanh(self.fc2(x))
        return acc

# ----------------------------
# Agent Wrapper
# ----------------------------
class Agent:
    def __init__(self, genome=None):
        self.net = PhysicsAgent()
        if genome is not None:
            self.set_genome(genome)
        self.fitness = None
        self.positions = []

    def get_genome(self):
        return np.concatenate([p.data.numpy().flatten() for p in self.net.parameters()])

    def set_genome(self, genome):
        pointer = 0
        for p in self.net.parameters():
            shape = p.data.shape
            size = p.data.numel()
            p.data = torch.tensor(genome[pointer:pointer+size].reshape(shape), dtype=torch.float32)
            pointer += size

    # ----------------------------
    # Vector Calculations
    # ----------------------------
    def nearest_obstacle_vector(self, pos, obstacles):
        if len(obstacles)==0: return np.array([0.0,0.0])
        distances=[np.linalg.norm(obs-pos) for obs in obstacles]
        nearest=obstacles[np.argmin(distances)]
        vec = nearest-pos
        return vec/(np.linalg.norm(vec)+1e-5)

    def nearest_goal_vector(self, pos, goals):
        if len(goals)==0: return np.array([0.0,0.0])
        distances=[np.linalg.norm(goal-pos) for goal in goals]
        nearest=goals[np.argmin(distances)]
        vec = nearest-pos
        return vec/(np.linalg.norm(vec)+1e-5)

    def nearest_agent_vector(self, pos, agents):
        distances=[np.linalg.norm(agent_pos-pos) for agent_pos in agents if not np.array_equal(agent_pos,pos)]
        if not distances: return np.array([0.0,0.0])
        nearest=agents[np.argmin(distances)]
        vec=nearest-pos
        return vec/(np.linalg.norm(vec)+1e-5)

    def average_agent_vector(self, pos, agents):
        others=[agent_pos for agent_pos in agents if not np.array_equal(agent_pos,pos)]
        if not others: return np.array([0.0,0.0])
        avg=np.mean([other-pos for other in others], axis=0)
        return avg/(np.linalg.norm(avg)+1e-5)

    # ----------------------------
    # Evaluate Agent (Physics + RL)
    # ----------------------------
    def evaluate(self, goals, obstacles, other_agents_positions, fine_tune=True):
        pos = np.array([0.0,0.0])
        vel = np.array([0.0,0.0])
        self.positions = [pos.copy()]
        collision_penalty=0
        goal_reward=0
        visited_goals=set()

        optimizer = torch.optim.SGD(self.net.parameters(), lr=LR) if fine_tune else None

        for _ in range(STEPS):
            obs_vec = self.nearest_obstacle_vector(pos, obstacles)
            goal_vec = self.nearest_goal_vector(pos, goals)
            nearest_agent_vec = self.nearest_agent_vector(pos, other_agents_positions)
            avg_agent_vec = self.average_agent_vector(pos, other_agents_positions)
            inp = torch.tensor(np.concatenate([pos, vel, obs_vec, goal_vec, nearest_agent_vec, avg_agent_vec]), dtype=torch.float32)

            acc = self.net(inp)
            acc_np = acc.detach().numpy()
            vel += acc_np
            pos += vel

            for obs in obstacles:
                if np.linalg.norm(pos-obs)<OBSTACLE_RADIUS:
                    collision_penalty+=1

            for idx,goal in enumerate(goals):
                if np.linalg.norm(pos-goal)<0.5 and idx not in visited_goals:
                    goal_reward += 10
                    visited_goals.add(idx)

            self.positions.append(pos.copy())
            other_agents_positions.append(pos.copy())

            if fine_tune:
                reward = goal_reward - 5*collision_penalty - np.linalg.norm(pos-goals[0])
                loss = -torch.tensor(reward, dtype=torch.float32)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.fitness = goal_reward - 5*collision_penalty - np.linalg.norm(pos-goals[0])

# ----------------------------
# GA Functions
# ----------------------------
def selection(population):
    selected=[]
    for _ in range(len(population)):
        a,b=random.sample(population,2)
        selected.append(a if a.fitness>b.fitness else b)
    return selected

def crossover(parent1,parent2):
    g1=parent1.get_genome()
    g2=parent2.get_genome()
    point=random.randint(0,len(g1)-1)
    return Agent(np.concatenate([g1[:point], g2[point:]])), Agent(np.concatenate([g2[:point], g1[point:]]))

def mutate(agent):
    genome=agent.get_genome()
    for i in range(len(genome)):
        if random.random()<MUTATION_RATE:
            genome[i]+=np.random.normal()
    agent.set_genome(genome)

# ----------------------------
# Dynamic Environment: Random goals/obstacles
# ----------------------------
def generate_dynamic_environment():
    goals = [np.random.uniform(0,6,2) for _ in range(random.randint(1,3))]
    obstacles = [np.random.uniform(0,6,2) for _ in range(random.randint(1,3))]
    obstacle_velocities = [np.random.uniform(-0.05,0.05,2) for _ in obstacles]
    return goals, obstacles, obstacle_velocities

# ----------------------------
# Main GA + RL Loop
# ----------------------------
def run_ga():
    population=[Agent() for _ in range(POPULATION_SIZE)]
    all_paths=[]

    for gen in range(GENERATIONS):
        goals, obstacles, obstacle_velocities = generate_dynamic_environment()
        for step in range(STEPS):
            for i, vel in enumerate(obstacle_velocities):
                obstacles[i] += vel

        for agent in population:
            agent.evaluate(goals, obstacles, [], fine_tune=True)

        best_agent=max(population, key=lambda a: a.fitness)
        print(f"Generation {gen}: Best Fitness={best_agent.fitness:.2f}")

        all_paths.append([agent.positions for agent in population])

        selected=selection(population)
        next_population=[]
        for i in range(0,POPULATION_SIZE,2):
            child1,child2=crossover(selected[i], selected[i+1])
            next_population.extend([child1, child2])
        for agent in next_population: mutate(agent)
        population = next_population

    return all_paths

# ----------------------------
# Main Function
# ----------------------------
def main():
    fig, ax=plt.subplots()
    ax.set_xlim(-1,7)
    ax.set_ylim(-1,7)

    all_paths = run_ga()

    agent_plots=[ax.plot([],[],"bo")[0] for _ in range(POPULATION_SIZE)]

    frames=[]
    for generation_paths in all_paths:
        for step in range(STEPS+1):
            frames.append([paths[step] for paths in generation_paths])

    def animate(i):
        for j, agent_plot in enumerate(agent_plots):
            agent_plot.set_data(frames[i][j][0], frames[i][j][1])
        return agent_plots

    ani=FuncAnimation(fig, animate, frames=len(frames), interval=200, blit=True)
    plt.show()

if __name__=="__main__":
    main()
