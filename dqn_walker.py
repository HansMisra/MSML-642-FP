import numpy as np
import torch
import random
import gym
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import pickle
import math
import os
import matplotlib.pyplot as plt

# Environment and Device
ENV = "BipedalWalker-v3"
MODEL_FILE = "/content/drive/MyDrive/MSML-642/DQN_2D/BipedalWalker/DQN/dqn_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# Hyperparameters
N_GAMES = 20000  # Increase to 20,000
MEM_SIZE = 1000000
BATCH_SIZE = 128  # Increase to speed up training
TARGET_UPDATE = 10  # Reduce frequency of target updates for efficiency
GAMMA = 0.99
EPSILON = 1
EPSILON_DEC = 1e-4  # Adjust epsilon decay for smoother exploration
EPSILON_MIN = 0.05
LR = 1e-4
SAVE_INTERVAL = 500  # Save every 500 games to reduce overhead

steps_taken = 0

# Agent class def
class Agent:
    def __init__(self, state_space, action_space):
        self.memory = ExperienceReplay(MEM_SIZE)
        self.action_space = action_space
        self.main_model = QNetwork(state_space.shape[0], action_space.shape[0], action_space.high[0]).to(DEVICE)
        self.target_model = QNetwork(state_space.shape[0], action_space.shape[0], action_space.high[0]).to(DEVICE)
        self.optimizer = optim.Adam(self.main_model.parameters(), lr=LR)
        self.epsilon = EPSILON  # Initialize epsilon

        # Load the target model with main model parameters
        self.target_model.load_state_dict(self.main_model.state_dict())
        self.target_model.eval()

    def step(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        if len(self.memory) > BATCH_SIZE:
            self.learn()

    def learn(self):
        state, action, reward, new_state, done = self.memory.sample()

        #state = np.array(state)
        q_eval = self.main_model(state)
        q_next = self.target_model(new_state)
        q_target = reward + GAMMA * (q_next) * (1 - done)
        loss = F.mse_loss(q_eval, q_target.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.main_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def choose_action(self, state):
        global steps_taken
        eps_threshold = EPSILON_MIN + (self.epsilon - EPSILON_MIN) * math.exp(-1 * steps_taken / EPSILON_DEC)
        steps_taken += 1
        #state = np.array(state)

        if type(state) == tuple:
          state = state[0]

        if np.random.random() < eps_threshold:
            return torch.from_numpy(self.action_space.sample())
        else:
            state = torch.FloatTensor(state.reshape(1, -1)).to(DEVICE)
            with torch.no_grad():
                return self.main_model(state).flatten().cpu().data

# Experience Replay Buffer, QNetwork class definitions (add these above the Agent class)
class ExperienceReplay:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def store_transition(self, state, action, reward, new_state, done):
        self.buffer.append((state, action, reward, new_state, done))


    def sample(self):
        sample = random.sample(self.buffer, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*sample)
        states = torch.tensor(states).float().to(DEVICE)
        actions = torch.stack(actions).long().to(DEVICE)
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32).reshape(-1, 1)).to(DEVICE)
        next_states = torch.tensor(next_states).float().to(DEVICE)
        dones = torch.from_numpy(np.array(dones, dtype=np.uint8).reshape(-1, 1)).float().to(DEVICE)
        return (states, actions, rewards, next_states, dones)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(QNetwork, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = self.l1(state)
        x = F.relu(x)
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x

# Other utility functions
def save_model(agent, path, scores):
    torch.save({
        'model_state_dict': agent.main_model.state_dict(),
        'target_state_dict': agent.target_model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'scores': scores,
        'steps_taken': steps_taken
    }, path)

def load_model(agent, path):
    checkpoint = torch.load(path)
    agent.main_model.load_state_dict(checkpoint['model_state_dict'])
    agent.target_model.load_state_dict(checkpoint['target_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.epsilon = checkpoint['epsilon']
    return checkpoint['scores'], checkpoint['steps_taken']

def plot_scores(scores):
    plt.plot(scores)
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.title('Training Progress')
    plt.grid()
    plt.show()

def main():
    #env = gym.make(ENV)
    env = BipedalWalkerWithObject(render_mode="human")
    env.reset()
    state_space = env.observation_space
    action_space = env.action_space
    agent = Agent(state_space, action_space)

    scores = []
    max_score = -10000
    max_game = 0
    start = datetime.datetime.now()

    # Load checkpoint if available
    load = input("\nload from model? [y/n]: ")
    if load == "y":
        load_path = input("\npath to load from: ")
        if os.path.exists(load_path):
            scores, steps_taken = load_model(agent, load_path)
            print(f"Model loaded from {load_path}. Resuming training.")
        else:
            print(f"No valid checkpoint found at {load_path}. Starting from scratch.")

    for game in range(N_GAMES):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            # Disable rendering for training to improve speed
            if game % 500 == 0:
                #env.render()
                pass

            action = agent.choose_action(observation)
            next_observation, reward, done, extra, _ = env.step(action)
            agent.step(observation, action, reward, next_observation, done)
            score += float(reward)
            observation = next_observation

        # Update target model every TARGET_UPDATE games
        if game % TARGET_UPDATE == 0:
            agent.target_model.load_state_dict(agent.main_model.state_dict())

        # Save checkpoint every SAVE_INTERVAL games
        if game % SAVE_INTERVAL == 0:
            save_path = f"{MODEL_FILE}_checkpoint_{game}.pth"
            save_model(agent, save_path, scores)
            print(f"Checkpoint saved at {save_path}")

        # Update max score if current score is better
        if score > max_score:
            max_score = score
            max_game = game
            save_model(agent, MODEL_FILE, scores)
            print(f"Best model saved with score {max_score} at game {game}")

        # Append the current score and print progress
        scores.append(score)
        avg_score = np.mean(scores[-100:])  # Average score of the last 100 games

        print(f"Game: {game}, Reward: {score:.2f}, Max Reward: {max_score:.2f} (at Game {max_game}), "
              f"Avg Score (last 100): {avg_score:.2f}")

    # Final save after training
    save_model(agent, MODEL_FILE, scores)

    # Plotting the scores to visualize learning
    plot_scores(scores)

    # Print total elapsed time
    end = datetime.datetime.now()
    elapsed = end - start
    print('Total time:', elapsed.total_seconds(), 'seconds')

if __name__ == "__main__":
    main()
