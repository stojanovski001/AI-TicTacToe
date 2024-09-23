import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from game import TicTacToe
from memory import ReplayMemory
from model import DQN

class Trainer:
    def __init__(self, device, policy_net=None):
        self.device = device
        self.num_episodes = int(os.getenv('NUM_EPISODES', 2500))
        self.gamma = float(os.getenv('GAMMA', 0.99))
        self.epsilon_start = float(os.getenv('EPSILON_START', 1.0))
        self.epsilon_end = float(os.getenv('EPSILON_END', 0.1))
        self.epsilon_decay = float(os.getenv('EPSILON_DECAY', 100000))
        self.learning_rate = float(os.getenv('LEARNING_RATE', 0.0001))
        self.target_update = int(os.getenv('TARGET_UPDATE', 1000))
        self.memory_capacity = int(os.getenv('MEMORY_CAPACITY', 10000))
        self.batch_size = int(os.getenv('BATCH_SIZE', 64))

        if policy_net is None:
            self.policy_net = DQN(device).to(device)
        else:
            self.policy_net = policy_net.to(device)
            self.policy_net.train()  # Set model to training mode

        self.opponent_net = DQN(device).to(device)
        self.opponent_net.load_state_dict(self.policy_net.state_dict())
        self.opponent_net.eval()  # Opponent net in evaluation mode

        self.target_net = DQN(device).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayMemory(self.memory_capacity)

        self.steps_done = 0

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

        batch_state = torch.stack(batch_state)
        batch_action = torch.tensor(batch_action, dtype=torch.long).unsqueeze(1).to(self.device)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float32).to(self.device)
        batch_next_state = torch.stack(batch_next_state)
        batch_done = torch.tensor(batch_done, dtype=torch.float32).to(self.device)

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(batch_state).gather(1, batch_action)

        # Compute V(s_{t+1}) for all next states
        with torch.no_grad():
            next_state_values = self.target_net(batch_next_state).max(1)[0]
            next_state_values = next_state_values * (1 - batch_done)

        # Compute the expected Q values
        expected_state_action_values = batch_reward + (self.gamma * next_state_values)

        # Compute loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(state_action_values.squeeze(), expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        for episode in range(self.num_episodes):
            game = TicTacToe()
            state = torch.FloatTensor(game.reset()).to(self.device)
            done = False

            while not done:
                # Calculate epsilon for this step
                epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                    np.exp(-1. * self.steps_done / self.epsilon_decay)

                print(f"Episode: {episode + 1}/{self.num_episodes}, Epsilon: {epsilon:.2f}")
                # AI's turn (Player 1)
                if random.random() < epsilon:
                    action = random.choice(game.available_actions())
                else:
                    with torch.no_grad():
                        q_values = self.policy_net(state)
                        masked_q_values = q_values.clone()
                        for i in range(9):
                            if i not in game.available_actions():
                                masked_q_values[i] = -float('inf')
                        action = torch.argmax(masked_q_values).item()
                        print(f"Q-values: {q_values}, Masked Q-values: {masked_q_values}, Action: {action}")

                # Take action
                next_state_np, reward, done = game.step(action, 1)
                next_state = torch.FloatTensor(next_state_np).to(self.device)

                if not done:
                    # Stronger opponent strategy
                    with torch.no_grad():
                        q_values = self.opponent_net(next_state)
                        masked_q_values = q_values.clone()
                        for i in range(9):
                            if i not in game.available_actions():
                                masked_q_values[i] = -float('inf')
                        opponent_action = torch.argmax(masked_q_values).item()

                    next_state_np, opponent_reward, done = game.step(opponent_action, -1)
                    next_state = torch.FloatTensor(next_state_np).to(self.device)

                    if opponent_reward == 1:
                        reward = -1  # Penalize if the opponent wins
                    elif opponent_reward == 0:
                        reward = 0  # Draw

                # Store the transition in memory
                self.memory.push((state, action, reward, next_state, done))

                state = next_state
                self.steps_done += 1

                # Perform optimization
                self.optimize_model()

                # Update target network
                if self.steps_done % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            # Update opponent network periodically
            if episode % 1000 == 0 and episode != 0:
                self.opponent_net.load_state_dict(self.policy_net.state_dict())

            # Print progress
            if (episode + 1) % 2500 == 0:
                print(f"Episode {episode + 1}/{self.num_episodes}")

        # Save the trained model
        torch.save(self.policy_net.state_dict(), 'tic_tac_toe_model.pth')
        print("Training complete and model saved.")

        return self.policy_net

    def save_model(self, filepath):
        torch.save(self.policy_net.state_dict(), filepath)