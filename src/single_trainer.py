import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
from datetime import datetime
from config import Config
from igt_env import IGTEnvironment
import gymnasium as gym

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # Enhanced network architecture with residual connections
        self.input_layer = nn.Linear(11, 512)
        self.ln1 = nn.LayerNorm(512)
        self.dropout1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(512, 512)
        self.ln2 = nn.LayerNorm(512)
        self.dropout2 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(512, 256)
        self.ln3 = nn.LayerNorm(256)
        self.dropout3 = nn.Dropout(0.3)
        
        # Separate streams for value and risk
        self.value_stream = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.risk_stream = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x):
        # Ensure input tensor has correct shape
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        # Main network with residual connections
        identity = x
        x = F.relu(self.ln1(self.input_layer(x)))
        x = self.dropout1(x)
        
        identity2 = x
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x + identity2)  # Residual connection
        
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.dropout3(x)
        
        # Triple stream architecture
        value = self.value_stream(x)
        risk = self.risk_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine streams with scaled advantage and risk-aware weighting
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        q_values = q_values + 0.1 * (risk - risk.mean(dim=1, keepdim=True))  # Risk-aware adjustment
        
        return q_values

class SingleTrainer:
    def __init__(self):
        self.env = IGTEnvironment()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = Config.EPSILON_START
        
        # Initialize Q-networks
        self.q_network = QNetwork(11, self.env.action_space.n).to(self.device)
        self.target_network = QNetwork(11, self.env.action_space.n).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize optimizer with gradient clipping
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=Config.LEARNING_RATE)
        
        # Initialize replay buffer
        self.replay_buffer = deque(maxlen=Config.REPLAY_BUFFER_SIZE)
        
        # Training metrics
        self.episode_rewards = []
        self.avg_rewards = []
        self.losses = []
        
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging directory and file"""
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join('logs', f'training_{timestamp}.csv')
        self.training_data = []
        
    def decay_epsilon(self, step):
        """Decay epsilon value"""
        self.epsilon = Config.EPSILON_END + (Config.EPSILON_START - Config.EPSILON_END) * \
                      np.exp(-1. * step / Config.EPSILON_DECAY)
        
    def select_action(self, state):
        """Select action using epsilon-greedy policy with smart exploration"""
        if np.random.random() < self.epsilon:
            # Temperature-based exploration
            deck_counts = np.array(self.env.deck_visits) + 1
            temperature = max(1.0, 10.0 * (1.0 - self.epsilon))  # Higher temperature early on
            exploration_probs = np.exp(-deck_counts / temperature)
            exploration_probs = exploration_probs / exploration_probs.sum()
            return np.random.choice(self.env.action_space.n, p=exploration_probs)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            q_values = self.q_network(state)
            # Add small noise to q-values for tie-breaking
            noise = torch.randn_like(q_values) * 0.01
            return (q_values + noise).argmax().item()
    
    def update_networks(self, batch_size=32):
        """Update Q-networks using experience replay with PER and n-step returns"""
        if len(self.replay_buffer) < Config.MIN_REPLAY_SIZE:
            return
        
        # Sample batch with prioritized experience replay
        batch = random.sample(self.replay_buffer, batch_size)
        
        # Convert batch to numpy arrays first for efficiency
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        
        # Convert to tensors efficiently
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Double Q-learning with n-step returns and risk-aware targets
        with torch.no_grad():
            # Get next actions from online network
            next_q_values = self.q_network(next_states)
            next_actions = next_q_values.argmax(1).unsqueeze(1)
            
            # Get target values from target network
            next_target_q_values = self.target_network(next_states)
            next_target_values = next_target_q_values.gather(1, next_actions).squeeze()
            
            # Calculate risk-aware target with CVaR and deck balancing
            sorted_values, _ = torch.sort(next_target_values)
            cvar_threshold = int(len(sorted_values) * Config.CVAR_ALPHA)
            cvar_values = sorted_values[:cvar_threshold]
            risk_adjusted_value = cvar_values.mean()
            
            # Calculate deck visit penalties for balancing
            deck_visits = torch.zeros(4).to(self.device)
            for action in actions:
                deck_visits[action] += 1
            deck_penalties = (deck_visits / deck_visits.sum()) * 0.2  # Penalty factor
            
            # Dynamic n-step returns with deck balancing
            n_step = min(4, self.env.max_steps - self.env.steps_taken)
            target_q_values = rewards + (1 - dones) * (Config.GAMMA ** n_step) * (
                0.7 * next_target_values + 
                0.2 * risk_adjusted_value - 
                deck_penalties[actions]  # Apply deck-specific penalties
            )
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Huber loss with gradient clipping
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        
        # Add L2 regularization and deck balance regularization
        l2_lambda = 1e-5
        balance_lambda = 0.1
        l2_reg = torch.tensor(0.).to(self.device)
        for param in self.q_network.parameters():
            l2_reg += torch.norm(param)
        
        # Add deck balance regularization
        q_values = self.q_network(states)
        deck_probs = F.softmax(q_values, dim=1)
        balance_loss = -torch.mean(torch.log(deck_probs + 1e-10)) * balance_lambda
        
        loss += l2_lambda * l2_reg + balance_loss
        
        # Optimize with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 0.5)
        self.optimizer.step()
        
        # Soft target network update
        tau = 0.001
        for target_param, online_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
        
        return loss.item()
    
    def log_episode(self, episode, total_reward, info):
        """Log episode data"""
        # Calculate additional metrics
        deck_picks = info['deck_picks']
        total_picks = sum(deck_picks)
        cd_ratio = (deck_picks[2] + deck_picks[3]) / total_picks if total_picks > 0 else 0
        ab_ratio = (deck_picks[0] + deck_picks[1]) / total_picks if total_picks > 0 else 0
        
        # Store metrics
        self.training_data.append({
            'episode': episode,
            'total_money': info['total_money'],
            'deck_picks': info['deck_picks'],
            'epsilon': self.epsilon,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'cd_ratio': cd_ratio,
            'ab_ratio': ab_ratio,
            'deck_A_ratio': deck_picks[0] / total_picks if total_picks > 0 else 0,
            'deck_B_ratio': deck_picks[1] / total_picks if total_picks > 0 else 0,
            'deck_C_ratio': deck_picks[2] / total_picks if total_picks > 0 else 0,
            'deck_D_ratio': deck_picks[3] / total_picks if total_picks > 0 else 0,
            'avg_reward_100': np.mean([d['total_money'] for d in self.training_data[-100:]]) if self.training_data else info['total_money']
        })
        
        # Save to CSV periodically
        if episode % 10 == 0:
            df = pd.DataFrame(self.training_data)
            df.to_csv(self.log_file, index=False)
            
            # Also save to training.log for dashboard compatibility
            df.to_csv('training.log', index=False)
            
        # Print progress
        print(f"Episode {episode}, Reward: {total_reward:.2f}, "
              f"Avg Reward (100): {self.training_data[-1]['avg_reward_100']:.2f}, "
              f"Epsilon: {self.epsilon:.2f}")
        print(f"Deck Picks: {info['deck_picks']}, "
              f"C+D Ratio: {cd_ratio:.2f}, "
              f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
    
    def train(self, num_episodes=1000):
        """Train the agent"""
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            episode_buffer = []  # Store episode transitions
            
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _, info = self.env.step(action)
                
                # Store transition
                episode_buffer.append((state, action, reward, next_state, done))
                
                total_reward += reward
                state = next_state
                
                # Decay epsilon
                self.epsilon = max(
                    Config.EPSILON_END,
                    self.epsilon - (Config.EPSILON_START - Config.EPSILON_END) / Config.EPSILON_DECAY
                )
            
            # Add episode transitions to replay buffer
            self.replay_buffer.extend(episode_buffer)
            
            # Perform multiple updates after each episode
            if len(self.replay_buffer) >= Config.MIN_REPLAY_SIZE:
                for _ in range(4):  # Multiple updates per episode
                    self.update_networks(Config.BATCH_SIZE)
            
            # Update target network periodically
            if episode % Config.TARGET_UPDATE_FREQ == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            # Log episode results
            self.episode_rewards.append(total_reward)
            avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else total_reward
            self.avg_rewards.append(avg_reward)
            
            print(f"Episode {episode}, Reward: {reward:.2f}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.2f}")
            print(f"Deck Picks: {info['deck_picks']}, C+D Ratio: {(info['deck_picks'][2] + info['deck_picks'][3])/sum(info['deck_picks']):.2f}")

if __name__ == "__main__":
    trainer = SingleTrainer()
    trainer.train() 