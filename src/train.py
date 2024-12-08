import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
from wandb.integration.sb3 import WandbCallback
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class RiskSensitiveQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(RiskSensitiveQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, state):
        return self.network(state)

class RiskSensitiveDQN:
    def __init__(self, env, alpha=0.05, learning_rate=3e-4, gamma=0.99):
        self.env = env
        self.alpha = alpha  # CVaR confidence level
        self.gamma = gamma
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        self.q_network = RiskSensitiveQNetwork(state_dim, action_dim)
        self.target_network = RiskSensitiveQNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
    def compute_cvar_loss(self, rewards, next_q_values):
        # Sort returns in ascending order for CVaR calculation
        sorted_returns = torch.sort(rewards + self.gamma * next_q_values)[0]
        
        # Calculate CVaR as mean of worst alpha% of returns
        cvar_threshold_idx = int(len(sorted_returns) * self.alpha)
        cvar_values = sorted_returns[:cvar_threshold_idx]
        cvar_loss = -torch.mean(cvar_values)  # Negative since we want to maximize
        
        return cvar_loss
        
    def train_step(self, states, actions, rewards, next_states, dones):
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Get current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            next_q_values[dones] = 0.0
            
        # Compute CVaR loss
        cvar_loss = self.compute_cvar_loss(rewards, next_q_values)
        
        # Compute TD error
        expected_q_values = rewards + self.gamma * next_q_values
        td_loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)
        
        # Combine losses
        total_loss = td_loss + cvar_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()

def setup_env(env_id):
    env = gym.make(env_id)
    env = DummyVecEnv([lambda: env])
    return env

def train_risk_sensitive_model(env, total_timesteps, batch_size=64, save_path=None):
    agent = RiskSensitiveDQN(env)
    
    # Initialize wandb logging
    wandb.init(project="risk_sensitive_rl", config={
        "alpha": agent.alpha,
        "gamma": agent.gamma,
        "batch_size": batch_size,
        "total_timesteps": total_timesteps
    })
    
    episode_rewards = []
    losses = []
    
    state, _ = env.reset()
    for t in range(total_timesteps):
        # Collect experience
        action = agent.select_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        
        # Store transition
        agent.memory.push(state, action, reward, next_state, done)
        
        if len(agent.memory) > batch_size:
            loss = agent.train_step(*agent.memory.sample(batch_size))
            losses.append(loss)
            
            # Log metrics
            wandb.log({
                "loss": loss,
                "episode_reward": info.get('episode', {}).get('r', 0),
                "raw_reward": info.get('raw_reward', 0),
                "prospect_reward": info.get('prospect_reward', 0)
            })
        
        if done:
            state, _ = env.reset()
        else:
            state = next_state
            
    if save_path:
        torch.save(agent.q_network.state_dict(), save_path)
    
    return agent

def evaluate(model, env, n_eval_episodes=10):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward

if __name__ == "__main__":
    # Initialize wandb
    wandb.init(project="risk_sensitive_rl", sync_tensorboard=True)
    
    # Setup environment
    env = setup_env("CustomEnv-v0")
    
    # Train model
    model = train_risk_sensitive_model(env, total_timesteps=100000, save_path="./models/risk_sensitive_model.pth")
    
    # Evaluate
    evaluate(model, env)
    
    # Save final model
    model.save("./models/final_model") 