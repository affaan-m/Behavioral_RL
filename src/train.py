import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
import torch
import torch.nn as nn
from collections import defaultdict
from igt_env import IGTEnvironment, BaselineIGTEnvironment
import json
from datetime import datetime
from save_results import save_training_results

# Register the custom environment
gym.register(
    id='IGT-v0',
    entry_point='src.igt_env:IGTEnvironment',
)

class EnhancedWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.deck_choices = []
        self.current_episode_decks = []
        self.deck_rewards = defaultdict(list)
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        self.current_episode_reward += reward
        
        # Get the action (deck choice) from the environment
        info = self.locals["infos"][0]
        if "deck_choice" in info:
            deck = info["deck_choice"]
            self.current_episode_decks.append(deck)
            self.deck_rewards[deck].append(reward)
        
        if self.locals["dones"][0]:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.deck_choices.extend(self.current_episode_decks)
            
            # Calculate deck statistics
            deck_counts = defaultdict(int)
            for deck in self.current_episode_decks:
                deck_counts[deck] += 1
            
            # Calculate deck preferences
            total_choices = len(self.current_episode_decks)
            deck_preferences = {
                f"deck_{deck}_preference": count/total_choices 
                for deck, count in deck_counts.items()
            }
            
            # Calculate mean rewards per deck
            deck_mean_rewards = {
                f"deck_{deck}_mean_reward": np.mean(rewards[-100:]) if rewards else 0
                for deck, rewards in self.deck_rewards.items()
            }
            
            # Calculate advantageous ratio (C+D choices)
            advantageous_choices = sum(1 for d in self.current_episode_decks if d in ['C', 'D'])
            advantageous_ratio = advantageous_choices / total_choices if total_choices > 0 else 0
            
            # Log metrics
            wandb.log({
                "episode": self.episode_count,
                "episode_reward": self.current_episode_reward,
                "mean_reward": np.mean(self.episode_rewards[-100:]),
                "advantageous_ratio": advantageous_ratio,
                "steps": self.num_timesteps,
                **deck_preferences,
                **deck_mean_rewards
            })
            
            # Reset episode tracking
            self.current_episode_reward = 0
            self.current_episode_decks = []
            
        return True
    
    def on_training_end(self):
        """Print final statistics"""
        print("\nFinal Statistics:")
        print("-" * 50)
        
        # Overall performance
        final_mean_reward = np.mean(self.episode_rewards[-100:])
        print(f"Final Mean Reward: {final_mean_reward:.2f}")
        
        # Per-deck statistics
        for deck in ['A', 'B', 'C', 'D']:
            deck_rewards = self.deck_rewards[deck]
            if deck_rewards:
                mean_reward = np.mean(deck_rewards[-100:])
                choice_freq = self.deck_choices.count(deck) / len(self.deck_choices)
                print(f"\nDeck {deck}:")
                print(f"  Mean Reward: {mean_reward:.2f}")
                print(f"  Choice Frequency: {choice_freq:.2%}")
        
        # Advantageous choices
        advantageous = sum(1 for d in self.deck_choices[-100:] if d in ['C', 'D'])
        print(f"\nFinal Advantageous Ratio: {advantageous/100:.2%}")
        print("-" * 50)

def setup_env(env_id):
    """Create and wrap the environment"""
    env = gym.make(env_id)
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env)
    return env

def save_training_data(model_type, metrics):
    """Save training data for dashboard visualization"""
    os.makedirs("data/training_results", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/training_results/{model_type}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Saved training data to {filename}")

def evaluate_model(model, env, n_eval_episodes=10):
    """Evaluate a trained model"""
    episode_rewards = []
    deck_choices = []
    
    for _ in range(n_eval_episodes):
        obs = env.reset()[0]
        done = False
        episode_reward = 0
        episode_decks = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = action.item()
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward
            episode_decks.append(info['deck_choice'])
            if done:
                break
        
        episode_rewards.append(episode_reward)
        deck_choices.extend(episode_decks)
    
    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    deck_counts = {deck: deck_choices.count(deck) for deck in 'ABCD'}
    total_choices = len(deck_choices)
    deck_frequencies = {deck: count/total_choices * 100 for deck, count in deck_counts.items()}
    
    # Calculate advantageous ratio (C+D vs A+B)
    advantageous_choices = deck_frequencies['C'] + deck_frequencies['D']
    disadvantageous_choices = deck_frequencies['A'] + deck_frequencies['B']
    
    print("\nFinal Statistics:")
    print("-" * 50)
    print(f"Final Mean Reward: {mean_reward:.2f}\n")
    
    for deck in 'ABCD':
        print(f"Deck {deck}:")
        print(f"  Choice Frequency: {deck_frequencies[deck]:.2f}%")
    
    print(f"\nAdvantageous Choices (C+D): {advantageous_choices:.2f}%")
    print(f"Disadvantageous Choices (A+B): {disadvantageous_choices:.2f}%")
    print("-" * 50)
    
    return {
        'mean_reward': mean_reward,
        'deck_frequencies': deck_frequencies,
        'advantageous_ratio': advantageous_choices
    }

def train(model_type="baseline"):
    """Train RL model on IGT
    Args:
        model_type: Either 'baseline' or 'risk_sensitive'
    """
    # Initialize wandb
    run = wandb.init(
        project="behavioral_rl",
        config={
            "model_type": model_type,
            "algorithm": "PPO",
            "learning_rate": 3e-4,        # Standard PPO learning rate
            "batch_size": 256,            # Smaller batch for IGT
            "n_steps": 2048,              # Standard PPO steps
            "gamma": 0.99,                # Standard discount
            "gae_lambda": 0.95,           # Standard GAE
            "n_epochs": 10,               # Standard PPO epochs
            "ent_coef": 0.01,             # Encourage exploration
            "vf_coef": 0.5,               # Standard value coefficient
            "max_grad_norm": 0.5,         # Standard gradient clipping
            "policy_kwargs": {
                "net_arch": [64, 64],     # Simpler network for IGT
                "activation_fn": nn.Tanh   # Tanh for better bounded values
            }
        }
    )
    
    # Create environment
    env_class = BaselineIGTEnvironment if model_type == "baseline" else IGTEnvironment
    env = env_class()
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env)
    
    # Initialize model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        batch_size=256,
        n_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=10,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device='cpu',
        policy_kwargs=dict(
            net_arch=[64, 64],
            activation_fn=nn.Tanh
        ),
        verbose=1
    )
    
    # Train model
    callback = EnhancedWandbCallback()
    model.learn(
        total_timesteps=100000,  # 1000 episodes of 100 steps each
        callback=callback,
        progress_bar=True
    )
    
    # Save model and training data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/igt_model_{model_type}_{timestamp}.zip"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate final performance
    metrics = evaluate_model(model, env)
    
    # Save training data
    training_data = {
        'model_type': model_type,
        'timestamp': timestamp,
        'metrics': metrics,
        'config': wandb.config._items
    }
    
    data_path = f"results/training_data_{model_type}_{timestamp}.json"
    with open(data_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    print(f"Training data saved to {data_path}")
    
    wandb.finish()
    
    # Save training results
    results = {
        'mean_rewards': episode_rewards,
        'deck_frequencies': deck_frequencies
    }
    save_training_results(args.model_type, results)
    
    return model

if __name__ == "__main__":
    train() 