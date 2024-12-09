import argparse
import json
import os
from datetime import datetime
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from igt_env import IGTEnvironment

def main():
    """Main training function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['baseline', 'risk_sensitive'], required=True)
    parser.add_argument('--trials', type=int, default=200)
    args = parser.parse_args()
    
    # Get environment parameters
    env_params = IGTEnvironment.get_env_parameters()
    
    # Create environment
    env = IGTEnvironment(max_steps=args.trials)
    
    if args.model == 'baseline':
        print("\nTraining baseline model...")
        model = train_baseline_model(env)
    else:
        print("\nTraining risk-sensitive model...")
        model = train_risk_sensitive_model(env)
    
    # Evaluate model
    results = evaluate_model(model, env, n_episodes=10)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/training_results/{args.model}_{timestamp}.json"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nModel metrics:")
    print(json.dumps(results, indent=2))
    print(f"\nSaved training data to {output_file}")

def train_risk_sensitive_model(env):
    """Train risk-sensitive model with phase-dependent parameters"""
    # Get environment parameters
    params = env.get_env_parameters()
    
    # Custom network architecture
    policy_kwargs = dict(
        net_arch=dict(
            pi=[64, 32],  # Policy network
            vf=[64, 32],  # Value network
            risk=[64, 32]  # Risk network
        )
    )
    
    # Configure model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    
    # Custom callback for phase-dependent behavior
    class PhaseCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.phase = 1
            self.trial_count = 0
            self.phase_boundary = params['learning']['phase_boundary']
        
        def _on_step(self):
            self.trial_count += 1
            if self.trial_count == self.phase_boundary and self.phase == 1:
                self.phase = 2
                # Adjust parameters for second phase
                self.model.ent_coef = 0.005  # Reduce exploration
                self.model.learning_rate = 1e-4  # Slower learning
            return True
    
    # Train model
    model.learn(
        total_timesteps=200000,
        callback=PhaseCallback(),
        progress_bar=True
    )
    
    return model

def evaluate_model(model, env, n_episodes=10):
    """Evaluate model performance"""
    rewards = []
    deck_choices = []
    
    for episode in range(n_episodes):
        obs = env.reset()[0]
        done = False
        episode_reward = 0
        episode_choices = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
            episode_choices.append(action)
        
        rewards.append(episode_reward)
        deck_choices.extend(episode_choices)
    
    # Calculate statistics
    results = {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'mean_total_money': float(np.mean(rewards)) + 2000,
        'deck_preferences': {
            'A': float(deck_choices.count(0) / len(deck_choices)),
            'B': float(deck_choices.count(1) / len(deck_choices)),
            'C': float(deck_choices.count(2) / len(deck_choices)),
            'D': float(deck_choices.count(3) / len(deck_choices))
        },
        'advantageous_ratio': float(
            sum(1 for x in deck_choices if x in [2,3]) / len(deck_choices)
        ),
        'first_100_stats': {
            'cd_ratio': float(
                sum(1 for x in deck_choices[:100] if x in [2,3]) / 100
            )
        },
        'second_100_stats': {
            'cd_ratio': float(
                sum(1 for x in deck_choices[100:200] if x in [2,3]) / 100
            )
        }
    }
    
    return results

if __name__ == "__main__":
    main() 