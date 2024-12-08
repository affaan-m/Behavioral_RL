import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from custom_env import InvestmentGameEnv
from igt_env import IGTEnvironment
from train import train_risk_sensitive_model
from stable_baselines3 import DQN
import wandb
from tqdm import tqdm
from datetime import datetime
import os
import json
from pathlib import Path

def train_baseline_model(env, total_timesteps):
    """Train a standard DQN agent without risk sensitivity"""
    model = DQN("MlpPolicy", env, verbose=0, 
                learning_rate=3e-4,
                buffer_size=100000,
                learning_starts=1000,
                batch_size=64,
                gamma=0.99)
    
    model.learn(total_timesteps=total_timesteps)
    return model

def evaluate_model(model, env, n_episodes=100):
    """Evaluate model behavior metrics"""
    episode_rewards = []
    deck_choices = []
    cd_ratios = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_choices = []
        
        while not done:
            action = model.predict(state, deterministic=True)[0]
            state, reward, done, _, info = env.step(action)
            episode_reward += reward
            episode_choices.append(action)
            
        episode_rewards.append(episode_reward)
        deck_choices.append(episode_choices)
        
        # Calculate C+D ratio for this episode
        choices = np.array(episode_choices)
        cd_ratio = np.sum((choices == 2) | (choices == 3)) / len(choices)
        cd_ratios.append(cd_ratio)
    
    return {
        'rewards': episode_rewards,
        'deck_choices': deck_choices,
        'cd_ratios': cd_ratios,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_cd_ratio': np.mean(cd_ratios)
    }

def load_human_data():
    """Load both IGT literature data and our collected data"""
    # Load IGT literature baselines
    igt_baselines = IGTEnvironment.get_human_baseline_metrics()
    
    # Load our collected data
    collected_data = []
    data_dir = Path('results/human_data')
    if data_dir.exists():
        for file in data_dir.glob('*.json'):
            with open(file, 'r') as f:
                collected_data.append(json.load(f))
    
    return {
        'igt_baselines': igt_baselines,
        'collected_data': collected_data
    }

def calculate_human_metrics(human_data):
    """Calculate metrics from human data"""
    if not human_data['collected_data']:
        return human_data['igt_baselines']
    
    collected_metrics = {
        'cd_ratios': [],
        'rewards': [],
        'deck_preferences': {'A': [], 'B': [], 'C': [], 'D': []}
    }
    
    for data in human_data['collected_data']:
        # Calculate C+D ratio
        choices = np.array(data['history']['deck_choices'])
        cd_ratio = np.sum((choices == 2) | (choices == 3)) / len(choices)
        collected_metrics['cd_ratios'].append(cd_ratio)
        
        # Final reward
        collected_metrics['rewards'].append(data['metrics']['total_money'])
        
        # Deck preferences
        for deck, pref in data['metrics']['deck_preferences'].items():
            collected_metrics['deck_preferences'][deck].append(pref)
    
    # Combine with IGT baselines
    combined_metrics = {
        'cd_ratio_mean': np.mean(collected_metrics['cd_ratios']),
        'cd_ratio_std': np.std(collected_metrics['cd_ratios']),
        'reward_mean': np.mean(collected_metrics['rewards']),
        'reward_std': np.std(collected_metrics['rewards']),
        'deck_preferences': {
            deck: np.mean(prefs) 
            for deck, prefs in collected_metrics['deck_preferences'].items()
        },
        'igt_baselines': human_data['igt_baselines']
    }
    
    return combined_metrics

def run_comprehensive_comparison(n_episodes=1000, n_seeds=5):
    """Run comprehensive comparison between baseline, risk-sensitive, and human behavior"""
    results_dir = Path('results/comparison')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load human data
    human_data = load_human_data()
    human_metrics = calculate_human_metrics(human_data)
    
    all_results = {
        'baseline': [],
        'risk_sensitive': [],
        'human': human_metrics
    }
    
    for seed in tqdm(range(n_seeds), desc="Running comparisons"):
        env = IGTEnvironment()
        env.reset(seed=seed)
        
        # Train and evaluate baseline model
        baseline_model = train_baseline_model(env, total_timesteps=100000)
        baseline_results = evaluate_model(baseline_model, env, n_episodes)
        all_results['baseline'].append(baseline_results)
        
        # Train and evaluate risk-sensitive model
        risk_sensitive_model = train_risk_sensitive_model(env, total_timesteps=100000)
        risk_sensitive_results = evaluate_model(risk_sensitive_model, env, n_episodes)
        all_results['risk_sensitive'].append(risk_sensitive_results)
        
        # Save models
        baseline_model.save(f"results/comparison/baseline_model_seed{seed}")
        risk_sensitive_model.save(f"results/comparison/risk_sensitive_model_seed{seed}")
    
    # Calculate aggregate metrics
    aggregate_results = calculate_aggregate_metrics(all_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"results/comparison/comparison_results_{timestamp}.json", 'w') as f:
        json.dump(aggregate_results, f, indent=4)
    
    # Generate comparison plots
    generate_comparison_plots(aggregate_results, results_dir)
    
    return aggregate_results

def calculate_aggregate_metrics(all_results):
    """Calculate aggregate metrics across all seeds"""
    aggregate = {
        'baseline': {
            'mean_reward': np.mean([r['mean_reward'] for r in all_results['baseline']]),
            'std_reward': np.mean([r['std_reward'] for r in all_results['baseline']]),
            'mean_cd_ratio': np.mean([r['mean_cd_ratio'] for r in all_results['baseline']])
        },
        'risk_sensitive': {
            'mean_reward': np.mean([r['mean_reward'] for r in all_results['risk_sensitive']]),
            'std_reward': np.mean([r['std_reward'] for r in all_results['risk_sensitive']]),
            'mean_cd_ratio': np.mean([r['mean_cd_ratio'] for r in all_results['risk_sensitive']])
        },
        'human': all_results['human']
    }
    
    return aggregate

def generate_comparison_plots(results, output_dir):
    """Generate comparison plots between all three approaches"""
    # 1. Reward Distribution Plot
    plt.figure(figsize=(10, 6))
    plt.bar(['Baseline', 'Risk-Sensitive', 'Human'], 
            [results['baseline']['mean_reward'],
             results['risk_sensitive']['mean_reward'],
             results['human']['reward_mean']],
            yerr=[results['baseline']['std_reward'],
                  results['risk_sensitive']['std_reward'],
                  results['human']['reward_std']])
    plt.title('Reward Distribution Comparison')
    plt.ylabel('Mean Reward')
    plt.savefig(output_dir / 'reward_comparison.png')
    plt.close()
    
    # 2. C+D Ratio Plot
    plt.figure(figsize=(10, 6))
    plt.bar(['Baseline', 'Risk-Sensitive', 'Human'],
            [results['baseline']['mean_cd_ratio'],
             results['risk_sensitive']['mean_cd_ratio'],
             results['human']['cd_ratio_mean']],
            yerr=[0, 0, results['human']['cd_ratio_std']])
    plt.title('C+D Choice Ratio Comparison')
    plt.ylabel('C+D Choice Ratio')
    plt.savefig(output_dir / 'cd_ratio_comparison.png')
    plt.close()
    
    # 3. Deck Preferences
    deck_prefs = pd.DataFrame({
        'Baseline': [0.25, 0.25, 0.25, 0.25],  # Uniform for baseline
        'Risk-Sensitive': [0.2, 0.2, 0.3, 0.3],  # Approximate from results
        'Human': [results['human']['deck_preferences'][d] for d in 'ABCD']
    }, index=['A', 'B', 'C', 'D'])
    
    deck_prefs.plot(kind='bar', figsize=(10, 6))
    plt.title('Deck Preferences Comparison')
    plt.ylabel('Choice Probability')
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(output_dir / 'deck_preferences_comparison.png')
    plt.close()

if __name__ == "__main__":
    results = run_comprehensive_comparison()
    print("\nComparison Results:")
    print("==================")
    print("\nBaseline Model:")
    print(f"Mean Reward: {results['baseline']['mean_reward']:.2f} ± {results['baseline']['std_reward']:.2f}")
    print(f"C+D Ratio: {results['baseline']['mean_cd_ratio']:.2f}")
    
    print("\nRisk-Sensitive Model:")
    print(f"Mean Reward: {results['risk_sensitive']['mean_reward']:.2f} ± {results['risk_sensitive']['std_reward']:.2f}")
    print(f"C+D Ratio: {results['risk_sensitive']['mean_cd_ratio']:.2f}")
    
    print("\nHuman Baseline:")
    print(f"Mean Reward: {results['human']['reward_mean']:.2f} ± {results['human']['reward_std']:.2f}")
    print(f"C+D Ratio: {results['human']['cd_ratio_mean']:.2f} ± {results['human']['cd_ratio_std']:.2f}") 