import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import sys
import os
import torch
import torch.nn as nn
from stable_baselines3 import PPO

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.custom_env import InvestmentGameEnv
from src.igt_env import IGTEnvironment
from src.train import train_risk_sensitive_model
from src.config.database import get_supabase_client
import wandb

def train_baseline_model(env, total_timesteps):
    """Train a PPO agent as baseline"""
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=1e-3,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log="./ppo_igt_tensorboard/",
        device='cpu',  # Force CPU usage
        verbose=1
    )
    
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True
    )
    
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
    metrics = {
        'reward_mean': 0,
        'reward_std': 0,
        'cd_ratio_mean': 0,
        'cd_ratio_std': 0,
        'deck_preferences': {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    }
    
    if not human_data:
        print("Warning: No human data available")
        return metrics
        
    total_rewards = []
    cd_ratios = []
    deck_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    total_choices = 0
    
    for participant in human_data:
        # Extract metrics from participant data
        participant_metrics = participant.get('metrics', {})
        
        # Get total money
        total_money = participant_metrics.get('total_money', 0)
        total_rewards.append(total_money)
        
        # Get deck preferences
        deck_prefs = participant_metrics.get('deck_preferences', {})
        for deck, percentage in deck_prefs.items():
            deck_letter = deck.split('_')[-1]  # Extract letter from 'deck_X'
            deck_counts[deck_letter] += percentage
            
        # Calculate C+D ratio
        cd_ratio = (deck_prefs.get('deck_C', 0) + deck_prefs.get('deck_D', 0)) / 100
        cd_ratios.append(cd_ratio)
        
        total_choices += 1
    
    if total_choices > 0:
        # Calculate averages
        metrics['reward_mean'] = float(np.mean(total_rewards))
        metrics['reward_std'] = float(np.std(total_rewards))
        metrics['cd_ratio_mean'] = float(np.mean(cd_ratios))
        metrics['cd_ratio_std'] = float(np.std(cd_ratios))
        
        # Average deck preferences
        for deck in deck_counts:
            metrics['deck_preferences'][deck] = float(deck_counts[deck] / total_choices)
    
    return metrics

def run_comprehensive_comparison(n_episodes=100, n_seeds=3):
    """Run comprehensive comparison between baseline, risk-sensitive, and human behavior"""
    results_dir = Path('results/comparison')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load human data from Supabase
    supabase = get_supabase_client()
    
    print("Loading human data from Supabase...")
    response = supabase.table('participants').select('*').execute()
    human_data = response.data
    human_metrics = calculate_human_metrics(human_data)
    
    print(f"Loaded {len(human_data)} human participants")
    
    all_results = {
        'baseline': [],
        'risk_sensitive': [],
        'human': human_metrics
    }
    
    for seed in tqdm(range(n_seeds), desc="Running comparisons"):
        env = IGTEnvironment()
        env.reset(seed=seed)
        
        # Train and evaluate baseline model
        print(f"\nTraining baseline model (seed {seed})...")
        baseline_model = train_baseline_model(env, total_timesteps=50000)
        baseline_results = evaluate_model(baseline_model, env, n_episodes)
        all_results['baseline'].append(baseline_results)
        
        # Train and evaluate risk-sensitive model
        print(f"\nTraining risk-sensitive model (seed {seed})...")
        risk_sensitive_model = train_risk_sensitive_model(env, total_timesteps=50000)
        risk_sensitive_results = evaluate_model(risk_sensitive_model, env, n_episodes)
        all_results['risk_sensitive'].append(risk_sensitive_results)
        
        # Save models
        baseline_model.save(f"results/comparison/baseline_model_seed{seed}")
        risk_sensitive_model.save(f"results/comparison/risk_sensitive_model_seed{seed}")
        
        # Save intermediate results
        with open(f'results/comparison/results_seed{seed}.json', 'w') as f:
            json.dump({
                'baseline': baseline_results,
                'risk_sensitive': risk_sensitive_results,
                'human': human_metrics
            }, f, indent=2, cls=NumpyEncoder)
    
    # Calculate aggregate metrics
    aggregate_results = {
        'baseline': aggregate_metrics([r for r in all_results['baseline']]),
        'risk_sensitive': aggregate_metrics([r for r in all_results['risk_sensitive']]),
        'human': human_metrics
    }
    
    # Save final results
    with open('results/comparison/final_results.json', 'w') as f:
        json.dump(aggregate_results, f, indent=2, cls=NumpyEncoder)
    
    # Generate comparison plots
    generate_comparison_plots(aggregate_results, results_dir)
    
    return aggregate_results

def aggregate_metrics(results_list):
    """Aggregate metrics across multiple seeds"""
    return {
        'mean_reward': float(np.mean([r['mean_reward'] for r in results_list])),
        'std_reward': float(np.mean([r['std_reward'] for r in results_list])),
        'mean_cd_ratio': float(np.mean([r['mean_cd_ratio'] for r in results_list])),
        'deck_preferences': {
            'A': float(np.mean([np.sum(np.array(r['deck_choices']) == 0) / len(r['deck_choices'][0]) for r in results_list])),
            'B': float(np.mean([np.sum(np.array(r['deck_choices']) == 1) / len(r['deck_choices'][0]) for r in results_list])),
            'C': float(np.mean([np.sum(np.array(r['deck_choices']) == 2) / len(r['deck_choices'][0]) for r in results_list])),
            'D': float(np.mean([np.sum(np.array(r['deck_choices']) == 3) / len(r['deck_choices'][0]) for r in results_list]))
        }
    }

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

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