import pandas as pd
import numpy as np
from pathlib import Path

def save_training_results(model_type, results):
    """Save training results to CSV file"""
    output_dir = Path('results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metrics
    episodes = range(len(results['mean_rewards']))
    mean_rewards = results['mean_rewards']
    deck_freqs = results['deck_frequencies']
    
    # Create DataFrame
    df = pd.DataFrame({
        'episode': episodes,
        'mean_reward': mean_rewards,
        'deck_A_freq': [freq['A'] for freq in deck_freqs],
        'deck_B_freq': [freq['B'] for freq in deck_freqs],
        'deck_C_freq': [freq['C'] for freq in deck_freqs],
        'deck_D_freq': [freq['D'] for freq in deck_freqs]
    })
    
    # Save to CSV
    df.to_csv(output_dir / f'{model_type}_results.csv', index=False)
    
if __name__ == "__main__":
    # Example usage
    results = {
        'mean_rewards': [100, 200, 300],
        'deck_frequencies': [
            {'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25},
            {'A': 0.2, 'B': 0.2, 'C': 0.3, 'D': 0.3},
            {'A': 0.1, 'B': 0.1, 'C': 0.4, 'D': 0.4}
        ]
    }
    save_training_results('example', results) 