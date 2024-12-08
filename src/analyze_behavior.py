import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from scipy import stats
from datetime import datetime

def load_human_data(data_dir='results/human_data'):
    """Load all human gameplay data"""
    human_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            with open(os.path.join(data_dir, filename), 'r') as f:
                data = json.load(f)
                human_data.append(data)
    return human_data

def load_ai_data(data_dir='results/metrics'):
    """Load AI experiment results"""
    baseline_data = []
    risk_sensitive_data = []
    
    for filename in os.listdir(data_dir):
        if filename.startswith('detailed_results_') and filename.endswith('.json'):
            with open(os.path.join(data_dir, filename), 'r') as f:
                data = json.load(f)
                baseline_data.append(data['baseline'])
                risk_sensitive_data.append(data['risk_sensitive'])
    
    return baseline_data, risk_sensitive_data

def calculate_behavioral_metrics(data, is_human=False):
    """Calculate key behavioral metrics"""
    if is_human:
        # For human data
        metrics = {
            'final_capitals': [d['metrics']['final_capital'] for d in data],
            'risk_seeking_ratios': [d['metrics']['risk_seeking_ratio'] for d in data],
            'avg_investments': [d['metrics']['avg_investment'] for d in data],
            'investment_after_gains': [d['metrics']['investment_after_gains'] for d in data],
            'investment_after_losses': [d['metrics']['investment_after_losses'] for d in data]
        }
    else:
        # For AI data
        metrics = {
            'final_capitals': [d['final_capitals'] for d in data],
            'risk_seeking_ratios': [d['risk_seeking_ratio'] for d in data],
            'avg_investments': np.mean([d['investment_props'] for d in data], axis=1),
            'investment_after_gains': [d['mean_investment_after_gains'] for d in data],
            'investment_after_losses': [d['mean_investment_after_losses'] for d in data]
        }
    
    return metrics

def plot_behavioral_comparison(human_metrics, baseline_metrics, risk_sensitive_metrics):
    """Create comparison plots between human and AI behavior"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Behavioral Comparison: Human vs AI', fontsize=16)
    
    # Plot 1: Final Capital Distribution
    sns.boxplot(data=[human_metrics['final_capitals'],
                     baseline_metrics['final_capitals'],
                     risk_sensitive_metrics['final_capitals']], 
                ax=axes[0,0])
    axes[0,0].set_xticklabels(['Human', 'Baseline AI', 'Risk-Sensitive AI'])
    axes[0,0].set_title('Final Capital Distribution')
    
    # Plot 2: Risk-Seeking Ratio
    sns.boxplot(data=[human_metrics['risk_seeking_ratios'],
                     baseline_metrics['risk_seeking_ratios'],
                     risk_sensitive_metrics['risk_seeking_ratios']], 
                ax=axes[0,1])
    axes[0,1].set_xticklabels(['Human', 'Baseline AI', 'Risk-Sensitive AI'])
    axes[0,1].set_title('Risk-Seeking Ratio')
    
    # Plot 3: Investment After Gains vs Losses
    data = {
        'Condition': ['After Gains']*3 + ['After Losses']*3,
        'Agent': ['Human', 'Baseline AI', 'Risk-Sensitive AI']*2,
        'Investment': [
            np.mean(human_metrics['investment_after_gains']),
            np.mean(baseline_metrics['investment_after_gains']),
            np.mean(risk_sensitive_metrics['investment_after_gains']),
            np.mean(human_metrics['investment_after_losses']),
            np.mean(baseline_metrics['investment_after_losses']),
            np.mean(risk_sensitive_metrics['investment_after_losses'])
        ]
    }
    df = pd.DataFrame(data)
    sns.barplot(data=df, x='Condition', y='Investment', hue='Agent', ax=axes[1,0])
    axes[1,0].set_title('Investment Behavior After Gains/Losses')
    
    # Plot 4: Statistical Analysis
    axes[1,1].axis('off')
    stats_text = []
    
    # Perform statistical tests
    for metric in ['final_capitals', 'risk_seeking_ratios']:
        f_stat, p_val = stats.f_oneway(human_metrics[metric],
                                     baseline_metrics[metric],
                                     risk_sensitive_metrics[metric])
        stats_text.append(f'{metric}:\nF={f_stat:.2f}, p={p_val:.3f}')
        
        # Post-hoc t-tests if ANOVA is significant
        if p_val < 0.05:
            t_stat_h_b, p_val_h_b = stats.ttest_ind(human_metrics[metric],
                                                   baseline_metrics[metric])
            t_stat_h_r, p_val_h_r = stats.ttest_ind(human_metrics[metric],
                                                   risk_sensitive_metrics[metric])
            stats_text.append(f'Human vs Baseline: p={p_val_h_b:.3f}')
            stats_text.append(f'Human vs Risk-Sensitive: p={p_val_h_r:.3f}')
    
    axes[1,1].text(0.1, 0.5, '\n'.join(stats_text), fontsize=10)
    
    plt.tight_layout()
    return fig

def calculate_similarity_scores(human_metrics, baseline_metrics, risk_sensitive_metrics):
    """Calculate similarity scores between human and AI behavior"""
    def calculate_score(human_vals, ai_vals, metric):
        # Normalize values
        human_vals = np.array(human_vals)
        ai_vals = np.array(ai_vals)
        
        # Calculate various similarity metrics
        mean_diff = np.abs(np.mean(human_vals) - np.mean(ai_vals))
        dist_overlap = 1 - stats.wasserstein_distance(
            human_vals/np.std(human_vals),
            ai_vals/np.std(ai_vals)
        )
        correlation = stats.pearsonr(
            human_vals[:min(len(human_vals), len(ai_vals))],
            ai_vals[:min(len(human_vals), len(ai_vals))]
        )[0]
        
        return {
            'mean_difference': mean_diff,
            'distribution_overlap': dist_overlap,
            'correlation': correlation
        }
    
    metrics_to_compare = ['final_capitals', 'risk_seeking_ratios', 
                         'investment_after_gains', 'investment_after_losses']
    
    similarity_scores = {
        'baseline': {},
        'risk_sensitive': {}
    }
    
    for metric in metrics_to_compare:
        similarity_scores['baseline'][metric] = calculate_score(
            human_metrics[metric], baseline_metrics[metric], metric)
        similarity_scores['risk_sensitive'][metric] = calculate_score(
            human_metrics[metric], risk_sensitive_metrics[metric], metric)
    
    return similarity_scores

def generate_report(human_metrics, baseline_metrics, risk_sensitive_metrics, 
                   similarity_scores):
    """Generate a comprehensive analysis report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f'results/analysis/behavioral_analysis_{timestamp}.txt'
    
    os.makedirs('results/analysis', exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write("Behavioral Analysis Report\n")
        f.write("=========================\n\n")
        
        # Basic statistics
        f.write("1. Basic Statistics\n")
        f.write("-----------------\n")
        agents = ['Human', 'Baseline AI', 'Risk-Sensitive AI']
        metrics = [human_metrics, baseline_metrics, risk_sensitive_metrics]
        
        for metric in ['final_capitals', 'risk_seeking_ratios']:
            f.write(f"\n{metric.replace('_', ' ').title()}:\n")
            for agent, m in zip(agents, metrics):
                f.write(f"{agent}: {np.mean(m[metric]):.2f} ± {np.std(m[metric]):.2f}\n")
        
        # Similarity Analysis
        f.write("\n2. Similarity Analysis\n")
        f.write("--------------------\n")
        
        for ai_type in ['baseline', 'risk_sensitive']:
            f.write(f"\n{ai_type.replace('_', ' ').title()} AI vs Human:\n")
            for metric, scores in similarity_scores[ai_type].items():
                f.write(f"\n{metric.replace('_', ' ').title()}:\n")
                for score_type, value in scores.items():
                    f.write(f"- {score_type.replace('_', ' ').title()}: {value:.3f}\n")
        
        # Behavioral Patterns
        f.write("\n3. Behavioral Patterns\n")
        f.write("--------------------\n")
        
        for agent, m in zip(agents, metrics):
            f.write(f"\n{agent}:\n")
            f.write(f"- Investment after gains: {np.mean(m['investment_after_gains']):.2%}\n")
            f.write(f"- Investment after losses: {np.mean(m['investment_after_losses']):.2%}\n")
            f.write(f"- Risk-seeking ratio: {np.mean(m['risk_seeking_ratios']):.2f}\n")
    
    return report_file

def main():
    # Load data
    human_data = load_human_data()
    baseline_data, risk_sensitive_data = load_ai_data()
    
    # Calculate metrics
    human_metrics = calculate_behavioral_metrics(human_data, is_human=True)
    baseline_metrics = calculate_behavioral_metrics(baseline_data)
    risk_sensitive_metrics = calculate_behavioral_metrics(risk_sensitive_data)
    
    # Generate plots
    fig = plot_behavioral_comparison(human_metrics, baseline_metrics, risk_sensitive_metrics)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('results/figures', exist_ok=True)
    fig.savefig(f'results/figures/behavioral_comparison_{timestamp}.png')
    
    # Calculate similarity scores
    similarity_scores = calculate_similarity_scores(human_metrics, baseline_metrics, 
                                                 risk_sensitive_metrics)
    
    # Generate report
    report_file = generate_report(human_metrics, baseline_metrics, risk_sensitive_metrics,
                                similarity_scores)
    
    print(f"Analysis complete. Report saved to {report_file}")

if __name__ == "__main__":
    main() 