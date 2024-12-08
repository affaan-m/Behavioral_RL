import os
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy import stats

class ModelHumanComparison:
    def __init__(self):
        self.results_dir = Path('results')
        self.metrics_dir = Path('metrics')
        self.human_baselines = self.load_human_baselines()
    
    def load_human_baselines(self):
        """Load or create default human baseline metrics"""
        # Default human baseline values based on literature
        return {
            'advantageous_ratio': 0.7,  # Healthy controls tend to choose C+D 70% of time
            'learning_rate': 0.15,  # Typical learning rate from literature
            'final_money_mean': -500,  # Average final money for healthy controls
            'final_money_std': 1000,  # Standard deviation of final money
            'deck_preferences': {
                'A': 0.2,
                'B': 0.2,
                'C': 0.3,
                'D': 0.3
            }
        }
    
    def load_model_data(self):
        """Load the most recent model training data"""
        try:
            if Path('training.log').exists():
                return pd.read_csv('training.log')
            
            # Try metrics directory
            run_dirs = list(self.metrics_dir.glob('run_*'))
            if not run_dirs:
                return pd.DataFrame()
                
            latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
            csv_files = list(latest_run.glob('*.csv'))
            if not csv_files:
                return pd.DataFrame()
                
            latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
            return pd.read_csv(latest_csv)
            
        except Exception as e:
            print(f"Error loading model data: {str(e)}")
            return pd.DataFrame()
    
    def analyze_all_runs(self):
        """Analyze model performance across all runs"""
        model_data = self.load_model_data()
        if model_data.empty:
            return pd.DataFrame(), {}
            
        # Calculate metrics for the last 100 episodes
        last_100 = model_data.tail(100)
        
        metrics = {
            'run_id': ['latest'],
            'cd_ratio': [last_100['cd_ratio'].mean() if 'cd_ratio' in last_100.columns 
                        else (last_100['deck_C_ratio'] + last_100['deck_D_ratio']).mean()],
            'learning_rate': [last_100['learning_rate'].mean() if 'learning_rate' in last_100.columns else 0],
            'final_reward': [last_100['total_money'].mean()],
            'overall_alignment': [0.0]  # Will be calculated below
        }
        
        # Calculate alignment score
        cd_alignment = 1 - abs(metrics['cd_ratio'][0] - self.human_baselines['advantageous_ratio'])
        reward_alignment = np.clip(1 - abs(metrics['final_reward'][0] - self.human_baselines['final_money_mean']) 
                                 / (2 * self.human_baselines['final_money_std']), 0, 1)
        metrics['overall_alignment'][0] = (cd_alignment + reward_alignment) / 2
        
        # Statistical tests
        stats_results = {}
        
        # CD ratio t-test
        cd_ratios = last_100['cd_ratio'].values if 'cd_ratio' in last_100.columns \
                   else (last_100['deck_C_ratio'] + last_100['deck_D_ratio']).values
        t_stat, p_val = stats.ttest_1samp(cd_ratios, self.human_baselines['advantageous_ratio'])
        stats_results['cd_ratio_ttest'] = {
            't_statistic': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05
        }
        
        # Reward t-test
        rewards = last_100['total_money'].values
        t_stat, p_val = stats.ttest_1samp(rewards, self.human_baselines['final_money_mean'])
        stats_results['reward_ttest'] = {
            't_statistic': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05
        }
        
        # Effect size (Cohen's d)
        d = (np.mean(rewards) - self.human_baselines['final_money_mean']) / self.human_baselines['final_money_std']
        stats_results['reward_effect_size'] = d
        
        return pd.DataFrame(metrics), stats_results
    
    def plot_alignment_metrics(self, df):
        """Create visualization of alignment metrics"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'C+D Choice Ratio vs Human Baseline',
                'Learning Progress',
                'Final Reward Distribution',
                'Overall Alignment Score'
            )
        )
        
        # 1. CD Ratio Comparison
        if 'cd_ratio' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=['Model', 'Human Baseline'],
                    y=[df['cd_ratio'].iloc[0], self.human_baselines['advantageous_ratio']],
                    name='C+D Ratio'
                ),
                row=1, col=1
            )
        
        # 2. Learning Progress
        model_data = self.load_model_data()
        if not model_data.empty and 'total_money' in model_data.columns:
            smoothed = model_data['total_money'].rolling(window=20, min_periods=1).mean()
            fig.add_trace(
                go.Scatter(
                    x=model_data.index,
                    y=smoothed,
                    name='Model Learning',
                    line=dict(color='blue')
                ),
                row=1, col=2
            )
            fig.add_hline(
                y=self.human_baselines['final_money_mean'],
                line_dash="dash",
                line_color="red",
                annotation_text="Human Baseline",
                row=1, col=2
            )
        
        # 3. Final Reward Distribution
        if not model_data.empty and 'total_money' in model_data.columns:
            fig.add_trace(
                go.Histogram(
                    x=model_data['total_money'].tail(100),
                    name='Model Rewards',
                    nbinsx=20
                ),
                row=2, col=1
            )
            fig.add_vline(
                x=self.human_baselines['final_money_mean'],
                line_dash="dash",
                line_color="red",
                annotation_text="Human Mean",
                row=2, col=1
            )
        
        # 4. Overall Alignment Score
        if 'overall_alignment' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=['Overall Alignment'],
                    y=[df['overall_alignment'].iloc[0]],
                    name='Alignment Score'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Model-Human Alignment Analysis"
        )
        
        return fig