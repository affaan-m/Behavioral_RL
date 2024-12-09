import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

class IGTVisualizer:
    def __init__(self):
        self.colors = {
            'baseline': '#2ecc71',  # Green
            'risk_sensitive': '#e74c3c',  # Red
            'human': '#3498db',  # Blue
            'deck_A': '#c0392b',  # Dark Red
            'deck_B': '#e67e22',  # Orange
            'deck_C': '#27ae60',  # Dark Green
            'deck_D': '#2980b9'   # Dark Blue
        }
        
    def load_data(self):
        """Load all results data"""
        results_dir = Path('results')
        self.baseline_data = pd.read_csv(results_dir / 'baseline_results.csv')
        self.risk_sensitive_data = pd.read_csv(results_dir / 'risk_sensitive_results.csv')
        self.human_data = pd.read_csv(results_dir / 'human_results.csv')
        
    def create_learning_curves_plot(self):
        """Create learning curves comparison plot"""
        fig = go.Figure()
        
        # Add baseline model curve
        fig.add_trace(go.Scatter(
            x=self.baseline_data['episode'],
            y=self.baseline_data['mean_reward'],
            name='Baseline Model',
            line=dict(color=self.colors['baseline']),
            mode='lines'
        ))
        
        # Add risk-sensitive model curve
        fig.add_trace(go.Scatter(
            x=self.risk_sensitive_data['episode'],
            y=self.risk_sensitive_data['mean_reward'],
            name='Risk-Sensitive Model',
            line=dict(color=self.colors['risk_sensitive']),
            mode='lines'
        ))
        
        # Add human data
        fig.add_trace(go.Scatter(
            x=self.human_data['episode'],
            y=self.human_data['mean_reward'],
            name='Human-like Behavior',
            line=dict(color=self.colors['human']),
            mode='lines'
        ))
        
        # Add phase transition line
        fig.add_vline(x=100, line_dash="dash", line_color="gray",
                     annotation_text="Phase Transition")
        
        fig.update_layout(
            title='Learning Curves Comparison',
            xaxis_title='Episode',
            yaxis_title='Mean Reward',
            template='plotly_white',
            xaxis_range=[0, 200],
            annotations=[
                dict(
                    x=50, y=max(self.baseline_data['mean_reward'])*1.1,
                    text="Exploration Phase",
                    showarrow=False
                ),
                dict(
                    x=150, y=max(self.baseline_data['mean_reward'])*1.1,
                    text="Exploitation Phase",
                    showarrow=False
                )
            ]
        )
        
        return fig
        
    def create_deck_preference_plot(self):
        """Create deck preference comparison plot"""
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Baseline Model', 'Risk-Sensitive Model', 'Human-like Behavior'),
            specs=[[{'type': 'pie'}, {'type': 'pie'}, {'type': 'pie'}]]
        )
        
        # Baseline model preferences
        baseline_prefs = [
            self.baseline_data['deck_A_freq'].iloc[-1],
            self.baseline_data['deck_B_freq'].iloc[-1],
            self.baseline_data['deck_C_freq'].iloc[-1],
            self.baseline_data['deck_D_freq'].iloc[-1]
        ]
        
        # Risk-sensitive model preferences
        risk_prefs = [
            self.risk_sensitive_data['deck_A_freq'].iloc[-1],
            self.risk_sensitive_data['deck_B_freq'].iloc[-1],
            self.risk_sensitive_data['deck_C_freq'].iloc[-1],
            self.risk_sensitive_data['deck_D_freq'].iloc[-1]
        ]
        
        # Human preferences
        human_prefs = [
            self.human_data['deck_A_freq'].iloc[-1],
            self.human_data['deck_B_freq'].iloc[-1],
            self.human_data['deck_C_freq'].iloc[-1],
            self.human_data['deck_D_freq'].iloc[-1]
        ]
        
        # Create pie charts
        fig.add_trace(
            go.Pie(labels=['Deck A', 'Deck B', 'Deck C', 'Deck D'],
                   values=baseline_prefs,
                   marker_colors=[self.colors[f'deck_{d}'] for d in 'ABCD']),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Pie(labels=['Deck A', 'Deck B', 'Deck C', 'Deck D'],
                   values=risk_prefs,
                   marker_colors=[self.colors[f'deck_{d}'] for d in 'ABCD']),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Pie(labels=['Deck A', 'Deck B', 'Deck C', 'Deck D'],
                   values=human_prefs,
                   marker_colors=[self.colors[f'deck_{d}'] for d in 'ABCD']),
            row=1, col=3
        )
        
        fig.update_layout(
            title='Deck Selection Preferences',
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
        
    def create_risk_analysis_plot(self):
        """Create risk analysis plot"""
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('Risk Taking Over Time', 'Risk vs Reward Analysis'))
        
        # Calculate risk metrics for full 200 episodes
        baseline_risk = (self.baseline_data['deck_A_freq'] + 
                        self.baseline_data['deck_B_freq'])
        risk_sensitive_risk = (self.risk_sensitive_data['deck_A_freq'] + 
                             self.risk_sensitive_data['deck_B_freq'])
        human_risk = (self.human_data['deck_A_freq'] + 
                     self.human_data['deck_B_freq'])
        
        # Risk taking over time
        fig.add_trace(
            go.Scatter(x=self.baseline_data['episode'],
                      y=baseline_risk,
                      name='Baseline Risk',
                      line=dict(color=self.colors['baseline'])),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=self.risk_sensitive_data['episode'],
                      y=risk_sensitive_risk,
                      name='Risk-Sensitive Risk',
                      line=dict(color=self.colors['risk_sensitive'])),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=self.human_data['episode'],
                      y=human_risk,
                      name='Human-like Risk',
                      line=dict(color=self.colors['human'])),
            row=1, col=1
        )
        
        # Add phase transition line
        fig.add_vline(x=100, line_dash="dash", line_color="gray",
                     row=1, col=1)
        
        # Risk vs Reward scatter plot
        fig.add_trace(
            go.Scatter(x=baseline_risk,
                      y=self.baseline_data['mean_reward'],
                      name='Baseline Model',
                      mode='markers',
                      marker=dict(color=self.colors['baseline'])),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=risk_sensitive_risk,
                      y=self.risk_sensitive_data['mean_reward'],
                      name='Risk-Sensitive Model',
                      mode='markers',
                      marker=dict(color=self.colors['risk_sensitive'])),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=human_risk,
                      y=self.human_data['mean_reward'],
                      name='Human-like Behavior',
                      mode='markers',
                      marker=dict(color=self.colors['human'])),
            row=2, col=1
        )
        
        fig.update_layout(
            height=800,
            title='Risk Analysis',
            template='plotly_white',
            showlegend=True
        )
        
        fig.update_xaxes(title_text='Episode', range=[0, 200], row=1, col=1)
        fig.update_xaxes(title_text='Risk Level (% High-Risk Decks)', row=2, col=1)
        fig.update_yaxes(title_text='% High-Risk Deck Selection', row=1, col=1)
        fig.update_yaxes(title_text='Mean Reward', row=2, col=1)
        
        return fig
        
    def create_statistical_comparison_plot(self):
        """Create statistical comparison plot"""
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Final Mean Rewards',
                                         'Learning Speed',
                                         'Risk Aversion',
                                         'Advantageous Deck Ratio'))
        
        # Final mean rewards comparison
        final_rewards = {
            'Baseline': self.baseline_data['mean_reward'].iloc[-1],
            'Risk-Sensitive': self.risk_sensitive_data['mean_reward'].iloc[-1],
            'Human-like': self.human_data['mean_reward'].iloc[-1]
        }
        
        # Learning speed (episodes to reach 75% of final performance)
        baseline_learning = len(self.baseline_data[
            self.baseline_data['mean_reward'] < 0.75 * final_rewards['Baseline']
        ])
        risk_learning = len(self.risk_sensitive_data[
            self.risk_sensitive_data['mean_reward'] < 0.75 * final_rewards['Risk-Sensitive']
        ])
        human_learning = len(self.human_data[
            self.human_data['mean_reward'] < 0.75 * final_rewards['Human-like']
        ])
        
        # Risk aversion (final % of safe deck selection)
        baseline_risk = (self.baseline_data['deck_A_freq'].iloc[-1] + 
                        self.baseline_data['deck_B_freq'].iloc[-1])
        risk_sensitive_risk = (self.risk_sensitive_data['deck_A_freq'].iloc[-1] + 
                             self.risk_sensitive_data['deck_B_freq'].iloc[-1])
        human_risk = (self.human_data['deck_A_freq'].iloc[-1] + 
                     self.human_data['deck_B_freq'].iloc[-1])
        
        # Advantageous deck ratio
        baseline_adv = (self.baseline_data['deck_C_freq'].iloc[-1] + 
                       self.baseline_data['deck_D_freq'].iloc[-1])
        risk_sensitive_adv = (self.risk_sensitive_data['deck_C_freq'].iloc[-1] + 
                            self.risk_sensitive_data['deck_D_freq'].iloc[-1])
        human_adv = (self.human_data['deck_C_freq'].iloc[-1] + 
                    self.human_data['deck_D_freq'].iloc[-1])
        
        # Create plots
        fig.add_trace(
            go.Bar(x=list(final_rewards.keys()),
                  y=list(final_rewards.values()),
                  marker_color=[self.colors['baseline'], 
                              self.colors['risk_sensitive'],
                              self.colors['human']]),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=['Baseline', 'Risk-Sensitive', 'Human-like'],
                  y=[baseline_learning, risk_learning, human_learning],
                  marker_color=[self.colors['baseline'], 
                              self.colors['risk_sensitive'],
                              self.colors['human']]),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=['Baseline', 'Risk-Sensitive', 'Human-like'],
                  y=[100 - baseline_risk*100, 100 - risk_sensitive_risk*100, 100 - human_risk*100],
                  marker_color=[self.colors['baseline'], 
                              self.colors['risk_sensitive'],
                              self.colors['human']]),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=['Baseline', 'Risk-Sensitive', 'Human-like'],
                  y=[baseline_adv*100, risk_sensitive_adv*100, human_adv*100],
                  marker_color=[self.colors['baseline'], 
                              self.colors['risk_sensitive'],
                              self.colors['human']]),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title='Statistical Comparisons',
            template='plotly_white',
            showlegend=False
        )
        
        # Update axes labels
        fig.update_yaxes(title_text='Mean Reward', row=1, col=1)
        fig.update_yaxes(title_text='Episodes', row=1, col=2)
        fig.update_yaxes(title_text='Risk Aversion %', row=2, col=1)
        fig.update_yaxes(title_text='Advantageous Deck %', row=2, col=2)
        
        return fig
    
    def save_dashboard(self, output_dir='visualizations'):
        """Generate and save all visualizations"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load data first
        self.load_data()
        
        # Create all plots
        learning_curves = self.create_learning_curves_plot()
        deck_preferences = self.create_deck_preference_plot()
        risk_analysis = self.create_risk_analysis_plot()
        statistical_comparison = self.create_statistical_comparison_plot()
        
        # Save individual plots
        learning_curves.write_html(f'{output_dir}/learning_curves.html')
        deck_preferences.write_html(f'{output_dir}/deck_preferences.html')
        risk_analysis.write_html(f'{output_dir}/risk_analysis.html')
        statistical_comparison.write_html(f'{output_dir}/statistical_comparison.html')
        
        # Create and save combined dashboard
        dashboard = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Learning Curves',
                          'Deck Preferences',
                          'Risk Analysis',
                          'Statistical Comparisons'),
            specs=[[{'type': 'scatter'}, {'type': 'pie'}],
                  [{'type': 'scatter'}, {'type': 'bar'}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Add learning curves
        if len(learning_curves.data) > 0:
            for trace in learning_curves.data:
                dashboard.add_trace(trace, row=1, col=1)
        
        # Add deck preferences
        if len(deck_preferences.data) > 0:
            dashboard.add_trace(deck_preferences.data[0], row=1, col=2)
        
        # Add risk analysis
        if len(risk_analysis.data) > 0:
            for trace in risk_analysis.data[:2]:  # Only add the first subplot
                dashboard.add_trace(trace, row=2, col=1)
        
        # Add statistical comparison
        if len(statistical_comparison.data) > 0:
            dashboard.add_trace(statistical_comparison.data[0], row=2, col=2)
        
        dashboard.update_layout(
            height=1200,
            width=1600,
            title='IGT Model Comparison Dashboard',
            template='plotly_white',
            showlegend=True
        )
        
        # Update axes labels
        dashboard.update_xaxes(title_text='Episode', row=1, col=1)
        dashboard.update_xaxes(title_text='Risk Level', row=2, col=1)
        dashboard.update_yaxes(title_text='Mean Reward', row=1, col=1)
        dashboard.update_yaxes(title_text='Risk Taking', row=2, col=1)
        
        dashboard.write_html(f'{output_dir}/dashboard.html')

if __name__ == "__main__":
    visualizer = IGTVisualizer()
    visualizer.load_data()
    visualizer.save_dashboard() 