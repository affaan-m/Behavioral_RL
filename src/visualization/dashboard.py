import os
import sys
from pathlib import Path

# Add the src directory to Python path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import glob
from analysis.model_human_comparison import ModelHumanComparison

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("IGT Learning Dashboard", style={'textAlign': 'center', 'marginBottom': '30px'}),
    
    dcc.Tabs([
        dcc.Tab(label='Training Progress', children=[
            html.Div([
                html.Div([
                    html.H3("Learning Progress", style={'textAlign': 'center'}),
                    dcc.Graph(id='training-progress'),
                ], className='six columns'),
                
                html.Div([
                    html.H3("Deck Selection Evolution", style={'textAlign': 'center'}),
                    dcc.Graph(id='deck-evolution'),
                ], className='six columns'),
                
                html.Div([
                    html.H3("Training Statistics", style={'textAlign': 'center'}),
                    html.Div(id='training-stats', style={
                        'padding': '20px',
                        'backgroundColor': '#f8f9fa',
                        'borderRadius': '5px',
                        'marginTop': '20px'
                    })
                ], className='twelve columns'),
                
                dcc.Interval(
                    id='training-interval',
                    interval=5*1000,  # 5 seconds
                    n_intervals=0
                )
            ], className='row')
        ]),
        
        dcc.Tab(label='Human Alignment', children=[
            html.Div([
                html.H3("Model-Human Comparison", style={'textAlign': 'center'}),
                dcc.Graph(id='alignment-metrics'),
                html.Div(id='statistical-summary', style={
                    'padding': '20px',
                    'backgroundColor': '#f8f9fa',
                    'borderRadius': '5px',
                    'marginTop': '20px'
                }),
                dcc.Interval(
                    id='alignment-interval',
                    interval=10*1000,
                    n_intervals=0
                )
            ])
        ]),
        
        dcc.Tab(label='Detailed Statistics', children=[
            html.Div([
                html.H3("Performance Metrics", style={'textAlign': 'center'}),
                dcc.Graph(id='performance-metrics'),
                html.Div([
                    html.H4("Statistical Analysis", style={'textAlign': 'center'}),
                    html.Div(id='detailed-stats', style={
                        'padding': '20px',
                        'backgroundColor': '#f8f9fa',
                        'borderRadius': '5px'
                    })
                ]),
                dcc.Interval(
                    id='stats-interval',
                    interval=10*1000,
                    n_intervals=0
                )
            ])
        ])
    ])
], style={'padding': '20px'})

def load_training_data():
    try:
        # First try loading from training.log
        if os.path.exists('training.log'):
            df = pd.read_csv('training.log')
            if not df.empty:
                if 'episode' not in df.columns:
                    df['episode'] = range(len(df))
                return df
        
        # Then try loading from metrics directory
        metrics_dir = Path('metrics')
        if not metrics_dir.exists():
            print("No metrics directory found")
            return pd.DataFrame({'episode': [], 'total_money': [], 'deck_picks': []})
            
        # Find the latest run directory
        run_dirs = [d for d in metrics_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
        if not run_dirs:
            print("No run directories found in metrics/")
            return pd.DataFrame({'episode': [], 'total_money': [], 'deck_picks': []})
            
        latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
        
        # Look for CSV files in the latest run directory
        csv_files = list(latest_run.glob('*.csv'))
        if not csv_files:
            print(f"No CSV files found in {latest_run}")
            return pd.DataFrame({'episode': [], 'total_money': [], 'deck_picks': []})
            
        latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
        df = pd.read_csv(latest_csv)
        
        if 'episode' not in df.columns:
            df['episode'] = range(len(df))
            
        return df
    except Exception as e:
        print(f"Error loading training data: {str(e)}")
        return pd.DataFrame({'episode': [], 'total_money': [], 'deck_picks': []})

@app.callback(
    [Output('training-progress', 'figure'),
     Output('deck-evolution', 'figure'),
     Output('training-stats', 'children')],
    Input('training-interval', 'n_intervals')
)
def update_training_progress(n):
    results = load_training_data()
    
    if results.empty:
        empty_fig = go.Figure(layout=go.Layout(
            title="No training data available yet. Please check metrics/ directory or training.log",
            xaxis_title="Episode",
            yaxis_title="Reward"
        ))
        return empty_fig, empty_fig, "No training data available. Please check metrics/ directory or training.log"
    
    # Training progress figure with multiple metrics
    progress_fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Raw rewards
    progress_fig.add_trace(
        go.Scatter(
            x=results['episode'].values,  # Ensure we're using values
            y=results['total_money'].values,
            name='Episode Reward',
            mode='lines',
            line=dict(width=1, color='gray'),
            opacity=0.3
        ),
        secondary_y=False
    )
    
    # Smoothed rewards
    window = 20
    smoothed = results['total_money'].rolling(window=window, min_periods=1).mean()
    progress_fig.add_trace(
        go.Scatter(
            x=results['episode'].values,
            y=smoothed.values,
            name=f'Smoothed Reward ({window}-ep)',
            mode='lines',
            line=dict(width=2, color='blue')
        ),
        secondary_y=False
    )
    
    # Add learning rate if available
    if 'learning_rate' in results.columns:
        progress_fig.add_trace(
            go.Scatter(
                x=results['episode'].values,
                y=results['learning_rate'].values,
                name='Learning Rate',
                mode='lines',
                line=dict(width=1, color='red')
            ),
            secondary_y=True
        )
    
    progress_fig.update_layout(
        title='Training Progress',
        xaxis_title='Episode',
        yaxis_title='Reward',
        yaxis2_title='Learning Rate',
        hovermode='x unified',
        showlegend=True
    )
    
    # Deck evolution figure
    deck_fig = go.Figure()
    
    if 'deck_picks' in results.columns:
        deck_names = ['A', 'B', 'C', 'D']
        colors = ['red', 'blue', 'green', 'purple']
        
        for i, (deck, color) in enumerate(zip(deck_names, colors)):
            deck_picks = [eval(picks)[i] if isinstance(picks, str) else picks[i] 
                         for picks in results['deck_picks']]
            total_picks = [sum(eval(picks)) if isinstance(picks, str) else sum(picks) 
                          for picks in results['deck_picks']]
            proportions = np.array(deck_picks) / np.array(total_picks)
            
            # Smooth the proportions
            smoothed_props = pd.Series(proportions).rolling(window=20, min_periods=1).mean()
            
            deck_fig.add_trace(go.Scatter(
                x=results['episode'].values,
                y=smoothed_props.values,
                name=f'Deck {deck}',
                mode='lines',
                line=dict(color=color, width=2)
            ))
    
    deck_fig.update_layout(
        title='Deck Selection Evolution',
        xaxis_title='Episode',
        yaxis_title='Proportion of Selections',
        hovermode='x unified',
        yaxis_range=[0, 1]
    )
    
    # Calculate training statistics
    last_100 = results.tail(100)
    last_20 = results.tail(20)
    
    stats_children = [
        html.Div([
            html.H5("Recent Performance Metrics"),
            html.P([
                html.Strong("Last 20 Episodes: "),
                f"Avg Reward: {last_20['total_money'].mean():.1f}, ",
                f"Std: {last_20['total_money'].std():.1f}"
            ]),
            html.P([
                html.Strong("Last 100 Episodes: "),
                f"Avg Reward: {last_100['total_money'].mean():.1f}, ",
                f"Std: {last_100['total_money'].std():.1f}"
            ])
        ])
    ]
    
    return progress_fig, deck_fig, stats_children

@app.callback(
    [Output('performance-metrics', 'figure'),
     Output('detailed-stats', 'children')],
    Input('stats-interval', 'n_intervals')
)
def update_detailed_stats(n):
    results = load_training_data()
    
    if results.empty:
        return go.Figure(), "No training data available"
    
    # Create subplots for detailed metrics
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=('Reward Distribution', 'Learning Curve',
                                     'Deck Preference Evolution', 'Risk Sensitivity'))
    
    # 1. Reward Distribution
    fig.add_trace(
        go.Histogram(x=results['total_money'], nbinsx=30, name='Reward Dist'),
        row=1, col=1
    )
    
    # 2. Learning Curve with Confidence Interval
    window = 50
    smoothed = results['total_money'].rolling(window=window, min_periods=1).mean()
    std = results['total_money'].rolling(window=window, min_periods=1).std()
    
    fig.add_trace(
        go.Scatter(
            x=results['episode'],
            y=smoothed,
            name='Mean Reward',
            line=dict(color='blue', width=2)
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=results['episode'],
            y=smoothed + std,
            fill=None,
            mode='lines',
            line_color='gray',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=results['episode'],
            y=smoothed - std,
            fill='tonexty',
            mode='lines',
            line_color='gray',
            name='±1 std'
        ),
        row=1, col=2
    )
    
    # 3. Deck Preference Evolution (if available)
    if 'deck_picks' in results.columns:
        deck_names = ['A', 'B', 'C', 'D']
        colors = ['red', 'blue', 'green', 'purple']
        
        for i, (deck, color) in enumerate(zip(deck_names, colors)):
            deck_picks = [eval(picks)[i] if isinstance(picks, str) else picks[i] 
                         for picks in results['deck_picks']]
            total_picks = [sum(eval(picks)) if isinstance(picks, str) else sum(picks) 
                          for picks in results['deck_picks']]
            proportions = np.array(deck_picks) / np.array(total_picks)
            
            fig.add_trace(
                go.Scatter(
                    x=results['episode'],
                    y=proportions,
                    name=f'Deck {deck}',
                    line=dict(color=color)
                ),
                row=2, col=1
            )
    
    # 4. Risk Sensitivity (C+D vs A+B ratio)
    if 'deck_picks' in results.columns:
        cd_ratio = []
        for picks in results['deck_picks']:
            picks = eval(picks) if isinstance(picks, str) else picks
            cd = picks[2] + picks[3]  # C + D
            ab = picks[0] + picks[1]  # A + B
            total = cd + ab
            cd_ratio.append(cd / total if total > 0 else 0)
        
        fig.add_trace(
            go.Scatter(
                x=results['episode'],
                y=cd_ratio,
                name='C+D Ratio',
                line=dict(color='green')
            ),
            row=2, col=2
        )
        
        # Add 0.5 reference line
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=2, col=2)
    
    fig.update_layout(height=800, showlegend=True)
    
    # Calculate detailed statistics
    recent = results.tail(100)
    stats_children = [
        html.Div([
            html.H5("Detailed Statistics (Last 100 Episodes)"),
            html.P([
                html.Strong("Reward Statistics: "),
                f"Mean: {recent['total_money'].mean():.1f}, ",
                f"Std: {recent['total_money'].std():.1f}, ",
                f"Max: {recent['total_money'].max():.1f}, ",
                f"Min: {recent['total_money'].min():.1f}"
            ]),
            html.P([
                html.Strong("Performance Trend: "),
                f"{'Improving' if recent['total_money'].corr(pd.Series(range(len(recent)))) > 0 else 'Declining'}"
            ])
        ])
    ]
    
    if 'deck_picks' in results.columns:
        last_picks = results['deck_picks'].iloc[-100:]
        deck_stats = {deck: 0 for deck in 'ABCD'}
        total_picks = 0
        
        for picks in last_picks:
            picks = eval(picks) if isinstance(picks, str) else picks
            for i, count in enumerate(picks):
                deck_stats['ABCD'[i]] += count
                total_picks += count
        
        deck_prefs = {deck: count/total_picks for deck, count in deck_stats.items()}
        
        stats_children.append(html.Div([
            html.H5("Deck Preferences (Last 100 Episodes)"),
            html.P([
                html.Strong(f"Deck {deck}: "),
                f"{pref*100:.1f}%, "
            ] for deck, pref in deck_prefs.items())
        ]))
    
    return fig, stats_children

if __name__ == '__main__':
    app.run_server(debug=True, port=8050) 