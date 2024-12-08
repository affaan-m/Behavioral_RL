from flask import Flask, render_template, jsonify, request
from igt_env import IGTEnvironment
from config.database import save_participant_data
import json
from datetime import datetime

app = Flask(__name__)
env = IGTEnvironment()

@app.route('/')
def index():
    return render_template('igt.html')

@app.route('/api/start', methods=['POST'])
def start_experiment():
    participant_info = request.json
    participant_info['timestamp'] = datetime.now().isoformat()
    participant_info['session_id'] = f"{participant_info['timestamp']}_{participant_info.get('age')}_{participant_info.get('gender')}"
    
    # Initialize new environment
    state = env.reset()[0]
    
    return jsonify({
        'session_id': participant_info['session_id'],
        'initial_state': state.tolist() if hasattr(state, 'tolist') else state,
        'total_money': env.total_money
    })

@app.route('/api/step', methods=['POST'])
def step():
    data = request.json
    action = data['action']
    
    state, reward, done, _, info = env.step(action)
    
    return jsonify({
        'state': state.tolist() if hasattr(state, 'tolist') else state,
        'reward': reward,
        'done': done,
        'total_money': info['total_money']
    })

@app.route('/api/save', methods=['POST'])
def save_results():
    data = request.json
    
    # Calculate metrics
    choices = data['history']['deck_choices']
    rewards = data['history']['rewards']
    
    # Calculate advantageous choices (C+D) in last 20 trials
    last_20_choices = choices[-20:]
    advantageous = sum(1 for c in last_20_choices if c in [2, 3]) / len(last_20_choices)
    
    # Calculate risk-seeking after losses
    risk_seeking = []
    for i in range(1, len(choices)):
        if rewards[i-1] < 0:  # After a loss
            risk_seeking.append(1 if choices[i] in [0, 1] else 0)
    risk_seeking_ratio = sum(risk_seeking) / len(risk_seeking) if risk_seeking else 0
    
    # Calculate deck preferences
    total_choices = len(choices)
    deck_preferences = {
        deck: choices.count(i) / total_choices
        for i, deck in enumerate(['A', 'B', 'C', 'D'])
    }
    
    # Prepare data for storage
    experiment_data = {
        'participant_info': data['participant_info'],
        'history': data['history'],
        'metrics': {
            'total_money': data['history']['total_money'][-1],
            'advantageous_ratio': advantageous,
            'risk_seeking_after_loss': risk_seeking_ratio,
            'deck_preferences': deck_preferences,
            'mean_reaction_time': sum(data['history']['reaction_times']) / len(data['history']['reaction_times'])
        }
    }
    
    # Save to Supabase
    result = save_participant_data(experiment_data)
    
    return jsonify({
        'success': result is not None,
        'message': 'Data saved successfully' if result is not None else 'Error saving data'
    })

if __name__ == '__main__':
    app.run(debug=True) 