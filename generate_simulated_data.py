import json
import uuid
import numpy as np
from datetime import datetime, timedelta

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def normalize_probs(probs):
    """Normalize probabilities to sum to 1"""
    probs = np.array(probs, dtype=float)
    return probs / np.sum(probs)

def generate_participant_data(is_control=True):
    """Generate data for one participant
    Args:
        is_control: If True, generates data for normal controls who learn to avoid bad decks
                   If False, generates data for prefrontal patients who persist with bad decks
    """
    history = []
    total_money = 2000  # Initial loan as per original paper
    deck_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    
    # Deck configurations exactly matching Bechara et al. 1994
    deck_configs = {
        'A': {'reward': 100, 'penalty': -250, 'penalty_prob': 0.5},  # EV = -25
        'B': {'reward': 100, 'penalty': -250, 'penalty_prob': 0.5},  # EV = -25
        'C': {'reward': 50, 'penalty': -50, 'penalty_prob': 0.5},   # EV = +25
        'D': {'reward': 50, 'penalty': -50, 'penalty_prob': 0.5}    # EV = +25
    }
    
    # Learning parameters based on paper
    exploration_phase = np.random.randint(10, 20)  # Initial exploration
    learning_phase = np.random.randint(30, 50) if is_control else 999  # Controls learn, patients don't
    
    for trial in range(100):  # 100 trials as per original paper
        # Probability distribution for deck selection
        if trial < exploration_phase:
            # Initial exploration phase - roughly equal probabilities
            probs = normalize_probs([0.25, 0.25, 0.25, 0.25])
        elif is_control and trial >= learning_phase:
            # Control subjects learn to prefer advantageous decks (C & D)
            base_probs = [0.1, 0.1, 0.4, 0.4]
            noise = 0.1 * np.random.random(4)
            probs = normalize_probs([p + n for p, n in zip(base_probs, noise)])
        else:
            # Pre-learning or patient behavior - preference for high immediate reward (A & B)
            base_probs = [0.4, 0.4, 0.1, 0.1]
            noise = 0.2 * np.random.random(4)
            probs = normalize_probs([p + n for p, n in zip(base_probs, noise)])
        
        # Select deck
        deck = np.random.choice(['A', 'B', 'C', 'D'], p=probs)
        deck_counts[deck] += 1
        
        # Calculate reward and penalty
        config = deck_configs[deck]
        reward = config['reward']
        penalty = config['penalty'] if np.random.random() < config['penalty_prob'] else 0
        
        net_reward = reward + penalty
        total_money += net_reward
        
        # Record trial data
        history.append({
            'trial': trial + 1,
            'deck': deck,
            'reward': float(reward),
            'penalty': float(penalty),
            'net_reward': float(net_reward),
            'total_money': float(total_money),
            'reaction_time': float(np.random.uniform(0.8, 2.5))  # Slightly longer RTs based on paper
        })
    
    # Calculate metrics
    total_trials = float(len(history))
    deck_preferences = {
        f'deck_{k}': float(v / total_trials * 100)
        for k, v in deck_counts.items()
    }
    
    # Calculate metrics mentioned in the paper
    blocks = [history[i:i+20] for i in range(0, len(history), 20)]  # Split into 5 blocks of 20 trials
    block_preferences = []
    for block in blocks:
        block_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        for trial in block:
            block_counts[trial['deck']] += 1
        good_choices = block_counts['C'] + block_counts['D']
        bad_choices = block_counts['A'] + block_counts['B']
        block_preferences.append((good_choices - bad_choices) / len(block))
    
    metrics = {
        'total_money': float(total_money),
        'completed_trials': len(history),
        'deck_preferences': deck_preferences,
        'advantageous_choices': float((deck_counts['C'] + deck_counts['D']) / total_trials * 100),
        'disadvantageous_choices': float((deck_counts['A'] + deck_counts['B']) / total_trials * 100),
        'block_preferences': block_preferences,
        'average_reaction_time': float(np.mean([t['reaction_time'] for t in history]))
    }
    
    return {
        'id': str(uuid.uuid4()),
        'type': 'control' if is_control else 'patient',
        'metrics': metrics,
        'history': history
    }

# Generate data
simulated_data = []
base_time = datetime.now() - timedelta(days=7)

# Generate control data (80% of participants)
for _ in range(80):
    data = generate_participant_data(is_control=True)
    data['timestamp'] = (base_time + timedelta(hours=np.random.randint(0, 24*7))).isoformat()
    simulated_data.append(data)

# Generate patient-like data (20% of participants)
for _ in range(20):
    data = generate_participant_data(is_control=False)
    data['timestamp'] = (base_time + timedelta(hours=np.random.randint(0, 24*7))).isoformat()
    simulated_data.append(data)

# Save to file
with open('simulated_igt_data.json', 'w') as f:
    json.dump(simulated_data, f, indent=2, cls=NumpyEncoder)

print(f"Generated {len(simulated_data)} simulated participants data")
print("Sample metrics from first participant:")
print(json.dumps(simulated_data[0]['metrics'], indent=2, cls=NumpyEncoder))
print("\nData saved to simulated_igt_data.json")