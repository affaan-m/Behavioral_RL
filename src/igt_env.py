import gymnasium as gym
from gymnasium import spaces
import numpy as np
from config import Config
from collections import deque

class IGTEnvironment(gym.Env):
    """
    Iowa Gambling Task (IGT) Environment
    
    This implements the classic IGT paradigm:
    - 4 decks (A, B, C, D)
    - Each deck has different reward/punishment schedules
    - Players make sequential choices between decks
    - Goal is to maximize long-term gains
    
    Deck characteristics:
    A & B (disadvantageous decks):
    - High immediate reward ($100)
    - Higher future punishments
    - Negative expected value (-$25 per trial)
    
    C & D (advantageous decks):
    - Lower immediate reward ($50)
    - Lower future punishments
    - Positive expected value (+$25 per trial)
    """
    
    def __init__(self, max_steps=100):
        super(IGTEnvironment, self).__init__()
        
        # Action space: 4 decks (0=A, 1=B, 2=C, 3=D)
        self.action_space = spaces.Discrete(4)
        
        # Enhanced state space with more features
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -1, 0, 0, 0, 0, -np.inf, -np.inf, -np.inf, -np.inf]),
            high=np.array([np.inf, np.inf, 4, max_steps, max_steps, max_steps, max_steps, np.inf, np.inf, np.inf, np.inf]),
            dtype=np.float32
        )
        
        self.max_steps = max_steps
        
        # Refined deck configurations for better learning dynamics
        self.deck_configs = {
            'A': {
                'reward': 100,
                'punishment_probs': [0.4] * 10,  # Reduced punishment frequency
                'punishments': [-250] * 10,
                'expected_value': -15
            },
            'B': {
                'reward': 100,
                'punishment_probs': [0.25] * 10,  # Adjusted for more balanced risk
                'punishments': [-650] * 10,
                'expected_value': -15
            },
            'C': {
                'reward': 50,
                'punishment_probs': [0.4] * 10,  # Increased punishment frequency
                'punishments': [-50] * 10,
                'expected_value': 25
            },
            'D': {
                'reward': 50,
                'punishment_probs': [0.15] * 10,  # Slightly increased risk
                'punishments': [-200] * 10,  # Reduced punishment magnitude
                'expected_value': 25
            }
        }
        
        # Initialize deck positions and visit counts
        self.deck_positions = {deck: 0 for deck in 'ABCD'}
        self.deck_visits = np.zeros(4)
        
        # Load prospect theory parameters from config
        self.lambda_loss = Config.LAMBDA_LOSS
        self.alpha = Config.ALPHA_GAIN
        self.beta = Config.BETA_LOSS
        self.reference_point = Config.REFERENCE_POINT
        
        # Enhanced exploration parameters
        self.exploration_bonus = Config.EXPLORATION_BONUS
        self.novelty_weight = Config.NOVELTY_WEIGHT
        self.visit_decay = 0.995  # Decay factor for visit counts
        
        # Track additional metrics
        self.cumulative_rewards = []
        self.last_n_rewards = deque(maxlen=10)
        self.deck_performance = {deck: [] for deck in 'ABCD'}
        
        self.reset()
        
    def get_state(self):
        """Enhanced state representation with fixed dimensionality"""
        # Calculate normalized deck visits
        if np.sum(self.deck_visits) > 0:
            normalized_visits = self.deck_visits / np.sum(self.deck_visits)
        else:
            normalized_visits = np.array([0.25, 0.25, 0.25, 0.25])
        
        # Basic state components
        basic_state = np.array([
            self.total_money / 1000.0,  # Normalized total money
            self.last_reward / 100.0,   # Normalized last reward
            self.last_deck,
            *normalized_visits          # Unpack normalized deck visits
        ])
        
        # Additional state components
        avg_reward = np.mean(self.last_n_rewards) if self.last_n_rewards else 0
        reward_std = np.std(self.last_n_rewards) if len(self.last_n_rewards) > 1 else 0
        
        # Recent deck performance (normalized)
        deck_metrics = np.array([
            np.mean(self.deck_performance[deck][-5:]) if self.deck_performance[deck] else 0
            for deck in 'ABCD'
        ]) / 100.0
        
        # Combine all state components (11 dimensions total)
        state = np.concatenate([
            basic_state,                     # 7 dimensions
            [avg_reward / 100.0],           # 1 dimension
            [reward_std / 100.0],           # 1 dimension
            [np.mean(deck_metrics)],        # 1 dimension
            [np.std(deck_metrics)]          # 1 dimension
        ])
        
        return state.astype(np.float32)  # Ensure float32 type
        
    def reset(self, seed=None):
        """Reset environment state"""
        super().reset(seed=seed)
        
        self.total_money = 2000
        self.last_reward = 0
        self.last_deck = -1
        self.steps_taken = 0
        self.deck_visits = np.zeros(4)
        self.deck_positions = {deck: 0 for deck in 'ABCD'}
        self.cumulative_rewards = []
        self.last_n_rewards = deque(maxlen=10)
        self.deck_performance = {deck: [] for deck in 'ABCD'}
        
        # Get initial state
        state = self.get_state()
        return state, {}
        
    def calculate_exploration_bonus(self, action):
        """Calculate exploration bonus based on visit counts"""
        total_visits = np.sum(self.deck_visits) + 1e-6
        visit_ratio = self.deck_visits[action] / total_visits
        novelty = 1.0 - visit_ratio
        return self.exploration_bonus * novelty
        
    def step(self, action):
        """Enhanced step function with improved reward shaping"""
        assert self.action_space.contains(action)
        
        # Get deck configuration
        deck = list('ABCD')[action]
        deck_config = self.deck_configs[deck]
        
        # Basic reward calculation
        reward = deck_config['reward']
        if np.random.random() < deck_config['punishment_probs'][self.deck_positions[deck]]:
            punishment = deck_config['punishments'][self.deck_positions[deck]]
            reward += punishment
        
        # Update metrics
        self.total_money += reward
        self.last_reward = reward
        self.last_deck = action
        self.deck_visits[action] += 1
        self.steps_taken += 1
        self.cumulative_rewards.append(reward)
        self.last_n_rewards.append(reward)
        self.deck_performance[deck].append(reward)
        
        # Calculate exploration bonus
        visit_counts = self.deck_visits + 1  # Add 1 to avoid division by zero
        exploration_bonus = self.exploration_bonus * (1 / np.sqrt(visit_counts[action]))
        
        # Apply prospect theory transformation
        transformed_reward = self._prospect_theory_transform(reward)
        
        # Combine immediate reward with exploration bonus
        shaped_reward = transformed_reward + exploration_bonus
        
        # Decay visit counts
        self.deck_visits *= self.visit_decay
        
        # Update deck position
        self.deck_positions[deck] = (self.deck_positions[deck] + 1) % 10
        
        # Check if episode is done
        done = self.steps_taken >= self.max_steps
        
        # Get enhanced state
        state = self.get_state()
        
        # Additional info
        info = {
            'total_money': self.total_money,
            'deck_picks': self.deck_visits.tolist(),
            'raw_reward': reward,
            'exploration_bonus': exploration_bonus,
            'shaped_reward': shaped_reward,
            'transformed_reward': transformed_reward
        }
        
        return state, shaped_reward, done, False, info
        
    def _prospect_theory_transform(self, reward):
        """Enhanced prospect theory transformation"""
        # Calculate gain/loss relative to reference point
        relative_outcome = reward - self.reference_point
        
        if relative_outcome >= 0:
            # Gain domain
            value = (relative_outcome ** self.alpha)
        else:
            # Loss domain
            value = -self.lambda_loss * ((-relative_outcome) ** self.beta)
        
        # Scale the transformed value
        return value / 100.0  # Normalize to reasonable range
        
    def render(self):
        """Print current state information"""
        print(f"Total Money: ${self.total_money}")
        print(f"Last Reward: ${self.last_reward}")
        print(f"Last Deck: {['A', 'B', 'C', 'D'][self.last_deck] if self.last_deck >= 0 else 'None'}")
        print("Deck Picks:", {deck: picks for deck, picks in zip('ABCD', self.deck_picks)})
        
    @staticmethod
    def get_human_baseline_metrics():
        """
        Return baseline metrics from human IGT studies
        Based on published data from Bechara et al. and other IGT papers
        """
        return {
            'advantageous_ratio': 0.6,  # Ratio of C+D choices in last 20 trials
            'learning_rate': 0.15,  # Average increase in advantageous choices per block
            'loss_aversion': 2.25,  # Lambda from prospect theory
            'risk_seeking_after_loss': 1.2,  # Ratio of risky choices after losses
            'final_money_mean': 2250,  # Mean final money for healthy controls
            'final_money_std': 750,  # Std of final money for healthy controls
            'deck_preferences': {
                'A': 0.2,
                'B': 0.2,
                'C': 0.3,
                'D': 0.3
            }
        }