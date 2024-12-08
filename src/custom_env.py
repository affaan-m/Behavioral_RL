import gymnasium as gym
from gymnasium import spaces
import numpy as np

class InvestmentGameEnv(gym.Env):
    """
    An investment game environment designed to test human-like risk preferences.
    
    The agent starts with initial capital and must make sequential investment decisions.
    Each action represents the proportion of current wealth to invest.
    
    Key features that test human biases:
    1. Asymmetric returns (losses hurt more than equivalent gains help)
    2. Sequential decision-making with clear reference points
    3. Different risk levels in gain/loss domains
    """
    
    def __init__(self, initial_capital=1000.0, max_steps=50):
        super(InvestmentGameEnv, self).__init__()
        
        self.initial_capital = initial_capital
        self.max_steps = max_steps
        
        # Action space: Investment proportion [0,1] in 0.1 increments
        self.action_space = spaces.Discrete(11)  # 0, 0.1, 0.2, ..., 1.0
        
        # State space: [current_capital, steps_remaining, previous_return]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -1]), 
            high=np.array([np.inf, self.max_steps, 1]),
            dtype=np.float32
        )
        
        # Market parameters
        self.up_prob = 0.5  # Probability of positive return
        self.up_return = 0.2  # 20% gain
        self.down_return = -0.15  # 15% loss
        
        # Prospect theory parameters
        self.lambda_loss = 2.25  # Loss aversion coefficient
        self.alpha = 0.88  # Risk aversion for gains
        self.beta = 0.88  # Risk aversion for losses
        self.reference_point = initial_capital  # Reference point updates with initial capital
        
        self.reset()
        
    def prospect_theory_value(self, reward):
        # Convert reward to gain/loss relative to reference point
        relative_outcome = reward - self.reference_point
        
        # Apply prospect theory value function
        if relative_outcome >= 0:
            value = relative_outcome ** self.alpha
        else:
            value = -self.lambda_loss * (-relative_outcome) ** self.beta
            
        return value
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        self.capital = self.initial_capital
        self.steps_remaining = self.max_steps
        self.previous_return = 0.0
        
        self.state = np.array([
            self.capital,
            self.steps_remaining,
            self.previous_return
        ])
        
        return self.state, {}
        
    def step(self, action):
        # Convert discrete action to investment proportion
        investment_prop = action * 0.1  # Convert 0-10 to 0.0-1.0
        
        # Calculate investment amount
        investment = self.capital * investment_prop
        
        # Generate market return
        if self.np_random.random() < self.up_prob:
            market_return = self.up_return
        else:
            market_return = self.down_return
            
        # Calculate returns
        investment_return = investment * market_return
        self.previous_return = market_return
        
        # Update capital
        old_capital = self.capital
        self.capital += investment_return
        
        # Calculate raw reward (change in capital)
        raw_reward = self.capital - old_capital
        
        # Apply prospect theory value function
        prospect_reward = self.prospect_theory_value(raw_reward)
        
        # Update remaining steps
        self.steps_remaining -= 1
        
        # Update state
        self.state = np.array([
            self.capital,
            self.steps_remaining,
            self.previous_return
        ])
        
        # Check if episode is done
        done = bool(self.steps_remaining <= 0 or self.capital <= 0)
        
        info = {
            'raw_reward': raw_reward,
            'prospect_reward': prospect_reward,
            'capital': self.capital,
            'investment_proportion': investment_prop,
            'market_return': market_return
        }
        
        return self.state, prospect_reward, done, False, info
        
    def render(self):
        print(f"Capital: {self.capital:.2f}, Steps Left: {self.steps_remaining}, Last Return: {self.previous_return:.2%}")