import numpy as np
from scipy import stats
from scipy.optimize import minimize
import pandas as pd

class ProspectTheoryModel:
    def __init__(self, alpha=0.88, lambda_=2.25, gamma=0.61):
        """
        Initialize Prospect Theory Model
        alpha: diminishing sensitivity parameter
        lambda_: loss aversion parameter
        gamma: probability weighting parameter
        """
        self.alpha = alpha
        self.lambda_ = lambda_
        self.gamma = gamma
    
    def value_function(self, x):
        """Prospect theory value function"""
        return np.where(x >= 0, x**self.alpha, -self.lambda_ * (-x)**self.alpha)
    
    def probability_weight(self, p):
        """Probability weighting function"""
        return p**self.gamma / (p**self.gamma + (1-p)**self.gamma)**(1/self.gamma)
    
    def fit(self, choices, outcomes):
        """Fit model parameters to observed choices and outcomes"""
        def neg_log_likelihood(params):
            self.alpha, self.lambda_, self.gamma = params
            # Calculate choice probabilities
            values = self.value_function(outcomes)
            probs = self.probability_weight(np.ones_like(choices) * 0.5)
            likelihood = np.sum(np.log(probs[choices]))
            return -likelihood
        
        result = minimize(neg_log_likelihood, 
                        x0=[self.alpha, self.lambda_, self.gamma],
                        bounds=[(0.1, 1), (1, 5), (0.1, 1)])
        self.alpha, self.lambda_, self.gamma = result.x
        return result.x

class ReverseLearningSensitivity:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.deck_values = np.zeros(4)
        
    def analyze_choices(self, choices, rewards):
        """Analyze how quickly participants adapt to changing rewards"""
        prediction_errors = []
        adaptability_scores = []
        
        for t, (choice, reward) in enumerate(zip(choices, rewards)):
            # Prediction error
            pe = reward - self.deck_values[choice]
            prediction_errors.append(pe)
            
            # Update deck value
            self.deck_values[choice] += self.learning_rate * pe
            
            # Calculate adaptability score
            if t > 0:
                prev_choice = choices[t-1]
                if reward < 0 and choice != prev_choice:
                    adaptability_scores.append(1)  # Changed after loss
                elif reward < 0 and choice == prev_choice:
                    adaptability_scores.append(0)  # Perseverated after loss
                
        return {
            'mean_prediction_error': np.mean(prediction_errors),
            'adaptability_score': np.mean(adaptability_scores) if adaptability_scores else 0
        }

class RiskSensitivityAnalysis:
    def analyze_risk_patterns(self, choices, outcomes):
        """Analyze risk-taking patterns"""
        # Calculate running statistics
        window_size = 20
        running_risk = []
        running_variance = []
        
        for i in range(len(choices) - window_size + 1):
            window_outcomes = outcomes[i:i+window_size]
            running_risk.append(np.std(window_outcomes))
            running_variance.append(np.var(window_outcomes))
            
        # Analyze deck preferences after losses
        post_loss_choices = []
        for i in range(1, len(choices)):
            if outcomes[i-1] < 0:
                post_loss_choices.append(choices[i])
                
        return {
            'risk_trajectory': running_risk,
            'outcome_variance': running_variance,
            'loss_response_distribution': np.bincount(post_loss_choices, minlength=4) / len(post_loss_choices)
        }

def analyze_behavior(choices, outcomes):
    """Comprehensive behavioral analysis"""
    # Initialize models
    pt_model = ProspectTheoryModel()
    learning_model = ReverseLearningSensitivity()
    risk_model = RiskSensitivityAnalysis()
    
    # Fit and analyze
    pt_params = pt_model.fit(choices, outcomes)
    learning_metrics = learning_model.analyze_choices(choices, outcomes)
    risk_metrics = risk_model.analyze_risk_patterns(choices, outcomes)
    
    # Basic statistics
    deck_preferences = np.bincount(choices, minlength=4) / len(choices)
    mean_return = np.mean(outcomes)
    
    return {
        'prospect_theory_parameters': {
            'alpha': pt_params[0],
            'lambda': pt_params[1],
            'gamma': pt_params[2]
        },
        'learning_metrics': learning_metrics,
        'risk_metrics': risk_metrics,
        'deck_preferences': deck_preferences,
        'mean_return': mean_return
    } 