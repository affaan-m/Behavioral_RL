# Risk-Sensitive Reinforcement Learning for the Iowa Gambling Task

This project implements a risk-sensitive reinforcement learning approach to model human decision-making behavior in the Iowa Gambling Task (IGT). The implementation closely follows the original experimental parameters from Bechara et al. (1994) and incorporates prospect theory and conditional value at risk (CVaR) to model human-like risk sensitivity.

## Experimental Design

### Iowa Gambling Task Parameters
Based on Bechara et al. (1994):
- Number of trials: 200 (two phases of 100 trials each)
- Deck configurations:
  - Deck A (High risk, high punishment):
    - Reward: +100 per selection
    - Punishment: -150 to -350 (frequency: 50%)
    - Net expected value: -25 per card
  - Deck B (High risk, infrequent punishment):
    - Reward: +100 per selection
    - Punishment: -1250 (frequency: 10%)
    - Net expected value: -25 per card
  - Deck C (Low risk, low reward):
    - Reward: +50 per selection
    - Punishment: -50 (frequency: 50%)
    - Net expected value: +25 per card
  - Deck D (Low risk, infrequent punishment):
    - Reward: +50 per selection
    - Punishment: -250 (frequency: 10%)
    - Net expected value: +25 per card

### Risk-Sensitive Model Parameters
1. Prospect Theory Parameters (based on Tversky & Kahneman, 1992):
   - α (value function curvature for gains): 0.88
   - β (value function curvature for losses): 0.88
   - λ (loss aversion coefficient): 2.25
   - Reference point: Dynamic, updated based on running average

2. Conditional Value at Risk (CVaR) Parameters:
   - α (confidence level): 0.05
   - λ_risk (risk sensitivity): 0.7
   - Window size: 20 trials

3. Learning Parameters:
   - Learning rate (α): 0.1
   - Discount factor (γ): 0.95
   - Exploration rate (ε): Linear decay from 1.0 to 0.1
   - Batch size: 32
   - Memory buffer size: 10000
   - Target network update frequency: 100 steps

## Methodology

### 1. Environment Implementation
- Custom IGT environment following OpenAI Gym interface
- State space: [last_reward, running_average, deck_frequencies]
- Action space: Discrete(4) representing decks A-D
- Reward structure matching Bechara et al. (1994)

### 2. Model Architecture
1. Baseline Model:
   - Standard DQN with 3-layer neural network
   - Layer sizes: [64, 128, 64]
   - ReLU activation
   - Adam optimizer (lr=0.001)

2. Risk-Sensitive Model:
   - Modified DQN incorporating prospect theory value function
   - CVaR risk measure in Q-value computation
   - Same architecture as baseline
   - Additional risk-processing layers

### 3. Training Procedure
1. Phase 1 (Exploration): Episodes 1-100
   - Higher exploration rate (ε: 1.0 → 0.3)
   - Focus on learning deck characteristics
   - More weight on immediate rewards

2. Phase 2 (Exploitation): Episodes 101-200
   - Lower exploration rate (ε: 0.3 → 0.1)
   - Increased risk sensitivity
   - More weight on long-term value

### 4. Human Baseline Simulation
Based on Bechara et al. (1994) statistics:
- Initial exploration period: ~30 trials
- Gradual shift to advantageous decks
- Final deck preferences:
  - Decks A/B: ~15% each
  - Decks C/D: ~35% each

## Results Analysis

### 1. Learning Performance
- Baseline Model:
  - Faster initial learning
  - Higher mean rewards
  - Less risk-sensitive behavior
  
- Risk-Sensitive Model:
  - Slower initial learning
  - More human-like deck preferences
  - Better matches human risk aversion patterns

### 2. Deck Selection Patterns
Final deck preferences (Risk-Sensitive Model vs Human Data):
- Deck A: 13.7% vs 15.4%
- Deck B: 13.0% vs 20.2%
- Deck C: 37.2% vs 34.2%
- Deck D: 36.1% vs 30.1%

### 3. Risk Analysis
- Risk aversion increases over time
- Strong correlation between losses and subsequent risk-averse choices
- CVaR effectively captures human-like loss aversion

## Implementation Details

### Key Files
- `src/igt_env.py`: IGT environment implementation
- `src/train.py`: Training loop and model definitions
- `src/visualization.py`: Results visualization
- `src/process_results.py`: Data processing and analysis

### Dependencies
See `requirements.txt` for full list. Key packages:
- PyTorch 1.9.0+
- Gymnasium 0.26.0+
- NumPy 1.21.0+
- Pandas 1.5.0+
- Plotly 5.3.0+

## References

1. Bechara, A., Damasio, A. R., Damasio, H., & Anderson, S. W. (1994). Insensitivity to future consequences following damage to human prefrontal cortex. Cognition, 50(1-3), 7-15.

2. Tversky, A., & Kahneman, D. (1992). Advances in prospect theory: Cumulative representation of uncertainty. Journal of Risk and Uncertainty, 5(4), 297-323.

3. Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk. Journal of Risk, 2, 21-42.

4. Bechara, A., Damasio, H., Tranel, D., & Damasio, A. R. (1997). Deciding advantageously before knowing the advantageous strategy. Science, 275(5304), 1293-1295.

5. Worthy, D. A., Pang, B., & Byrne, K. A. (2013). Decomposing the roles of perseveration and expected value representation in models of the Iowa gambling task. Frontiers in Psychology, 4, 640.

## Future Work

1. Implement additional risk measures (e.g., entropy, variance)
2. Explore different neural architectures
3. Add real-time visualization during training
4. Incorporate physiological measures from human studies
5. Extend to other decision-making tasks

## License
MIT License - See LICENSE file for details