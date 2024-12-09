# Initialize config package 

class Config:
    # Environment parameters
    INITIAL_MONEY = 2000
    MAX_STEPS = 100
    REFERENCE_POINT = 50.0  # Set to average reward of advantageous decks
    
    # Deck parameters
    DECK_A_REWARD = 100
    DECK_A_PENALTY = -250
    DECK_A_PENALTY_PROB = 0.5
    
    DECK_B_REWARD = 100
    DECK_B_PENALTY = -1250
    DECK_B_PENALTY_PROB = 0.1
    
    DECK_C_REWARD = 50
    DECK_C_PENALTY = -50
    DECK_C_PENALTY_PROB = 0.5
    
    DECK_D_REWARD = 50
    DECK_D_PENALTY = -250
    DECK_D_PENALTY_PROB = 0.1
    
    # Prospect theory parameters - Matched to human behavior
    ALPHA = 0.88        # Risk aversion for gains (same as simulated data)
    BETA = 0.88         # Risk aversion for losses
    LAMBDA = 2.25       # Loss aversion (standard prospect theory value)
    
    # Training parameters
    LEARNING_RATE = 5e-5       # Reduced for more stable learning
    BATCH_SIZE = 128           # Increased for better stability
    BUFFER_SIZE = 50000        # Increased to maintain longer history
    LEARNING_STARTS = 1000     # Start learning after building some experience
    GAMMA = 0.99              # Standard discount factor
    
    # Risk sensitivity parameters
    CVAR_ALPHA = 0.2          # Risk sensitivity parameter
    
    # Exploration parameters - Tuned for human-like exploration
    EXPLORATION_BONUS = 0.1     # Moderate exploration bonus
    EXPLORATION_DECAY = 0.997   # Slow decay for sustained exploration
    MIN_EXPLORATION = 0.05      # Lower minimum for convergence
    INITIAL_EXPLORATION = 1.0   # Start with full exploration
    
    # Novelty and curiosity parameters
    NOVELTY_WEIGHT = 0.2        # Increased for better deck sampling
    CURIOSITY_WEIGHT = 0.1      # Moderate curiosity drive
    NOVELTY_DECAY = 0.995       # Slower decay
    CURIOSITY_DECAY = 0.99      # Moderate decay
    MEMORY_SIZE = 2000          # Increased memory for better patterns
    SURPRISE_THRESHOLD = 0.3     # Moderate surprise threshold