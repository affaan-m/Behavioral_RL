class Config:
    # Environment settings
    ENV_ID = "CustomEnv-v0"
    INITIAL_MONEY = 2000
    
    # Training settings
    TOTAL_TIMESTEPS = 100000
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 128
    GAMMA = 0.99
    
    # Risk-sensitive settings
    CVAR_ALPHA = 0.2
    
    # Prospect theory parameters
    LAMBDA_LOSS = 2.25
    ALPHA_GAIN = 0.88
    BETA_LOSS = 0.88
    REFERENCE_POINT = 50.0  # Set to average reward of advantageous decks
    
    # Exploration settings
    EPSILON_START = 1.0
    EPSILON_END = 0.02
    EPSILON_DECAY = 10000
    EXPLORATION_BONUS = 0.1
    
    # Experience replay settings
    BUFFER_SIZE = 50000
    MIN_BUFFER_SIZE = 1000
    TARGET_UPDATE_FREQ = 100
    
    # Deck configurations
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
    
    # Hardware settings
    DEVICE = "cuda"
    N_ENVS = 4
    
    # Logging settings
    LOG_DIR = "./metrics"
    MODEL_DIR = "./checkpoints"
    TRAINING_LOG = "./training.log" 