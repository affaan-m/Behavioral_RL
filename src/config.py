class Config:
    # Environment settings
    ENV_ID = "CustomEnv-v0"
    
    # Training settings
    TOTAL_TIMESTEPS = 100000
    LEARNING_RATE = 5e-5  # Reduced for more stable learning
    BATCH_SIZE = 256  # Much larger batch size for better stability
    GAMMA = 0.99  # Keep high discount factor
    
    # Risk-sensitive settings
    CVAR_ALPHA = 0.3  # Slightly more risk-sensitive
    
    # Prospect theory parameters
    LAMBDA_LOSS = 2.25  # Standard prospect theory value
    ALPHA_GAIN = 0.88  # Keep balanced gain sensitivity
    BETA_LOSS = 0.88  # Keep matching loss sensitivity
    REFERENCE_POINT = 0.0  # Lower reference point to reduce bias
    
    # Exploration and intrinsic motivation
    EXPLORATION_BONUS = 0.3  # Lower exploration bonus
    NOVELTY_WEIGHT = 0.3  # Lower novelty weight
    
    # Experience replay settings
    REPLAY_BUFFER_SIZE = 50000  # Smaller buffer to focus on recent experiences
    MIN_REPLAY_SIZE = 5000  # Smaller initial experience requirement
    TARGET_UPDATE_FREQ = 50  # More frequent target updates
    
    # Exploration settings
    EPSILON_START = 1.0
    EPSILON_END = 0.05  # Very low final exploration
    EPSILON_DECAY = 50000  # Slower decay for better exploration
    
    # Hardware settings
    DEVICE = "cuda"
    N_ENVS = 4
    
    # Logging settings
    LOG_DIR = "./metrics"
    MODEL_DIR = "./checkpoints"
    TRAINING_LOG = "./training.log" 