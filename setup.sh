# Create a new conda environment for RL
conda create -n rl_env python=3.8
conda activate rl_env

# Install core RL libraries
pip install gymnasium  # Modern version of OpenAI Gym
pip install stable-baselines3[extra]  # Popular RL framework
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # PyTorch with CUDA support
pip install wandb  # For experiment tracking
pip install numpy pandas matplotlib  # For data handling and visualization 