# config.py
import torch
import torch.nn as nn
import os

# --- 1. Device and Environment ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 2. Data Configuration (NEW for Moon Dataset) ---
DATA_CONFIG = {
    'n_samples': 500,   # Total number of data points
    'noise': 0.2,        # Amount of noise (0.0 to 1.0)
    'test_size': 0.2,    # 20% of data will be used for testing
    'random_state': 42   # For reproducible results
}

# --- 3. Model A Configuration (Minimal MLP) ---
MODEL_CONFIG = {
    'input_size': 2,           # Moon dataset has 2 features (x, y)
    'output_size': 2,          # Binary classification (class 0, class 1)
    'hidden_layers': [4],     # MODEL A: Minimal hidden layers 
    'activation_fn': nn.ReLU,
    'use_dropout': False       # Not needed for such a small model
}

# --- 4. Training Configuration ---
TRAINING_CONFIG = {
    'epochs': 300,
    'batch_size': 32,
    'learning_rate': 0.01
}

# --- 5. Output Configuration ---
BASE_RESULT_DIR = 'results_moon_mlp'
# (Các tên file cụ thể sẽ được tạo trong utils.py hoặc main.py)