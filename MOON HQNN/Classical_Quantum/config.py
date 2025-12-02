# config.py
import torch
import torch.nn as nn
import os

# --- 1. Device and Environment ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 2. Data Configuration (Moon Dataset) ---
DATA_CONFIG = {
    'n_samples': 500,   # Total number of data points
    'noise': 0.2,        # Amount of noise (0.0 to 1.0)
    'test_size': 0.2,    # 20% of data will be used for testing
    'random_state': 42   # For reproducible results
}

# --- 3. Quantum Layer Configuration ---
Q_CONFIG = {
    # The QLayer will be our 2-class classifier
    'n_qubits': 2,        
    'n_layers_ansatz': 3  # 3 learnable layers
}

# --- 4. Model A Configuration (Classical PRE-PROCESSING) ---
MODEL_CONFIG = {
    'input_size': 2,           # Model A takes the 2 raw features
    'hidden_layers': [4],   # Model A's hidden structure
    # The output of Model A must match the input of the QLayer
    'output_size': Q_CONFIG['n_qubits'], # Model A outputs 2 features
    'activation_fn': nn.ReLU,
    'use_dropout': False
}

# --- 5. Training Configuration ---
TRAINING_CONFIG = {
    'epochs': 300,
    'batch_size': 32,
    'learning_rate': 0.01
}

# --- 6. Output Configuration ---
# New results directory for this experiment
BASE_RESULT_DIR = 'results_moon_hqnn'