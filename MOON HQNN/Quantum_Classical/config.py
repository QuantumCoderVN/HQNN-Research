# config.py
import torch
import torch.nn as nn
import os

# --- 1. Device and Environment ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 2. Data Configuration (Moon Dataset) ---
DATA_CONFIG = {
    'n_samples': 500,
    'noise': 0.2,
    'test_size': 0.2,
    'random_state': 42
}

# --- 3. Quantum Layer Configuration (NEW) ---
Q_CONFIG = {
    'n_qubits': 2,        # Must match MODEL_CONFIG input_size
    'n_layers_ansatz': 3  # Number of learnable layers in the PQC
}

# --- 4. Model A Configuration (Classical Part) ---
# This config now defines the part AFTER the quantum layer
MODEL_CONFIG = {
    'input_size': Q_CONFIG['n_qubits'], # Input to MLP comes from QLayer
    'output_size': 2,
    'hidden_layers': [4],     
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
BASE_RESULT_DIR = 'results_moon_hqnn' # New directory for hybrid model