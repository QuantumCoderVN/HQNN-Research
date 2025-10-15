# File: config.py

import os
import torch

# ====================================================================
# --- 1. DEVICE AND ENVIRONMENT CONFIGURATION ---
# ====================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix for OpenMP (OMP) runtime error

# ====================================================================
# --- 2. TRAINING AND DATA HYPERPARAMETERS ---
# ====================================================================
TRAINING_CONFIG = {
    'epochs': 50,
    'batch_size': 64,
    'learning_rate': 0.001,
    'n_train_samples': 2000,
    'n_test_samples': 400,
    'img_size': 8  # Image size is reduced for faster quantum simulation
}

# ====================================================================
# --- 3. MODEL ARCHITECTURE CONFIGURATION ---
# ====================================================================
MODEL_CONFIG = {
    'output_size': 10,  # Corresponds to 10 digits (0-9)
    'fc_hidden_size': 128,

    # --- Quantum Layer Specifics ---
    'n_qubits': 4,  # Number of qubits in the quantum circuit
    'n_layers_ansatz': 1  # Number of layers in the quantum ansatz (e.g., StronglyEntanglingLayers)
}

# ====================================================================
# --- 4. OUTPUT AND FILE CONFIGURATION ---
# ====================================================================
BASE_RESULT_DIR = 'results_mnist_hqnn'
DATA_DIR = './data_mnist'

# File names for saving
LOG_FILE = 'training_log.txt'
MODEL_SAVE_PATH = os.path.join(BASE_RESULT_DIR, 'hybrid_qnn_model.pth')
QUANTUM_CIRCUIT_PLOT_PATH = os.path.join(BASE_RESULT_DIR, 'quantum_circuit.png')
TRAINING_METRICS_PLOT_PATH = os.path.join(BASE_RESULT_DIR, 'training_metrics.png')
QUANTUM_WEIGHTS_PLOT_PATH = os.path.join(BASE_RESULT_DIR, 'quantum_weights_evolution.png')
PREDICTIONS_PLOT_PATH = os.path.join(BASE_RESULT_DIR, 'test_predictions.png')
CONFUSION_MATRIX_PLOT_PATH = os.path.join(BASE_RESULT_DIR, 'confusion_matrix.png')
CLASSIFICATION_REPORT_FILE = os.path.join(BASE_RESULT_DIR, 'classification_report.txt')