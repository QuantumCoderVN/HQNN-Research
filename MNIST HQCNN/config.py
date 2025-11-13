# File: config.py

import os
import torch

# ====================================================================
# --- 1. DEVICE AND ENVIRONMENT CONFIGURATION ---
# ====================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ====================================================================
# --- 2. TRAINING AND DATA HYPERPARAMETERS ---
# ====================================================================
TRAINING_CONFIG = {
    'epochs': 5,  ### MODIFIED ### (MNIST is simpler, fewer epochs needed)
    'batch_size': 64,
    'learning_rate': 0.001,
    'n_train_samples': 5000,
    'n_test_samples': 1000,
    'img_size': 28,  ### MODIFIED ### (Standard MNIST image size)
    'num_color_channels': 1  ### MODIFIED ### (MNIST is grayscale)
}

# ====================================================================
# --- 3. MODEL ARCHITECTURE CONFIGURATION ---
# ====================================================================
MODEL_CONFIG = {
    'output_size': 10,
    
    # --- CNN Specifics ---
    ### MODIFIED ### (Input channels is 1 for grayscale, not 3)
    'cnn_channels': [1, 16, 32],  # Input (1), then 16, then 32 output channels
    'cnn_kernel_size': 3,
    'cnn_pool_size': 2,
    'cnn_linear_features': 64, # Features after CNN and before quantum compression

    # --- Quantum Layer Specifics ---
    'n_qubits': 4,
    'n_layers_ansatz': 1
}

# ====================================================================
# --- 4. OUTPUT AND FILE CONFIGURATION ---
# ====================================================================
### MODIFIED ### (Changed all paths from cifar10 to mnist)
BASE_RESULT_DIR = 'results_mnist_hqcnn' 
DATA_DIR = './data_mnist'

# File names for saving
LOG_FILE = 'training_log.txt'
MODEL_SAVE_PATH = os.path.join(BASE_RESULT_DIR, 'hybrid_hqcnn_mnist_model.pth')
QUANTUM_CIRCUIT_PLOT_PATH = os.path.join(BASE_RESULT_DIR, 'quantum_circuit.png')
TRAINING_METRICS_PLOT_PATH = os.path.join(BASE_RESULT_DIR, 'training_metrics.png')
QUANTUM_WEIGHTS_PLOT_PATH = os.path.join(BASE_RESULT_DIR, 'quantum_weights_evolution.png')
PREDICTIONS_PLOT_PATH = os.path.join(BASE_RESULT_DIR, 'test_predictions.png')
CONFUSION_MATRIX_PLOT_PATH = os.path.join(BASE_RESULT_DIR, 'confusion_matrix.png')
CLASSIFICATION_REPORT_FILE = os.path.join(BASE_RESULT_DIR, 'classification_report.txt')