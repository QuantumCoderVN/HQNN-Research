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
    'epochs': 50,  # CIFAR-10 is more complex, may need more epochs
    'batch_size': 64,
    'learning_rate': 0.001,
    'n_train_samples': 5000, # Using a larger subset for CIFAR-10
    'n_test_samples': 1000,
    'img_size': 32, # CIFAR-10 images are 32x32
    'num_color_channels': 3 # CIFAR-10 has 3 color channels (RGB)
}

# ====================================================================
# --- 3. MODEL ARCHITECTURE CONFIGURATION ---
# ====================================================================
MODEL_CONFIG = {
    'output_size': 10,  # 10 classes in CIFAR-10
    'fc_hidden_size': 256, # Increased hidden size for a more complex dataset

    # --- Quantum Layer Specifics ---
    'n_qubits': 4,
    'n_layers_ansatz': 1
}

# ====================================================================
# --- 4. OUTPUT AND FILE CONFIGURATION ---
# ====================================================================
BASE_RESULT_DIR = 'results_cifar10_hqnn' # Updated directory name
DATA_DIR = './data_cifar10' # Updated data directory

# File names for saving
LOG_FILE = 'training_log.txt'
MODEL_SAVE_PATH = os.path.join(BASE_RESULT_DIR, 'hybrid_qnn_cifar10_model.pth')
QUANTUM_CIRCUIT_PLOT_PATH = os.path.join(BASE_RESULT_DIR, 'quantum_circuit.png')
TRAINING_METRICS_PLOT_PATH = os.path.join(BASE_RESULT_DIR, 'training_metrics.png')
QUANTUM_WEIGHTS_PLOT_PATH = os.path.join(BASE_RESULT_DIR, 'quantum_weights_evolution.png')
PREDICTIONS_PLOT_PATH = os.path.join(BASE_RESULT_DIR, 'test_predictions.png')
CONFUSION_MATRIX_PLOT_PATH = os.path.join(BASE_RESULT_DIR, 'confusion_matrix.png')
CLASSIFICATION_REPORT_FILE = os.path.join(BASE_RESULT_DIR, 'classification_report.txt')