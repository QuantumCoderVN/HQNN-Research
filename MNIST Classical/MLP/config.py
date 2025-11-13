# config.py

import os
import torch
import torch.nn as nn # Added for specifying activation function

# ====================================================================
# --- 1. DEVICE AND ENVIRONMENT CONFIGURATION ---
# ====================================================================
# Check for CUDA availability to utilize GPU for acceleration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setting this environment variable can prevent a common OpenMP (OMP) library conflict error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ====================================================================
# --- 2. MODEL ARCHITECTURE CONFIGURATION (For MLP) ---
# ====================================================================
# Grouping all hyperparameters related to the model architecture.
MODEL_CONFIG = {
    'input_size': 28 * 28,          # Original MNIST image size (28x28=784) after flattening
    'output_size': 10,              # Number of classes (digits 0-9)
    # Define a 2-hidden-layer MLP: Input -> 128 nodes -> 64 nodes -> Output
    'hidden_layers': [128, 64],
    'activation_fn': nn.ReLU,       # Activation function (e.g., nn.ReLU, nn.Sigmoid, nn.Tanh)
    'use_dropout': True,            # Whether to use dropout regularization
    'dropout_rate': 0.5             # Dropout probability
}


# ====================================================================
# --- 3. TRAINING AND DATA HYPERPARAMETERS ---
# ====================================================================
# Grouping all parameters related to the training process.
TRAINING_CONFIG = {
    'epochs': 50,                   # Number of times to loop through the entire training set
    'batch_size': 128,              # Number of samples per gradient update
    'learning_rate': 0.001,         # Optimizer's learning rate
    # Set to None to use the full dataset, or an integer to limit the samples (for fast testing)
    'n_train_samples': None,
    # Number of test samples is often relative to train, or set to None for full test set
    'n_test_samples': None,
    'img_size': 28                  # We now use the standard 28x28 MNIST size
}


# ====================================================================
# --- 4. OUTPUT AND FILE CONFIGURATION ---
# ====================================================================
# Base directory for all experiment results
BASE_RESULT_DIR = 'results'

# Specific data paths
DATA_DIR = './data'

# File names for saving
LOG_FILE = 'training_log.txt'
MODEL_SAVE_PATH = os.path.join(BASE_RESULT_DIR, 'mlp_mnist_model.pth')
TRAINING_METRICS_PLOT_PATH = os.path.join(BASE_RESULT_DIR, 'training_metrics.png')
PREDICTIONS_PLOT_PATH = os.path.join(BASE_RESULT_DIR, 'test_predictions.png')
CONFUSION_MATRIX_PLOT_PATH = os.path.join(BASE_RESULT_DIR, 'confusion_matrix.png')
CLASSIFICATION_REPORT_FILE = os.path.join(BASE_RESULT_DIR, 'classification_report.txt')

# The final MODEL_SAVE_PATH and other paths will be set dynamically in main.py
# to include a timestamp/experiment name for better tracking.