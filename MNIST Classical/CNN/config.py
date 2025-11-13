# config.py

import os
import torch
import torch.nn as nn

# ====================================================================
# --- 1. DEVICE AND ENVIRONMENT CONFIGURATION ---
# ====================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ====================================================================
# --- 2. MODEL ARCHITECTURE CONFIGURATION (For CNN) ---
# ====================================================================
# We replace the MLP_CONFIG with a CNN_CONFIG.
# The structure is slightly different, focusing on convolutional and fully-connected layers.
MODEL_CONFIG = {
    'model_type': 'CNN',
    'input_channels': 1,            # MNIST images are grayscale, so 1 input channel
    'output_size': 10,              # Number of classes (digits 0-9)
    
    # Define the convolutional layers
    # Format: (output_channels, kernel_size, stride, padding)
    'conv_layers': [
        (16, 5, 1, 2),  # Layer 1: 16 filters, 5x5 kernel, stride 1, padding 2
        (32, 5, 1, 2)   # Layer 2: 32 filters, 5x5 kernel, stride 1, padding 2
    ],
    
    # Define the pooling layer
    'pool_kernel_size': 2, # 2x2 pooling
    
    # Define the fully-connected (linear) layers after the convolutions
    # Note: The input size for the first FC layer (7*7*32) is calculated
    # automatically in models.py. We just define the hidden sizes.
    'fc_layers': [128],  # One hidden FC layer with 128 nodes
    
    'activation_fn': nn.ReLU,
    'use_dropout': True,
    'dropout_rate': 0.5
}


# ====================================================================
# --- 3. TRAINING AND DATA HYPERPARAMETERS ---
# ====================================================================
# This section remains largely the same as for the MLP
TRAINING_CONFIG = {
    'epochs': 50,                   # CNNs often converge faster, 20 epochs is a good start
    'batch_size': 128,
    'learning_rate': 0.001,
    'n_train_samples': None,        # Use full dataset
    'n_test_samples': None,         # Use full dataset
    'img_size': 28                  # Standard 28x28 size
}


# ====================================================================
# --- 4. OUTPUT AND FILE CONFIGURATION ---
# ====================================================================
BASE_RESULT_DIR = 'results'
DATA_DIR = './data'

# --- IMPORTANT: Update file names to reflect the new model ---
LOG_FILE = 'training_log_cnn.txt'
MODEL_SAVE_PATH = os.path.join(BASE_RESULT_DIR, 'cnn_mnist_model.pth')
TRAINING_METRICS_PLOT_PATH = os.path.join(BASE_RESULT_DIR, 'training_metrics_cnn.png')
PREDICTIONS_PLOT_PATH = os.path.join(BASE_RESULT_DIR, 'test_predictions_cnn.png')
CONFUSION_MATRIX_PLOT_PATH = os.path.join(BASE_RESULT_DIR, 'confusion_matrix_cnn.png')
CLASSIFICATION_REPORT_FILE = os.path.join(BASE_RESULT_DIR, 'classification_report_cnn.txt')