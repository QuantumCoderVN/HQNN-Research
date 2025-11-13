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
IMG_HEIGHT = 32
IMG_WIDTH = 32
NUM_COLOR_CHANNELS = 3 

MODEL_CONFIG = {
    'input_channels': NUM_COLOR_CHANNELS,
    'output_size': 10,       
    # Deeper VGG-style architecture
    'conv_layers': [
        # Block 1: 3x32x32 -> 32x32x32 -> 16x16x32 (Max pooling here)
        {'out_channels': 32, 'kernel_size': 3, 'padding': 1, 'pool_after': False}, 
        {'out_channels': 32, 'kernel_size': 3, 'padding': 1, 'pool_after': True}, 
        # Block 2: 32x16x16 -> 64x16x16 -> 8x8x64
        {'out_channels': 64, 'kernel_size': 3, 'padding': 1, 'pool_after': False},
        {'out_channels': 64, 'kernel_size': 3, 'padding': 1, 'pool_after': True}, 
        # Block 3: 64x8x8 -> 128x8x8 -> 4x4x128
        {'out_channels': 128, 'kernel_size': 3, 'padding': 1, 'pool_after': False},
        {'out_channels': 128, 'kernel_size': 3, 'padding': 1, 'pool_after': True}, 
    ],
    # ADJUSTMENT: Reduced FC layer size to fight overfitting
    'fc_hidden_size': 128,   
    'activation_fn': nn.ReLU,
    'use_dropout': True,
    # ADJUSTMENT: Increased dropout rate for stronger regularization
    'dropout_rate': 0.6 
}


# ====================================================================
# --- 3. TRAINING AND DATA HYPERPARAMETERS ---
# ====================================================================
TRAINING_CONFIG = {
    'epochs': 100, 
    'batch_size': 128,             
    'learning_rate': 0.001,
    # ADJUSTMENT: Added L2 Regularization (Weight Decay)
    'weight_decay': 1e-4, 
    'n_train_samples': None,
    'n_test_samples': None,
    'img_size': 32 
}


# ====================================================================
# --- 4. OUTPUT AND FILE CONFIGURATION ---
# ====================================================================
BASE_RESULT_DIR = 'results_cifar_cnn' # Changed directory name for new experiments
DATA_DIR = './data_cifar'

# File names for saving
LOG_FILE = 'training_log.txt'
MODEL_SAVE_PATH = os.path.join(BASE_RESULT_DIR, 'cnn_cifar10_reg_model.pth')
TRAINING_METRICS_PLOT_PATH = os.path.join(BASE_RESULT_DIR, 'training_metrics.png')
PREDICTIONS_PLOT_PATH = os.path.join(BASE_RESULT_DIR, 'test_predictions.png')
CONFUSION_MATRIX_PLOT_PATH = os.path.join(BASE_RESULT_DIR, 'confusion_matrix.png')
CLASSIFICATION_REPORT_FILE = os.path.join(BASE_RESULT_DIR, 'classification_report.txt')