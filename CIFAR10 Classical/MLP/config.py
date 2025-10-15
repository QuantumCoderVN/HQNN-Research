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
# --- 2. MODEL ARCHITECTURE CONFIGURATION (For MLP) ---
# ====================================================================
IMG_HEIGHT = 32
IMG_WIDTH = 32
NUM_COLOR_CHANNELS = 3 # CIFAR-10 is a color dataset (RGB)

MODEL_CONFIG = {
    # Input size: 32 * 32 * 3 = 3072
    'input_size': IMG_HEIGHT * IMG_WIDTH * NUM_COLOR_CHANNELS,
    'output_size': 10, # 10 classes (airplane, car, bird, etc.)
    # Increased complexity for larger, color images
    'hidden_layers': [512, 128], 
    'activation_fn': nn.ReLU,
    'use_dropout': True,
    'dropout_rate': 0.5 
}


# ====================================================================
# --- 3. TRAINING AND DATA HYPERPARAMETERS ---
# ====================================================================
TRAINING_CONFIG = {
    'epochs': 50, # Increased epochs due to higher complexity
    'batch_size': 256, # Larger batch size is often preferred for GPU/CIFAR
    'learning_rate': 0.001,
    'n_train_samples': None,
    'n_test_samples': None,
    'img_size': 32 # CIFAR standard size
}


# ====================================================================
# --- 4. OUTPUT AND FILE CONFIGURATION ---
# ====================================================================
BASE_RESULT_DIR = 'results_cifar'
DATA_DIR = './data_cifar'

# File names for saving
LOG_FILE = 'training_log.txt'
MODEL_SAVE_PATH = os.path.join(BASE_RESULT_DIR, 'mlp_cifar10_model.pth')
TRAINING_METRICS_PLOT_PATH = os.path.join(BASE_RESULT_DIR, 'training_metrics.png')
PREDICTIONS_PLOT_PATH = os.path.join(BASE_RESULT_DIR, 'test_predictions.png')
CONFUSION_MATRIX_PLOT_PATH = os.path.join(BASE_RESULT_DIR, 'confusion_matrix.png')
CLASSIFICATION_REPORT_FILE = os.path.join(BASE_RESULT_DIR, 'classification_report.txt')