# utils.py

import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from config import (
    BASE_RESULT_DIR, LOG_FILE, TRAINING_METRICS_PLOT_PATH,
    PREDICTIONS_PLOT_PATH, CONFUSION_MATRIX_PLOT_PATH,
    CLASSIFICATION_REPORT_FILE, MODEL_CONFIG, TRAINING_CONFIG,
    IMG_HEIGHT, NUM_COLOR_CHANNELS # Added for prediction plot logging
)

# --- 1. Directory and Logging Utilities ---

def setup_results_directory(experiment_name="mlp_cifar10_experiment"):
    """
    Creates or cleans the results directory.
    Uses BASE_RESULT_DIR from config.py.
    """
    global BASE_RESULT_DIR 
    
    # Create a unique path for the current experiment
    RESULT_DIR = os.path.join(BASE_RESULT_DIR, experiment_name) 
    
    if os.path.exists(RESULT_DIR):
        shutil.rmtree(RESULT_DIR)
    os.makedirs(RESULT_DIR)
    
    BASE_RESULT_DIR = RESULT_DIR
    
    print(f"Created and using result directory: {BASE_RESULT_DIR}")
    return RESULT_DIR

def write_log(message, filename=LOG_FILE):
    """
    Writes a message to the log file within the current results directory.
    """
    filepath = os.path.join(BASE_RESULT_DIR, filename)
    with open(filepath, 'a') as f:
        f.write(message + '\n')


# --- 2. Parameter Counting Utility ---

def count_parameters(model):
    """
    Calculates the total number of trainable parameters in the model.
    """
    classical_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = classical_params
    
    return classical_params, 0, total_params


# --- 3. Plotting Utilities ---

def plot_predictions(model, data_loader, device, num_images=10, filename=PREDICTIONS_PLOT_PATH):
    """
    Displays and saves the model's predictions on a sample of images.
    """
    print("\n--- Displaying Predictions ---")
    model.eval()
    
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 3))

    dataiter = iter(data_loader)
    images, labels = next(dataiter)

    images_to_show = images[:num_images].to(device)
    labels_to_show = labels[:num_images].to(device)

    # CIFAR-10 Class names (for better display)
    class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    with torch.no_grad():
        outputs = model(images_to_show)
        _, predicted = torch.max(outputs.data, 1)

    for idx in range(num_images):
        ax = axes[idx]
        img = images_to_show[idx].cpu().numpy().transpose((1, 2, 0)) # CIFAR is (C, H, W), need (H, W, C) for Matplotlib
        
        # Reverse normalization (optional but helps visualization)
        # Note: True denormalization is complex due to the normalize values, but this simple transpose is essential
        # For simplicity in a general utility, we rely on Matplotlib's auto-scaling, but we MUST transpose
        
        ax.imshow(img)

        true_label = labels_to_show[idx].item()
        pred_label = predicted[idx].item()

        is_correct = (true_label == pred_label)
        color = "green" if is_correct else "red" 

        ax.set_title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}", color=color)
        ax.axis('off')

    plt.suptitle(f"Sample Predictions (Image Size: {IMG_HEIGHT}x{IMG_HEIGHT}x{NUM_COLOR_CHANNELS})", y=1.02)
    plt.tight_layout()
    
    filepath = os.path.join(BASE_RESULT_DIR, os.path.basename(filename))
    plt.savefig(filepath)
    plt.close()
    print(f"Predictions plot saved to {filepath}")


def plot_training_metrics(train_loss, train_acc, test_loss, test_acc, filename=TRAINING_METRICS_PLOT_PATH):
    """
    Plots and saves the Loss and Accuracy curves over epochs.
    """
    num_epochs = TRAINING_CONFIG['epochs']
    
    plt.figure(figsize=(12, 5))

    # Plot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_loss, marker='o', markersize=3, linestyle='-', color='b', label='Train Loss')
    plt.plot(range(1, num_epochs + 1), test_loss, marker='o', markersize=3, linestyle='--', color='r', label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Plot 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_acc, marker='o', markersize=3, linestyle='-', color='g', label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), test_acc, marker='o', markersize=3, linestyle='--', color='m', label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    plt.suptitle(f"Model: MLP (CIFAR-10), Epochs: {num_epochs}, LR: {TRAINING_CONFIG['learning_rate']}", y=1.05)
    plt.tight_layout()
    
    filepath = os.path.join(BASE_RESULT_DIR, os.path.basename(filename))
    plt.savefig(filepath)
    plt.close()
    print(f"Training metrics plot saved to {filepath}")


# --- 4. Evaluation and Reporting Utilities ---

def evaluate_and_report(all_labels, all_preds):
    """
    Computes and saves the Confusion Matrix and Classification Report.
    """
    print("\n--- Detailed Evaluation Metrics ---")
    num_classes = MODEL_CONFIG['output_size']
    target_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    filepath_cm = os.path.join(BASE_RESULT_DIR, os.path.basename(CONFUSION_MATRIX_PLOT_PATH))
    plt.savefig(filepath_cm)
    plt.close()
    print(f"Confusion matrix saved to {filepath_cm}")

    # Classification Report
    class_report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
    print("\nClassification Report:\n", class_report)

    filepath_cr = os.path.join(BASE_RESULT_DIR, os.path.basename(CLASSIFICATION_REPORT_FILE))
    with open(filepath_cr, 'w') as f:
        f.write(class_report)
    print(f"Classification report saved to {filepath_cr}")