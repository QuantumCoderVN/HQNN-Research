# utils.py

import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Import configuration from config.py
from config import (
    BASE_RESULT_DIR, LOG_FILE, TRAINING_METRICS_PLOT_PATH,
    PREDICTIONS_PLOT_PATH, CONFUSION_MATRIX_PLOT_PATH,
    CLASSIFICATION_REPORT_FILE, MODEL_CONFIG, TRAINING_CONFIG
)

# --- 1. Directory and Logging Utilities ---

def setup_results_directory(experiment_name="mlp_experiment"):
    """
    Creates or cleans the results directory.
    Creates a unique directory path based on the base name and experiment name.
    """
    global BASE_RESULT_DIR # Use global to update the base directory for other functions
    
    # Create a unique path for the current experiment
    # For a simple run, we'll just use the base name, but can be extended with a timestamp
    RESULT_DIR = os.path.join(BASE_RESULT_DIR, experiment_name) 
    
    # Clean up previous run if it exists
    if os.path.exists(RESULT_DIR):
        shutil.rmtree(RESULT_DIR)
    os.makedirs(RESULT_DIR)
    
    # Update global path for all subsequent file operations
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
    
    Args:
        model (nn.Module): The PyTorch model (MLP).
        
    Returns:
        tuple: (classical_params, total_params)
    """
    # For a purely classical model, all parameters are classical
    classical_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = classical_params
    
    # Return 0 for quantum params for compatibility with main.py log message
    return classical_params, 0, total_params


# --- 3. Plotting Utilities ---

def plot_predictions(model, data_loader, device, num_images=10, filename=PREDICTIONS_PLOT_PATH):
    """
    Displays and saves the model's predictions on a sample of images.
    """
    print("\n--- Displaying Predictions ---")
    model.eval()
    
    # Adjusting figure size for better display of 10 images
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 3))

    # Get one batch of images
    dataiter = iter(data_loader)
    images, labels = next(dataiter)

    images_to_show = images[:num_images].to(device)
    labels_to_show = labels[:num_images].to(device)

    with torch.no_grad():
        outputs = model(images_to_show)
        # Get the class with the highest probability
        _, predicted = torch.max(outputs.data, 1)

    for idx in range(num_images):
        ax = axes[idx]
        # Images are 1-channel. Squeeze removes the channel dimension (1x28x28 -> 28x28)
        img = images_to_show[idx].cpu().numpy().squeeze()
        ax.imshow(img, cmap='gray')

        true_label = labels_to_show[idx].item()
        pred_label = predicted[idx].item()

        is_correct = (true_label == pred_label)
        color = "green" if is_correct else "red" # Color-code for correct/incorrect predictions

        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
        ax.axis('off')

    plt.suptitle(f"Sample Predictions (Image Size: {TRAINING_CONFIG['img_size']}x{TRAINING_CONFIG['img_size']})", y=1.02)
    plt.tight_layout()
    
    # Ensure the path is correct using the updated base directory
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
    plt.ylabel('Accuracy') # Accuracy is already a ratio (0.0 to 1.0)
    plt.grid(True)
    plt.legend()

    plt.suptitle(f"Model: MLP, Epochs: {num_epochs}, LR: {TRAINING_CONFIG['learning_rate']}", y=1.05)
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
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    filepath_cm = os.path.join(BASE_RESULT_DIR, os.path.basename(CONFUSION_MATRIX_PLOT_PATH))
    plt.savefig(filepath_cm)
    plt.close()
    print(f"Confusion matrix saved to {filepath_cm}")

    # Classification Report
    target_names = [str(i) for i in range(num_classes)]
    class_report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
    print("\nClassification Report:\n", class_report)

    filepath_cr = os.path.join(BASE_RESULT_DIR, os.path.basename(CLASSIFICATION_REPORT_FILE))
    with open(filepath_cr, 'w') as f:
        f.write(class_report)
    print(f"Classification report saved to {filepath_cr}")