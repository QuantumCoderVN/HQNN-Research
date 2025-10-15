# File: utils.py

import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pennylane as qml
from pennylane import numpy as pnp
from config import *

# --- 1. Directory and Logging Utilities ---

def setup_results_directory(experiment_name="hqnn_mnist_experiment"):
    """Creates or cleans the results directory for the current experiment run."""
    global BASE_RESULT_DIR
    RESULT_DIR = os.path.join(BASE_RESULT_DIR, experiment_name)
    if os.path.exists(RESULT_DIR):
        shutil.rmtree(RESULT_DIR)
    os.makedirs(RESULT_DIR)
    BASE_RESULT_DIR = RESULT_DIR
    print(f"Created and using result directory: {BASE_RESULT_DIR}")
    return RESULT_DIR

def write_log(message, filename=LOG_FILE):
    """Writes a message to the log file within the current results directory."""
    filepath = os.path.join(BASE_RESULT_DIR, filename)
    with open(filepath, 'a') as f:
        f.write(message + '\n')

# --- 2. Parameter Counting Utility ---

def count_parameters(model):
    """Counts classical, quantum, and total parameters of the HybridQNN model."""
    classical_params, quantum_params = 0, 0
    for name, param in model.named_parameters():
        if 'quantum_layer' in name:
            quantum_params += param.numel()
        else:
            classical_params += param.numel()
    total_params = classical_params + quantum_params
    return classical_params, quantum_params, total_params

# --- 3. Plotting Utilities ---

def plot_quantum_circuit(qnode_func, filename=QUANTUM_CIRCUIT_PLOT_PATH):
    """Draws and saves the quantum circuit architecture."""
    print("\n--- Plotting Quantum Circuit ---")
    dummy_inputs = pnp.random.rand(MODEL_CONFIG['n_qubits'], requires_grad=False)
    dummy_weights = pnp.random.rand(MODEL_CONFIG['n_layers_ansatz'], MODEL_CONFIG['n_qubits'], 3, requires_grad=False)
    
    fig, ax = qml.draw_mpl(qnode_func)(dummy_inputs, dummy_weights)
    ax.set_title("Quantum Circuit Architecture")
    filepath = os.path.join(BASE_RESULT_DIR, os.path.basename(filename))
    plt.savefig(filepath)
    plt.close()
    print(f"Quantum circuit plot saved to {filepath}")

def plot_predictions(model, data_loader, device, num_images=10, filename=PREDICTIONS_PLOT_PATH):
    """Displays and saves the model's predictions on a sample of images."""
    print("\n--- Displaying Predictions ---")
    model.eval()
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 3))
    images, labels = next(iter(data_loader))
    images_to_show, labels_to_show = images[:num_images].to(device), labels[:num_images].to(device)

    with torch.no_grad():
        outputs = model(images_to_show)
        _, predicted = torch.max(outputs.data, 1)

    for idx in range(num_images):
        ax = axes[idx]
        ax.imshow(images_to_show[idx].cpu().numpy().squeeze(), cmap='gray')
        color = "green" if (labels_to_show[idx] == predicted[idx]) else "red"
        ax.set_title(f"True: {labels_to_show[idx].item()}\nPred: {predicted[idx].item()}", color=color)
        ax.axis('off')

    filepath = os.path.join(BASE_RESULT_DIR, os.path.basename(filename))
    plt.savefig(filepath)
    plt.close()
    print(f"Predictions plot saved to {filepath}")

def plot_training_metrics(train_loss, train_acc, test_loss, test_acc, filename=TRAINING_METRICS_PLOT_PATH):
    """Plots and saves the Loss and Accuracy curves over epochs."""
    num_epochs = TRAINING_CONFIG['epochs']
    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_loss, 'b-o', markersize=3, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), test_loss, 'r--o', markersize=3, label='Test Loss')
    plt.title('Training and Test Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.grid(True); plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_acc, 'g-o', markersize=3, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), test_acc, 'm--o', markersize=3, label='Test Accuracy')
    plt.title('Training and Test Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.grid(True); plt.legend()

    filepath = os.path.join(BASE_RESULT_DIR, os.path.basename(filename))
    plt.savefig(filepath)
    plt.close()
    print(f"Training metrics plot saved to {filepath}")

def plot_quantum_weights_evolution(history, filename=QUANTUM_WEIGHTS_PLOT_PATH):
    """Plots the evolution of quantum parameters over epochs."""
    num_epochs = TRAINING_CONFIG['epochs']
    plt.figure(figsize=(10, 6))
    plt.title('Quantum Layer Weights Evolution'); plt.xlabel('Epoch'); plt.ylabel('Weight Value')
    
    weights_array = np.array(history)
    reshaped_weights = weights_array.reshape(num_epochs, -1)

    for i in range(reshaped_weights.shape[1]):
        plt.plot(range(1, num_epochs + 1), reshaped_weights[:, i])

    filepath = os.path.join(BASE_RESULT_DIR, os.path.basename(filename))
    plt.savefig(filepath)
    plt.close()
    print(f"Quantum weights evolution plot saved to {filepath}")

# --- 4. Evaluation and Reporting Utilities ---

def evaluate_and_report(all_labels, all_preds):
    """Computes and saves the Confusion Matrix and Classification Report."""
    print("\n--- Detailed Evaluation Metrics ---")
    target_names = [str(i) for i in range(MODEL_CONFIG['output_size'])]

    # Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix'); plt.xlabel('Predicted Label'); plt.ylabel('True Label')
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