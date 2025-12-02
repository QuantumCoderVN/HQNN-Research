# utils.py
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pennylane as qml
from pennylane import numpy as pnp
from config import BASE_RESULT_DIR, TRAINING_CONFIG, MODEL_CONFIG, Q_CONFIG

# --- 1. Directory and Logging ---
def setup_results_directory(experiment_name="moon_mlp_qnn_experiment"):
    """Creates or cleans the results directory for the current experiment run."""
    global BASE_RESULT_DIR
    RESULT_DIR = os.path.join(BASE_RESULT_DIR, experiment_name)
    if os.path.exists(RESULT_DIR):
        shutil.rmtree(RESULT_DIR)
    os.makedirs(RESULT_DIR)
    BASE_RESULT_DIR = RESULT_DIR
    print(f"Created and using result directory: {BASE_RESULT_DIR}")
    return RESULT_DIR

def write_log(message, filename="training_log.txt"):
    """Writes a message to the log file within the current results directory."""
    filepath = os.path.join(BASE_RESULT_DIR, filename)
    with open(filepath, 'a') as f:
        f.write(message + '\n')

# --- 2. Parameter Counting (HYBRID VERSION) ---
def count_parameters(model):
    """Counts classical and quantum parameters."""
    classical_params = 0
    quantum_params = 0
    for name, param in model.named_parameters():
        if 'q_weights' in name: # Our quantum weights are named 'q_weights'
            quantum_params += param.numel()
        else:
            classical_params += param.numel()
    total_params = classical_params + quantum_params
    return classical_params, quantum_params, total_params

# --- 3. Training Plot ---
def plot_training_metrics(train_loss, train_acc, test_loss, test_acc, filename="training_metrics.png"):
    """Plots and saves the Loss and Accuracy curves over epochs."""
    num_epochs = TRAINING_CONFIG['epochs']
    
    plt.figure(figsize=(12, 5))
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_loss, 'b-o', markersize=3, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), test_loss, 'r--o', markersize=3, label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True); plt.legend()
    
    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_acc, 'g-o', markersize=3, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), test_acc, 'm--o', markersize=3, label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.grid(True); plt.legend()
    
    filepath = os.path.join(BASE_RESULT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Training metrics plot saved to {filepath}")

# --- 4. Evaluation Report ---
def evaluate_and_report(all_labels, all_preds, filename="classification_report.txt"):
    """Computes and saves the Confusion Matrix and Classification Report."""
    print("\n--- Detailed Evaluation Metrics ---")
    target_names = ['Class 0', 'Class 1']
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True')
    filepath_cm = os.path.join(BASE_RESULT_DIR, "confusion_matrix.png")
    plt.savefig(filepath_cm)
    plt.close()
    print(f"Confusion matrix saved to {filepath_cm}")

    # Classification Report
    class_report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
    print("\nClassification Report:\n", class_report)
    filepath_cr = os.path.join(BASE_RESULT_DIR, filename)
    with open(filepath_cr, 'w') as f:
        f.write(class_report)
    print(f"Classification report saved to {filepath_cr}")

# --- 5. Decision Boundary Plot ---
def plot_decision_boundary(model, X, y, device, filename="decision_boundary.png"):
    """
    Plots the decision boundary for a 2D classification model.
    """
    print("Plotting decision boundary...")
    model.eval()
    
    # 1. Create a grid of points
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # 2. Get model predictions for every point in the grid
    grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)
    with torch.no_grad():
        Z = model(grid_tensor)
        Z = torch.max(Z, 1)[1] # Get the index of the max logit
        Z = Z.cpu().numpy().reshape(xx.shape)

    # 3. Plot the contour and the data points
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', cmap=plt.cm.RdYlBu)
    
    plt.title(f"Hybrid QNN (MLP -> QLayer) Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    
    filepath = os.path.join(BASE_RESULT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Decision boundary plot saved to {filepath}")

# --- 6. Plot Quantum Circuit ---
def plot_quantum_circuit(qnode_func, filename="quantum_circuit.png"):
    """Draws and saves the quantum circuit architecture."""
    print("\n--- Plotting Quantum Circuit ---")
    
    # Create dummy inputs and weights
    dummy_inputs = pnp.random.rand(Q_CONFIG['n_qubits'], requires_grad=False)
    dummy_weights = pnp.random.rand(Q_CONFIG['n_layers_ansatz'], 
                                   Q_CONFIG['n_qubits'], 3, requires_grad=False)
    
    fig, ax = qml.draw_mpl(qnode_func)(dummy_inputs, dummy_weights)
    ax.set_title("Quantum Circuit Architecture")
    filepath = os.path.join(BASE_RESULT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Quantum circuit plot saved to {filepath}")