# utils.py
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from config import BASE_RESULT_DIR, TRAINING_CONFIG, MODEL_CONFIG

# --- 1. Directory and Logging (Keep as-is) ---
def setup_results_directory(experiment_name="moon_mlp_experiment"):
    global BASE_RESULT_DIR
    RESULT_DIR = os.path.join(BASE_RESULT_DIR, experiment_name)
    if os.path.exists(RESULT_DIR):
        shutil.rmtree(RESULT_DIR)
    os.makedirs(RESULT_DIR)
    BASE_RESULT_DIR = RESULT_DIR
    print(f"Created and using result directory: {BASE_RESULT_DIR}")
    return RESULT_DIR

def write_log(message, filename="training_log.txt"):
    filepath = os.path.join(BASE_RESULT_DIR, filename)
    with open(filepath, 'a') as f:
        f.write(message + '\n')

# --- 2. Parameter Counting (Keep as-is) ---
def count_parameters(model):
    classical_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return classical_params, 0, classical_params # (classical, quantum, total)

# --- 3. Training Plot (Keep as-is) ---
def plot_training_metrics(train_loss, train_acc, test_loss, test_acc, filename="training_metrics.png"):
    num_epochs = TRAINING_CONFIG['epochs']
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_loss, 'b-o', markersize=3, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), test_loss, 'r--o', markersize=3, label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True); plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_acc, 'g-o', markersize=3, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), test_acc, 'm--o', markersize=3, label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.grid(True); plt.legend()
    
    filepath = os.path.join(BASE_RESULT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Training metrics plot saved to {filepath}")

# --- 4. Evaluation Report (Keep as-is) ---
def evaluate_and_report(all_labels, all_preds, filename="classification_report.txt"):
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
    class_report = classification_report(all_labels, all_preds, target_names=target_names)
    print("\nClassification Report:\n", class_report)
    filepath_cr = os.path.join(BASE_RESULT_DIR, filename)
    with open(filepath_cr, 'w') as f:
        f.write(class_report)
    print(f"Classification report saved to {filepath_cr}")

# --- 5. NEW: Decision Boundary Plot ---
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
    
    plt.title(f"MLP (Model A) Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    
    filepath = os.path.join(BASE_RESULT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Decision boundary plot saved to {filepath}")