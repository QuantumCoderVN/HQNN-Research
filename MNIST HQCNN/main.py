# File: main.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import time

# Import configurations
from config import *
# Import data loading function
### MODIFIED ###
from data_loader import get_mnist_data_loaders 
# Import model classes
from models import HybridQNN, quantum_circuit
# Import utilities
from utils import *

def main():
    # --- 1. SETUP & CONFIGURATION ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_name = f"HQCNN_MNIST_{timestamp}" 
    final_result_dir = setup_results_directory(experiment_name=experiment_name)

    write_log(f"Experiment: {experiment_name}")
    write_log(f"Using device: {DEVICE}")
    print(f"Using device: {DEVICE}")
    if DEVICE.type != "cpu":
        torch.cuda.empty_cache()

    # --- 2. DATA PREPARATION ---
    train_loader, test_loader = get_mnist_data_loaders()

    # --- 3. MODEL DEFINITION & SUMMARY ---
    model = HybridQNN().to(DEVICE)
    print("\n--- Model Architecture ---")
    print(model)
    write_log(f"\n--- Model Architecture ---\n{model}")

    classical_params, quantum_params, total_params = count_parameters(model)
    params_message = (f"Number of parameters:\n"
                      f"  - Classical parameters: {classical_params}\n"
                      f"  - Quantum parameters: {quantum_params}\n"
                      f"  - Total parameters: {total_params}")
    print(params_message)
    write_log(params_message)

    plot_quantum_circuit(qnode_func=quantum_circuit)

    # --- 4. TRAINING SETUP ---
    NUM_EPOCHS = TRAINING_CONFIG['epochs']
    LEARNING_RATE = TRAINING_CONFIG['learning_rate']
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Lists to store metrics for plotting
    train_loss_list, train_acc_list = [], []
    test_loss_list, test_acc_list = [], []
    quantum_weights_history = []

    print("\n--- Training Started ---")
    write_log("\n--- Training Started ---")

    # --- 5. TRAINING LOOP ---
    # (No changes needed in the training loop)
    for epoch in range(NUM_EPOCHS):
        if hasattr(model, 'quantum_layer'):
            quantum_weights_history.append(model.quantum_layer.weights.detach().cpu().numpy().copy())
        
        # --- Training Phase ---
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # --- Validation (Testing) Phase ---
        model.eval()
        test_loss_epoch, correct_test, total_test = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss_epoch += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss = test_loss_epoch / len(test_loader)
        test_accuracy = correct_test / total_test
        test_loss_list.append(test_loss)
        test_acc_list.append(test_accuracy)

        log_message = (f"Epoch {epoch+1}/{NUM_EPOCHS}: "
                       f"Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, "
                       f"Test Loss = {test_loss:.4f}, Test Acc = {test_accuracy:.4f}")
        print(log_message)
        write_log(log_message)

    print("--- Training Complete ---")
    write_log("--- Training Complete ---")

    # --- 6. SAVE MODEL AND REPORT RESULTS ---
    final_model_path = os.path.join(final_result_dir, os.path.basename(MODEL_SAVE_PATH))
    torch.save(model.state_dict(), final_model_path)
    print(f"Model weights saved to {final_model_path}")
    write_log(f"Model weights saved to {final_model_path}")

    # Plotting and Reporting
    plot_predictions(model, test_loader, DEVICE)
    plot_training_metrics(train_loss_list, train_acc_list, test_loss_list, test_acc_list)
    if quantum_weights_history:
        plot_quantum_weights_evolution(quantum_weights_history)
    evaluate_and_report(all_labels, all_preds)

if __name__ == "__main__":
    main()