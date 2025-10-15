# main.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import time 

# Import configurations
from config import (
    DEVICE, TRAINING_CONFIG, LOG_FILE, BASE_RESULT_DIR, MODEL_SAVE_PATH,
    MODEL_CONFIG
)
# Import data loading function (CIFAR-10)
from data_loader import get_cifar10_data_loaders # Changed from get_mnist_data_loaders
# Import model class
from models import MLP
# Import utilities
from utils import (
    setup_results_directory, write_log, plot_predictions, 
    plot_training_metrics, evaluate_and_report, count_parameters
)

def main():
    # --- 1. SETUP & CONFIGURATION ---
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_name = f"MLP_CIFAR10_{timestamp}" # Changed experiment name
    
    final_result_dir = setup_results_directory(experiment_name=experiment_name)
    
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
    
    write_log(f"Experiment: {experiment_name}")
    write_log(f"Using device: {DEVICE}")
    print(f"Using device: {DEVICE}")
    if DEVICE.type != "cpu":
        torch.cuda.empty_cache()

    write_log("\n--- Hyperparameters ---")
    write_log(f"MODEL_CONFIG: {MODEL_CONFIG}")
    write_log(f"TRAINING_CONFIG: {TRAINING_CONFIG}")

    # --- 2. DATA PREPARATION ---
    train_loader, test_loader = get_cifar10_data_loaders() # Changed data loader function

    # --- 3. MODEL DEFINITION & SUMMARY ---
    model = MLP().to(DEVICE) 
    
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

    # --- 4. TRAINING SETUP ---
    NUM_EPOCHS = TRAINING_CONFIG['epochs']
    LEARNING_RATE = TRAINING_CONFIG['learning_rate']
    
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    print("\n--- Training Started ---")
    write_log("\n--- Training Started ---")

    # --- 5. TRAINING LOOP (No change in logic) ---
    for epoch in range(NUM_EPOCHS):
        
        # --- Training Phase ---
        model.train() 
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
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
        test_loss_epoch = 0.0
        correct_test = 0
        total_test = 0
        all_preds = []
        all_labels = []

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
                       f"Train Loss = {train_loss:.4f}, Train Accuracy = {train_acc:.4f}, "
                       f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy:.4f}")
        print(log_message)
        write_log(log_message)

    print("--- Training Complete ---")
    write_log("--- Training Complete ---")

    # --- 6. SAVE MODEL AND REPORT RESULTS ---
    
    final_model_save_path = os.path.join(final_result_dir, os.path.basename(MODEL_SAVE_PATH))
    torch.save(model.state_dict(), final_model_save_path)
    print(f"Model weights saved to {final_model_save_path}")
    write_log(f"Model weights saved to {final_model_save_path}")

    plot_predictions(model, test_loader, DEVICE)
    plot_training_metrics(train_loss_list, train_acc_list, test_loss_list, test_acc_list)
    evaluate_and_report(all_labels, all_preds)

if __name__ == "__main__":
    main()