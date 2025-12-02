# main.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np

# Import configurations
from config import *

# Import data loading function
### THAY ĐỔI 1 ###
from data_loader import get_moon_data_loaders

# Import model class (vẫn là MLP)
from models import MLP

# Import utilities
### THAY ĐỔI 2 ###
from utils import (
    setup_results_directory, write_log, count_parameters,
    plot_training_metrics, evaluate_and_report,
    plot_decision_boundary  # Hàm mới của chúng ta
)

def main():
    # --- 1. SETUP & CONFIGURATION ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_name = f"Moon_MLP_ModelA_{timestamp}"
    final_result_dir = setup_results_directory(experiment_name=experiment_name)
    
    write_log(f"Experiment: {experiment_name}")
    write_log(f"Using device: {DEVICE}")
    print(f"Using device: {DEVICE}")

    # Log hyperparameters
    write_log("\n--- Hyperparameters ---")
    write_log(f"MODEL_CONFIG: {MODEL_CONFIG}")
    write_log(f"DATA_CONFIG: {DATA_CONFIG}")
    write_log(f"TRAINING_CONFIG: {TRAINING_CONFIG}")

    # --- 2. DATA PREPARATION ---
    ### THAY ĐỔI 3 ###
    # Lấy thêm X_test, y_test để vẽ biểu đồ
    train_loader, test_loader, X_test_np, y_test_np = get_moon_data_loaders()

    # --- 3. MODEL DEFINITION & SUMMARY ---
    model = MLP().to(DEVICE) 
    
    print("\n--- Model A Architecture ---")
    print(model)
    write_log(f"\n--- Model A Architecture ---\n{model}")

    # Đếm tham số (sẽ rất nhỏ)
    classical_params, _, total_params = count_parameters(model)
    params_message = (f"Number of parameters: {total_params}")
    print(params_message)
    write_log(params_message)

    # --- 4. TRAINING SETUP ---
    NUM_EPOCHS = TRAINING_CONFIG['epochs']
    LEARNING_RATE = TRAINING_CONFIG['learning_rate']
    
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 5. TRAINING LOOP (Giữ nguyên y hệt) ---
    train_loss_list, train_acc_list = [], []
    test_loss_list, test_acc_list = [], []

    print("\n--- Training Started ---")
    write_log("\n--- Training Started ---")

    for epoch in range(NUM_EPOCHS):
        
        # --- Training Phase ---
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        
        # Dữ liệu bây giờ là 'features' và 'labels'
        for features, labels in train_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(features) # Không cần flatten
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
            for features, labels in test_loader:
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                outputs = model(features)
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

        # Log
        if (epoch + 1) % 20 == 0: # Chỉ in log mỗi 20 epoch
            log_message = (f"Epoch {epoch+1}/{NUM_EPOCHS}: "
                           f"Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, "
                           f"Test Loss = {test_loss:.4f}, Test Acc = {test_accuracy:.4f}")
            print(log_message)
            write_log(log_message)

    print("--- Training Complete ---")
    write_log("--- Training Complete ---")

    # --- 6. SAVE MODEL AND REPORT RESULTS ---
    model_save_path = os.path.join(final_result_dir, "model_A.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model weights saved to {model_save_path}")

    # Plotting and Reporting
    plot_training_metrics(train_loss_list, train_acc_list, test_loss_list, test_acc_list)
    evaluate_and_report(all_labels, all_preds)
    
    ### THAY ĐỔI 4 ###
    # Vẽ biểu đồ quan trọng nhất!
    plot_decision_boundary(model, X_test_np, y_test_np, DEVICE)

if __name__ == "__main__":
    main()