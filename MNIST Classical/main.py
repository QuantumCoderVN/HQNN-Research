# main.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import time # Import time for better experiment naming

# Import configurations
from config import (
    DEVICE, TRAINING_CONFIG, LOG_FILE, BASE_RESULT_DIR, MODEL_SAVE_PATH,
    MODEL_CONFIG
)
# Import data loading function
from data_loader import get_mnist_data_loaders
# Import model class (now MLP)
from models import MLP
# Import utilities
from utils import (
    setup_results_directory, write_log, plot_predictions, 
    plot_training_metrics, evaluate_and_report, count_parameters
)

def main():
    # --- 1. SETUP & CONFIGURATION ---
    
    # Generate a unique experiment name with a timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_name = f"MLP_MNIST_{timestamp}"
    
    # Initialize the results directory and get the final path
    final_result_dir = setup_results_directory(experiment_name=experiment_name)
    
    # Set KMP_DUPLICATE_LIB_OK via os.environ for stability (though often handled in config.py)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
    
    # Log device info
    write_log(f"Experiment: {experiment_name}")
    write_log(f"Using device: {DEVICE}")
    print(f"Using device: {DEVICE}")
    if DEVICE.type != "cpu":
        torch.cuda.empty_cache()

    # Log hyperparameters for record-keeping
    write_log("\n--- Hyperparameters ---")
    write_log(f"MODEL_CONFIG: {MODEL_CONFIG}")
    write_log(f"TRAINING_CONFIG: {TRAINING_CONFIG}")

    # --- 2. DATA PREPARATION ---
    train_loader, test_loader = get_mnist_data_loaders()

    # --- 3. MODEL DEFINITION & SUMMARY ---
    # Instantiate the MLP model
    model = MLP().to(DEVICE) 
    
    print("\n--- Model Architecture ---")
    print(model)
    write_log(f"\n--- Model Architecture ---\n{model}")

    # Print and log the number of parameters
    classical_params, quantum_params, total_params = count_parameters(model)
    params_message = (f"Number of parameters:\n"
                      f"  - Classical parameters: {classical_params}\n"
                      f"  - Quantum parameters: {quantum_params}\n" # Will be 0 for MLP
                      f"  - Total parameters: {total_params}")
    print(params_message)
    write_log(params_message)

    # --- 4. TRAINING SETUP ---
    NUM_EPOCHS = TRAINING_CONFIG['epochs']
    LEARNING_RATE = TRAINING_CONFIG['learning_rate']
    
    # Loss function: CrossEntropyLoss is standard for multi-class classification
    criterion = nn.CrossEntropyLoss() 
    # Optimizer: Adam is a robust choice
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Lists to store metrics for plotting
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    print("\n--- Training Started ---")
    write_log("\n--- Training Started ---")

    # --- 5. TRAINING LOOP ---
    for epoch in range(NUM_EPOCHS):
        
        # --- Training Phase ---
        model.train() # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
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
        model.eval() # Set model to evaluation mode
        test_loss_epoch = 0.0
        correct_test = 0
        total_test = 0
        all_preds = []
        all_labels = []

        with torch.no_grad(): # Disable gradient calculations during testing
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss_epoch += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                
                # Store predictions and labels for final reporting
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss = test_loss_epoch / len(test_loader)
        test_accuracy = correct_test / total_test
        test_loss_list.append(test_loss)
        test_acc_list.append(test_accuracy)

        # Log metrics for the current epoch
        log_message = (f"Epoch {epoch+1}/{NUM_EPOCHS}: "
                       f"Train Loss = {train_loss:.4f}, Train Accuracy = {train_acc:.4f}, "
                       f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy:.4f}")
        print(log_message)
        write_log(log_message)

    print("--- Training Complete ---")
    write_log("--- Training Complete ---")

    # --- 6. SAVE MODEL AND REPORT RESULTS ---
    
    # Construct the final model save path within the unique results directory
    final_model_save_path = os.path.join(final_result_dir, os.path.basename(MODEL_SAVE_PATH))
    torch.save(model.state_dict(), final_model_save_path)
    print(f"Model weights saved to {final_model_save_path}")
    write_log(f"Model weights saved to {final_model_save_path}")

    # Plotting and Reporting
    plot_predictions(model, test_loader, DEVICE)
    plot_training_metrics(train_loss_list, train_acc_list, test_loss_list, test_acc_list)
    evaluate_and_report(all_labels, all_preds)

if __name__ == "__main__":
    main()