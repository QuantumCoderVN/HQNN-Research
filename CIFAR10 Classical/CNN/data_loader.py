# File: data_loader.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# Import configuration variables
from config import TRAINING_CONFIG, DATA_DIR, IMG_HEIGHT, IMG_WIDTH

def get_cifar10_data_loaders():
    """
    Downloads and preprocesses the CIFAR-10 dataset, then creates DataLoader objects.
    Includes data augmentation for the training set.
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    
    # Extract configurations
    BATCH_SIZE = TRAINING_CONFIG['batch_size']
    N_TRAIN_SAMPLES = TRAINING_CONFIG.get('n_train_samples')
    N_TEST_SAMPLES = TRAINING_CONFIG.get('n_test_samples')
    
    # Standard Mean and STD Dev for CIFAR-10 (3 color channels)
    CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR_STD = [0.2023, 0.1994, 0.2010]

    # --- 1. Define Preprocessing Transformations ---

    # Transformation pipeline for the TRAINING data with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),          # Randomly crop the image
        transforms.RandomHorizontalFlip(p=0.5),      # Randomly flip the image horizontally
        transforms.RandomRotation(15),                 # Randomly rotate the image
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), # Adjust color
        transforms.ToTensor(),                         # Convert image to PyTorch Tensor (scales to 0-1)
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)    # Normalize the tensor
    ])

    # Transformation pipeline for the TEST data (NO augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),    # Ensure image is resized to the target size
        transforms.ToTensor(),                         # Convert to PyTorch Tensor
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)    # Normalize the tensor
    ])

    # --- 2. Load Datasets ---
    # Apply the respective transformations
    train_dataset = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_transform)

    # --- 3. Optional Sample Limiting (For quick testing/debugging) ---
    if N_TRAIN_SAMPLES is not None and N_TRAIN_SAMPLES < len(train_dataset):
        # Limit the number of training samples
        train_dataset.data = train_dataset.data[:N_TRAIN_SAMPLES]
        train_dataset.targets = train_dataset.targets[:N_TRAIN_SAMPLES]

    if N_TEST_SAMPLES is not None and N_TEST_SAMPLES < len(test_dataset):
        # Limit the number of testing samples
        test_dataset.data = test_dataset.data[:N_TEST_SAMPLES]
        test_dataset.targets = test_dataset.targets[:N_TEST_SAMPLES]

    # --- 4. Create DataLoaders ---
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- 5. Logging and Verification ---
    print("\n--- Data Loading Details ---")
    print(f"Dataset: CIFAR-10")
    print(f"Image dimensions: {IMG_HEIGHT}x{IMG_WIDTH}x{3}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    return train_loader, test_loader