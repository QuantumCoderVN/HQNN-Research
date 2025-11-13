# File: data_loader.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from config import TRAINING_CONFIG, DATA_DIR

def get_cifar10_data_loaders():
    """
    Downloads and preprocesses the CIFAR-10 dataset, then creates DataLoader objects.

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Extract configurations
    BATCH_SIZE = TRAINING_CONFIG['batch_size']
    N_TRAIN_SAMPLES = TRAINING_CONFIG['n_train_samples']
    N_TEST_SAMPLES = TRAINING_CONFIG['n_test_samples']
    IMG_SIZE = TRAINING_CONFIG['img_size']

    # Mean and std for CIFAR-10
    CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR_STD = [0.2470, 0.2435, 0.2616]

    # --- 1. Define Preprocessing Transformations ---
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])

    # --- 2. Load Datasets ---
    # Ensure we are loading CIFAR10
    train_dataset = datasets.CIFAR10(DATA_DIR, train=True, download=False, transform=transform)
    test_dataset = datasets.CIFAR10(DATA_DIR, train=False, download=False, transform=transform)

    # --- 3. Limit number of samples using Subset ---
    train_indices = list(range(len(train_dataset)))
    test_indices = list(range(len(test_dataset)))
    
    train_subset = Subset(train_dataset, train_indices[:N_TRAIN_SAMPLES])
    test_subset = Subset(test_dataset, test_indices[:N_TEST_SAMPLES])

    # --- 4. Create DataLoaders ---
    train_loader = DataLoader(dataset=train_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_subset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 5. Logging and Verification ---
    print("\n--- Data Loading Details ---")
    print(f"Dataset: CIFAR-10") # This should now print CIFAR-10
    print(f"Image dimensions: {IMG_SIZE}x{IMG_SIZE}x{TRAINING_CONFIG['num_color_channels']}")
    print(f"Number of training samples: {len(train_subset)}")
    print(f"Number of test samples: {len(test_subset)}")

    return train_loader, test_loader