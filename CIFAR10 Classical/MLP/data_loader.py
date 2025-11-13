# data_loader.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import TRAINING_CONFIG, DATA_DIR, MODEL_CONFIG, IMG_HEIGHT, IMG_WIDTH, NUM_COLOR_CHANNELS

def get_cifar10_data_loaders():
    """
    Downloads and preprocesses the CIFAR-10 dataset, then creates DataLoader objects.
    
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
    transform = transforms.Compose([
        # CIFAR-10 is 32x32, we ensure it's resized if config changes
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD) # Normalize with CIFAR-10 values
    ])

    # --- 2. Load Datasets ---
    # Change from MNIST to CIFAR10
    train_dataset = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=transform)

    # CIFAR-10 uses different attributes (.data, .targets) than MNIST for limiting samples
    if N_TRAIN_SAMPLES is not None and N_TRAIN_SAMPLES < len(train_dataset):
        train_dataset.data = train_dataset.data[:N_TRAIN_SAMPLES]
        train_dataset.targets = train_dataset.targets[:N_TRAIN_SAMPLES]

    if N_TEST_SAMPLES is not None and N_TEST_SAMPLES < len(test_dataset):
        test_dataset.data = test_dataset.data[:N_TEST_SAMPLES]
        test_dataset.targets = test_dataset.targets[:N_TEST_SAMPLES]

    # --- 3. Create DataLoaders ---
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- 4. Logging and Verification ---
    print("\n--- Data Loading Details ---")
    print(f"Dataset: CIFAR-10")
    print(f"Image dimensions: {IMG_HEIGHT}x{IMG_WIDTH}x{NUM_COLOR_CHANNELS}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    return train_loader, test_loader