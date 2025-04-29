# config.py
class Config:
    # Dataset settings
    DATASET = 'cifar10'  # Options: 'cifar10', 'cifar100'
    DATASET_PATHS = {
        'cifar10': r"G:\dataset\cifar-10",
        'cifar100': r"G:\dataset\cifar-100-python\cifar-100-python"
    }
    IMAGE_SHAPE = (3, 32, 32)  # Channels-first for PyTorch
    NUM_CLASSES = {
        'cifar10': 10,
        'cifar100': 100
    }

    # Model hyperparameters
    CONV_FILTERS = [32, 64, 64]  # Filters for 3 conv layers
    FILTER_SIZE = (3, 3)
    POOL_SIZE = (2, 2)
    DENSE_UNITS = 256

    # Training hyperparameters
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 64
    EPOCHS = 30
    VALIDATION_SPLIT = 0.1  # 10% of training data for validation

    # Experiment hyperparameters
    EXPERIMENT_LR = 0.001
    EXPERIMENT_EPOCHS = 10

    # Checkpoint settings
    CHECKPOINT_DIR = r"checkpoints"
    CHECKPOINT_INTERVAL = 1  # Save every epoch