# CIFAR-10/CIFAR-100 CNN Classifier with Checkpointing

**Author**: Hiwa Aziz Abbas

This project implements a Convolutional Neural Network (CNN) for image classification on the **CIFAR-10** and **CIFAR-100** datasets using **PyTorch**. It fulfills the requirements of the Computer Vision Homework, featuring a CNN with 3 convolutional layers, BatchNormalization, MaxPooling, and Dense layers, trained for 30 epochs with a batch size of 64. The project supports checkpointing to resume training, a progress bar, F1 scores in percentages, and detailed data split information.

## Project Structure

- `config.py`: Configuration for dataset paths, model hyperparameters, and training settings.
- `model.py`: Defines the CNN architecture with 3 conv layers, BatchNorm, MaxPooling, and Dense layers.
- `trainer.py`: Handles data loading, preprocessing, training, evaluation, and checkpointing.
- `main.py`: Orchestrates the pipeline (load data, train, evaluate, plot).
- `requirements.txt`: Lists Python dependencies.
- `README.md`: This file.
- `checkpoints/`: Directory for saved checkpoints (ignored by `.gitignore`).

## Features

- **Dataset Support**: Processes CIFAR-10 (PNG images from `train.7z`) and CIFAR-100 (pickle files).
- **CNN Architecture**: 3 convolutional layers (32, 64, 64 filters, 3x3 kernels), BatchNorm, ReLU, 2x2 MaxPooling, 256-unit Dense layer with ReLU, and output layer (10 or 100 classes).
- **Training**: Uses Adam optimizer (learning rate 0.0001), sparse categorical crossentropy loss, 30 epochs, batch size 64.
- **Checkpointing**: Saves model weights, optimizer state, epoch, and F1 scores after each epoch; resumes training from the latest checkpoint.
- **Output**:
  - Progress bar for training epochs (via `tqdm`).
  - Final train/validation/test F1 scores in percentages.
  - Data split info: training/test samples, percentages, number of classes.
  - Loss curve plot.
- **Experiments**: Tests alternative learning rate (0.001) for 10 epochs.
- **Error Handling**: Fixed `UnboundLocalError` for checkpoint resuming when training is complete.

## Requirements

- Python 3.8+
- PyTorch with CUDA (for GPU) or CPU
- Dependencies (listed in `requirements.txt`):