# trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import pickle
import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
from model import build_cnn_model
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class Trainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
        self.history = {"loss": [], "val_loss": [], "f1_score": [], "val_f1_score": []}
        self.start_epoch = 0
        self.optimizer = None

    def load_and_preprocess_data(self):
        dataset = self.config.DATASET
        dataset_path = self.config.DATASET_PATHS[dataset]

        if dataset == 'cifar100':
            # Load CIFAR-100 pickle files
            def unpickle(file):
                with open(file, 'rb') as fo:
                    dict = pickle.load(fo, encoding='bytes')
                return dict

            train_file = os.path.join(dataset_path, 'train')
            test_file = os.path.join(dataset_path, 'test')
            train_data = unpickle(train_file)
            test_data = unpickle(test_file)

            x_train = train_data[b'data']
            y_train = train_data[b'fine_labels']
            x_test = test_data[b'data']
            y_test = test_data[b'fine_labels']

            x_train = x_train.reshape(-1, 3, 32, 32).astype('float32') / 255.0
            x_test = x_test.reshape(-1, 3, 32, 32).astype('float32') / 255.0

            x_train = torch.tensor(x_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.long)
            x_test = torch.tensor(x_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.long)

        elif dataset == 'cifar10':
            # Load CIFAR-10 images and labels
            train_dir = os.path.join(dataset_path, 'train')
            labels_file = os.path.join(dataset_path, 'trainLabels.csv')
            test_dir = os.path.join(dataset_path, 'test')

            # Load train labels
            labels_df = pd.read_csv(labels_file)
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            label_to_idx = {name: idx for idx, name in enumerate(class_names)}

            # Load training images
            x_train = []
            y_train = []
            for idx in labels_df['id']:
                img_path = os.path.join(train_dir, f"{idx}.png")
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img).transpose(2, 0, 1)  # To channels-first (3, 32, 32)
                x_train.append(img_array)
                label = labels_df[labels_df['id'] == idx]['label'].values[0]
                y_train.append(label_to_idx[label])

            x_train = np.array(x_train, dtype='float32') / 255.0
            y_train = np.array(y_train, dtype='int64')

            # Split train into train+validation
            train_idx, val_idx = train_test_split(
                range(len(x_train)),
                test_size=self.config.VALIDATION_SPLIT,
                random_state=42
            )
            x_test = x_train[val_idx]
            y_test = y_train[val_idx]
            x_train = x_train[train_idx]
            y_train = y_train[train_idx]

            x_train = torch.tensor(x_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.long)
            x_test = torch.tensor(x_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.long)

        # Print data split information
        total_samples = len(x_train) + len(x_test)
        train_percent = (len(x_train) / total_samples) * 100
        test_percent = (len(x_test) / total_samples) * 100
        num_classes = self.config.NUM_CLASSES[dataset]
        print("Data Split Information:")
        print(f"  Training samples: {len(x_train)} ({train_percent:.2f}%)")
        print(f"  Test samples: {len(x_test)} ({test_percent:.2f}%)")
        print(f"  Total samples: {total_samples}")
        print(f"  Number of classes: {num_classes}")
        print("Training data shape:", x_train.shape)
        print("Test data shape:", x_test.shape)

        return x_train, y_train, x_test, y_test

    def save_checkpoint(self, epoch, train_f1=None, val_f1=None, filename=None):
        if not os.path.exists(self.config.CHECKPOINT_DIR):
            os.makedirs(self.config.CHECKPOINT_DIR)
        if filename is None:
            filename = os.path.join(self.config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'dataset': self.config.DATASET,
            'train_f1': train_f1,
            'val_f1': val_f1
        }
        torch.save(checkpoint, filename)
        print(f"Saved checkpoint: {filename}")

    def load_checkpoint(self, checkpoint_path=None):
        if checkpoint_path is None:
            # Find the latest checkpoint
            checkpoint_dir = self.config.CHECKPOINT_DIR
            if not os.path.exists(checkpoint_dir):
                return False
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
            if not checkpoints:
                return False
            checkpoint_path = os.path.join(checkpoint_dir, max(checkpoints, key=lambda f: int(f.split('_')[-1].split('.')[0])))

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if checkpoint['dataset'] != self.config.DATASET:
            print(f"Warning: Checkpoint dataset ({checkpoint['dataset']}) differs from current dataset ({self.config.DATASET})")
            return False

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.history = checkpoint['history']
        self.train_f1 = checkpoint.get('train_f1', None)
        self.val_f1 = checkpoint.get('val_f1', None)
        print(f"Loaded checkpoint: {checkpoint_path} (epoch {self.start_epoch})")
        return True

    def build_and_compile_model(self):
        self.model = build_cnn_model(
            input_shape=self.config.IMAGE_SHAPE,
            num_classes=self.config.NUM_CLASSES[self.config.DATASET],
            conv_filters=self.config.CONV_FILTERS,
            filter_size=self.config.FILTER_SIZE,
            pool_size=self.config.POOL_SIZE,
            dense_units=self.config.DENSE_UNITS
        )
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)

    def train_model(self, x_train, y_train, x_test, y_test):
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, pin_memory=(self.device.type == "cuda"))
        test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, pin_memory=(self.device.type == "cuda"))

        criterion = nn.CrossEntropyLoss()
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)

        # Try to load checkpoint
        checkpoint_loaded = self.load_checkpoint()

        # If training is complete, use stored F1 scores
        if self.start_epoch >= self.config.EPOCHS and checkpoint_loaded:
            if self.train_f1 is not None and self.val_f1 is not None:
                print(f"Training already complete at epoch {self.start_epoch}")
                print(f"Final Train F1 Score: {self.train_f1:.2f}%")
                print(f"Final Validation F1 Score: {self.val_f1:.2f}%")
                return
            else:
                # Recompute F1 scores if not stored
                print("Recomputing F1 scores from loaded model...")
                self.model.eval()
                train_preds, train_labels = [], []
                val_preds, val_labels = [], []
                with torch.no_grad():
                    for inputs, labels in train_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        _, predicted = torch.max(outputs, 1)
                        train_preds.extend(predicted.cpu().numpy())
                        train_labels.extend(labels.cpu().numpy())
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        _, predicted = torch.max(outputs, 1)
                        val_preds.extend(predicted.cpu().numpy())
                        val_labels.extend(labels.cpu().numpy())
                train_f1 = f1_score(train_labels, train_preds, average='macro') * 100
                val_f1 = f1_score(val_labels, val_preds, average='macro') * 100
                self.history["f1_score"].append(train_f1)
                self.history["val_f1_score"].append(val_f1)
                print(f"Final Train F1 Score: {train_f1:.2f}%")
                print(f"Final Validation F1 Score: {val_f1:.2f}%")
                return

        for epoch in tqdm(range(self.start_epoch, self.config.EPOCHS), desc="Training Epochs"):
            self.model.train()
            train_loss = 0.0
            epoch_train_preds, epoch_train_labels = [], []

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                epoch_train_preds.extend(predicted.cpu().numpy())
                epoch_train_labels.extend(labels.cpu().numpy())

            train_loss /= len(train_dataset)
            self.history["loss"].append(train_loss)

            self.model.eval()
            val_loss = 0.0
            val_preds, val_labels = [], []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            val_loss /= len(test_dataset)
            self.history["val_loss"].append(val_loss)

            # Compute F1 scores for the last epoch
            if epoch == self.config.EPOCHS - 1:
                train_f1 = f1_score(epoch_train_labels, epoch_train_preds, average='macro') * 100
                val_f1 = f1_score(val_labels, val_preds, average='macro') * 100
                self.history["f1_score"].append(train_f1)
                self.history["val_f1_score"].append(val_f1)
                print(f"Final Train F1 Score: {train_f1:.2f}%")
                print(f"Final Validation F1 Score: {val_f1:.2f}%")

            # Save checkpoint with F1 scores
            if (epoch + 1) % self.config.CHECKPOINT_INTERVAL == 0:
                self.save_checkpoint(epoch, train_f1 if epoch == self.config.EPOCHS - 1 else None, val_f1 if epoch == self.config.EPOCHS - 1 else None)

    def plot_loss_curve(self):
        plt.plot(self.history['loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def evaluate_model(self, x_test, y_test):
        test_dataset = TensorDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, pin_memory=(self.device.type == "cuda"))

        self.model.eval()
        test_loss, test_preds, test_labels = 0.0, [], []
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                test_preds.extend(predicted.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        test_loss /= len(test_dataset)
        test_f1 = f1_score(test_labels, test_preds, average='macro') * 100
        print(f"Test F1 Score: {test_f1:.2f}%")
        return test_f1

    def run_experiment(self, x_train, y_train, x_test, y_test):
        exp_model = build_cnn_model(
            input_shape=self.config.IMAGE_SHAPE,
            num_classes=self.config.NUM_CLASSES[self.config.DATASET],
            conv_filters=self.config.CONV_FILTERS,
            filter_size=self.config.FILTER_SIZE,
            pool_size=self.config.POOL_SIZE,
            dense_units=self.config.DENSE_UNITS
        )
        exp_model.to(self.device)

        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, pin_memory=(self.device.type == "cuda"))
        test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, pin_memory=(self.device.type == "cuda"))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(exp_model.parameters(), lr=self.config.EXPERIMENT_LR)

        for epoch in tqdm(range(self.config.EXPERIMENT_EPOCHS), desc="Experiment Epochs"):
            exp_model.train()
            train_loss = 0.0
            epoch_train_preds, epoch_train_labels = [], []

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = exp_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                epoch_train_preds.extend(predicted.cpu().numpy())
                epoch_train_labels.extend(labels.cpu().numpy())

            train_loss /= len(train_dataset)

            exp_model.eval()
            val_loss = 0.0
            val_preds, val_labels = [], []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = exp_model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            if epoch == self.config.EXPERIMENT_EPOCHS - 1:
                train_f1 = f1_score(epoch_train_labels, epoch_train_preds, average='macro') * 100
                val_f1 = f1_score(val_labels, val_preds, average='macro') * 100

        print(f"Experiment Train F1 Score: {train_f1:.2f}%")
        print(f"Experiment Validation F1 Score: {val_f1:.2f}%")
        print("Experiment with learning rate 0.001 completed.")