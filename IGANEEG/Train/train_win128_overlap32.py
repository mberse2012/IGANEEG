
"""
Training script - Window length 128, Overlap 32
Train using data with win128_overlap32 configuration
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')

# Add model path
sys.path.append('../Model')
from IGANEEG import IGANEEG, Discriminator, Generator
from config import Config
from utils import set_seed, count_parameters, plot_training_curves, save_generated_samples


class IGANEEGTrainer:
    """IGANEEG Trainer"""

    def __init__(self, model: IGANEEG, device: torch.device, config_name: str):

        self.model = model
        self.device = device
        self.config_name = config_name

        # Optimizers
        self.optimizer_d = optim.Adam(model.discriminator.parameters(),
                                      lr=Config.LEARNING_RATE_D,
                                      betas=(Config.BETA1, Config.BETA2))
        self.optimizer_g = optim.Adam(model.generator.parameters(),
                                      lr=Config.LEARNING_RATE_G,
                                      betas=(Config.BETA1, Config.BETA2))

        # Learning rate schedulers
        self.scheduler_d = optim.lr_scheduler.StepLR(self.optimizer_d,
                                                     step_size=Config.LR_STEP_SIZE,
                                                     gamma=Config.LR_GAMMA)
        self.scheduler_g = optim.lr_scheduler.StepLR(self.optimizer_g,
                                                     step_size=Config.LR_STEP_SIZE,
                                                     gamma=Config.LR_GAMMA)

        # Loss records
        self.d_losses = []
        self.g_losses = []

        # Create save directory
        self.save_dir = f"./results_{config_name}"
        os.makedirs(self.save_dir, exist_ok=True)

    def load_data(self, data_path: str, dataset_name: str) -> Tuple[DataLoader, DataLoader]:
        """Load training and test data"""
        # Build file path
        data_file = os.path.join(data_path, f"{dataset_name}_{self.config_name}.pkl")

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file does not exist: {data_file}")

        # Load data
        with open(data_file, 'rb') as f:
            data = pickle.load(f)

        X = data['X']  # [sample_count, time_points, feature_count]
        y = data['y']  # [sample_count]

        print(f"Loaded data: {dataset_name}_{self.config_name}")
        print(f"Data shape: X={X.shape}, y={y.shape}")
        print(f"Label distribution: SZ={np.sum(y == 1)}, HC={np.sum(y == 0)}")

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)

        # 7:3 split for training and test sets
        train_size = int(0.7 * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE,
                                  shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE,
                                 shuffle=False, drop_last=False)

        print(f"Training set size: {len(train_dataset)}")
        print(f"Test set size: {len(test_dataset)}")

        return train_loader, test_loader

    def train_discriminator(self, real_eeg: torch.Tensor, batch_size: int) -> float:
        """Train discriminator for one batch"""
        self.optimizer_d.zero_grad()

        # Real samples
        real_labels = torch.ones(batch_size, 1).to(self.device)
        real_output = self.model.discriminator(real_eeg, training=True)
        d_loss_real = nn.BCELoss()(real_output[:, 1:2], real_labels)

        # Generated samples
        noise = torch.randn(batch_size, Config.NOISE_SIZE).to(self.device)

        # Sample real features from feature pool
        if self.model.discriminator.feature_pool.get_pool_size() > 0:
            real_features = self.model.discriminator.feature_pool.sample_random_features(batch_size)
        else:
            real_features = torch.zeros(batch_size, Config.FEATURE_SIZE).to(self.device)

        fake_eeg = self.model.generator(noise, real_features)

        # Convert generated EEG to format required by discriminator
        fake_eeg_reshaped = fake_eeg.unsqueeze(1).expand(-1, real_eeg.shape[1], -1)

        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        fake_output = self.model.discriminator(fake_eeg_reshaped.detach(), training=False)
        d_loss_fake = nn.BCELoss()(fake_output[:, 1:2], fake_labels)

        # Total loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.optimizer_d.step()

        return d_loss.item()

    def train_generator(self, batch_size: int) -> float:
        """Train generator for one batch"""
        self.optimizer_g.zero_grad()

        # Generate samples
        noise = torch.randn(batch_size, Config.NOISE_SIZE).to(self.device)

        # Sample real features from feature pool
        if self.model.discriminator.feature_pool.get_pool_size() > 0:
            real_features = self.model.discriminator.feature_pool.sample_random_features(batch_size)
        else:
            real_features = torch.zeros(batch_size, Config.FEATURE_SIZE).to(self.device)

        fake_eeg = self.model.generator(noise, real_features)

        # Hope generator's generated samples are considered real by discriminator
        labels = torch.ones(batch_size, 1).to(self.device)

        # Convert generated EEG to format required by discriminator
        fake_eeg_reshaped = fake_eeg.unsqueeze(1).expand(-1, 128, -1)

        output = self.model.discriminator(fake_eeg_reshaped, training=False)
        g_loss = nn.BCELoss()(output[:, 1:2], labels)

        g_loss.backward()
        self.optimizer_g.step()

        return g_loss.item()

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        total_d_loss = 0
        total_g_loss = 0
        num_batches = 0

        for batch_idx, (real_eeg, _) in enumerate(train_loader):
            real_eeg = real_eeg.to(self.device)
            batch_size = real_eeg.size(0)

            # Train discriminator
            d_loss = self.train_discriminator(real_eeg, batch_size)

            # Train generator
            g_loss = self.train_generator(batch_size)

            total_d_loss += d_loss
            total_g_loss += g_loss
            num_batches += 1

            if batch_idx % 10 == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}, D_Loss: {d_loss:.4f}, G_Loss: {g_loss:.4f}')

        avg_d_loss = total_d_loss / num_batches
        avg_g_loss = total_g_loss / num_batches

        return avg_d_loss, avg_g_loss

    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluate model performance"""
        self.model.eval()

        total_d_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for real_eeg, labels in test_loader:
                real_eeg = real_eeg.to(self.device)
                labels = labels.to(self.device)
                batch_size = real_eeg.size(0)

                # Discriminator evaluation
                real_output = self.model.discriminator(real_eeg, training=False)
                d_loss = nn.BCELoss()(real_output[:, 1:2],
                                      torch.ones(batch_size, 1).to(self.device))
                total_d_loss += d_loss.item()

                # Calculate accuracy
                predictions = (real_output[:, 1] > 0.5).float()
                correct_predictions += (predictions == labels.float()).sum().item()
                total_predictions += labels.size(0)

        metrics = {
            'discriminator_loss': total_d_loss / len(test_loader),
            'accuracy': correct_predictions / total_predictions
        }

        return metrics

    def train(self, train_loader: DataLoader, test_loader: DataLoader, epochs: int = Config.EPOCHS):
        """Train the model"""
        print(f"\nStarting training for {self.config_name} configuration model...")
        print("=" * 60)

        best_accuracy = 0.0

        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            print('-' * 40)

            # Training phase
            self.model.train()
            avg_d_loss, avg_g_loss = self.train_epoch(train_loader)

            # Record losses
            self.d_losses.append(avg_d_loss)
            self.g_losses.append(avg_g_loss)

            # Update learning rates
            self.scheduler_d.step()
            self.scheduler_g.step()

            print(f'Training completed - D_Loss: {avg_d_loss:.4f}, G_Loss: {avg_g_loss:.4f}')

            # Evaluation phase
            if epoch % 10 == 0:
                metrics = self.evaluate(test_loader)
                print(f'Evaluation - D_Loss: {metrics["discriminator_loss"]:.4f}, Accuracy: {metrics["accuracy"]:.4f}')

                # Save best model
                if metrics['accuracy'] > best_accuracy:
                    best_accuracy = metrics['accuracy']
                    self.save_model(epoch, is_best=True)

            # Regular model saving
            if epoch % 20 == 0:
                self.save_model(epoch)

            # Save generated samples
            if epoch % 30 == 0:
                save_generated_samples(self.model.generator, self.model.discriminator,
                                       num_samples=5, save_path=f"{self.save_dir}/epoch_{epoch}")

        print(f"\nTraining completed! Best accuracy: {best_accuracy:.4f}")

        # Save final results
        self.plot_losses()
        self.save_final_results()

    def save_model(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        if is_best:
            save_path = f'{self.save_dir}/best_model.pth'
        else:
            save_path = f'{self.save_dir}/model_epoch_{epoch}.pth'

        torch.save({
            'epoch': epoch,
            'discriminator_state_dict': self.model.discriminator.state_dict(),
            'generator_state_dict': self.model.generator.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'd_losses': self.d_losses,
            'g_losses': self.g_losses,
            'config': self.config_name
        }, save_path)

        if is_best:
            print(f'Best model saved: {save_path}')

    def plot_losses(self):
        """Plot loss curves"""
        plot_training_curves(self.d_losses, self.g_losses,
                             save_path=f"{self.save_dir}/training_curves.png")

    def save_final_results(self):
        """Save final results"""
        results = {
            'config': self.config_name,
            'd_losses': self.d_losses,
            'g_losses': self.g_losses,
            'final_d_loss': self.d_losses[-1] if self.d_losses else None,
            'final_g_loss': self.g_losses[-1] if self.g_losses else None
        }

        with open(f"{self.save_dir}/training_results.pkl", 'wb') as f:
            pickle.dump(results, f)

        print(f"Training results saved: {self.save_dir}/training_results.pkl")


def main():
    """Main function"""
    # Set random seed
    set_seed(42)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration information
    config_name = "win128_overlap32"
    data_path = "../Data Preprocessing/sliding_window_results"

    # Create model
    model = IGANEEG().to(device)

    # Print model information
    total_params, trainable_params = count_parameters(model)
    print(f"Model total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = IGANEEGTrainer(model, device, config_name)

    # Load data
    try:
        # Try to load NNCI data
        train_loader, test_loader = trainer.load_data(data_path, "NNCI")

        # Start training
        trainer.train(train_loader, test_loader)

    except FileNotFoundError as e:
        print(f"Data loading failed: {e}")
        print("Please ensure data preprocessing script has been run and sliding window data has been generated")
    except Exception as e:
        print(f"Error during training: {e}")


if __name__ == "__main__":
    main()
