
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import os


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters of a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def save_generated_samples(generator: torch.nn.Module,
                           discriminator: torch.nn.Module,
                           num_samples: int = 10,
                           save_path: str = "./generated_samples"):
    """Save generated EEG samples"""
    os.makedirs(save_path, exist_ok=True)

    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        for i in range(num_samples):
            # Generate noise
            noise = torch.randn(1, 64)

            # Sample from feature pool
            if discriminator.feature_pool.get_pool_size() > 0:
                real_features = discriminator.feature_pool.sample_random_features(1)
            else:
                real_features = torch.zeros(1, 128)

            # Generate EEG signal
            generated_eeg = generator(noise, real_features)

            # Save generated signal
            np.save(os.path.join(save_path, f"generated_eeg_{i + 1}.npy"),
                    generated_eeg.cpu().numpy())

            # Plot signal
            plt.figure(figsize=(12, 4))
            plt.plot(generated_eeg.cpu().numpy().flatten())
            plt.title(f"Generated EEG Sample {i + 1}")
            plt.xlabel("Time Points")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.savefig(os.path.join(save_path, f"generated_eeg_{i + 1}.png"))
            plt.close()


def plot_training_curves(d_losses: List[float], g_losses: List[float],
                         save_path: str = "./training_curves.png"):
    """Plot training loss curves"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(d_losses, label='Discriminator Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Discriminator Training Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(g_losses, label='Generator Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Training Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_model(generator: torch.nn.Module,
                   discriminator: torch.nn.Module,
                   test_loader: torch.utils.data.DataLoader,
                   device: torch.device) -> dict:
    """Evaluate model performance"""
    generator.eval()
    discriminator.eval()

    total_d_loss = 0
    total_g_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for real_eeg, labels in test_loader:
            real_eeg = real_eeg.to(device)
            labels = labels.to(device)
            batch_size = real_eeg.size(0)

            # Evaluate discriminator
            real_output = discriminator(real_eeg, training=False)
            d_loss = torch.nn.BCELoss()(real_output[:, 1:2],
                                        torch.ones(batch_size, 1).to(device))
            total_d_loss += d_loss.item()

            # Evaluate generator
            noise = torch.randn(batch_size, 64).to(device)
            if discriminator.feature_pool.get_pool_size() > 0:
                real_features = discriminator.feature_pool.sample_random_features(batch_size)
            else:
                real_features = torch.zeros(batch_size, 128).to(device)

            fake_eeg = generator(noise, real_features)
            fake_eeg_reshaped = fake_eeg.unsqueeze(1).expand(-1, real_eeg.shape[1], -1)
            fake_output = discriminator(fake_eeg_reshaped, training=False)

            g_loss = torch.nn.BCELoss()(fake_output[:, 1:2],
                                        torch.ones(batch_size, 1).to(device))
            total_g_loss += g_loss.item()

            # Calculate accuracy
            predictions = (real_output[:, 1] > 0.5).float()
            correct_predictions += (predictions == labels.float()).sum().item()
            total_predictions += labels.size(0)

    metrics = {
        'discriminator_loss': total_d_loss / len(test_loader),
        'generator_loss': total_g_loss / len(test_loader),
        'accuracy': correct_predictions / total_predictions
    }

    return metrics


def create_directories(paths: List[str]):
    """Create necessary directories"""
    for path in paths:
        os.makedirs(path, exist_ok=True)
        print(f"Directory created: {path}")


def log_model_info(model: torch.nn.Module, model_name: str = "Model"):
    """Log model information"""
    total_params, trainable_params = count_parameters(model)

    print(f"\n{model_name} Information:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model structure:")
    print(model)


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    loss: float,
                    save_path: str):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    print(f"Checkpoint saved: {save_path}")


def load_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    checkpoint_path: str):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"Epoch: {epoch}, Loss: {loss}")

    return epoch, loss
