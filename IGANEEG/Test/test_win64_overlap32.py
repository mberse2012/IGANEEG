
"""
IGANEEG Model Test Script - Window 64, Overlap 32
Test configuration: Window length 64, Overlap length 32
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from scipy.spatial.distance import pdist, squareform
import pickle
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Model'))

from IGANEEG import IGANEEG
from utils import set_seed, load_sliding_window_data, calculate_mmd


class IGANEEGTester:
    def __init__(self, config_name="win64_overlap32"):
        self.config_name = config_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Configuration parameters
        self.window_size = 64
        self.overlap = 32
        self.noise_dim = 64
        self.sampling_rate = 128

        # Path configuration
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data Preprocessing')
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Train')
        self.results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'results_{config_name}')

        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)

        # Set random seed
        set_seed(42)

    def load_test_data(self):
        """Load test data"""
        print(f"Loading test data: {self.config_name}")

        # Load sliding window data
        data_path = os.path.join(self.data_dir, f'sliding_window_{self.config_name}.pkl')

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file does not exist: {data_path}")

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        X = data['X']  # [N, window_size, channels]
        y = data['y']  # [N]

        print(f"Data shape: X={X.shape}, y={y.shape}")
        print(f"Class distribution: SZ={np.sum(y == 1)}, HC={np.sum(y == 0)}")

        # 7:3 split (consistent with training)
        total_samples = len(X)
        train_size = int(total_samples * 0.7)

        # Test set
        X_test = X[train_size:]
        y_test = y[train_size:]

        print(f"Test set size: {len(X_test)}")
        print(f"Test set class distribution: SZ={np.sum(y_test == 1)}, HC={np.sum(y_test == 0)}")

        return X_test, y_test

    def load_trained_model(self):
        """Load trained model"""
        print(f"Loading trained model: {self.config_name}")

        # Create model
        model = IGANEEG(noise_dim=self.noise_dim).to(self.device)

        # Load model weights
        model_path = os.path.join(self.model_dir, f'results_{self.config_name}', 'best_model.pth')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file does not exist: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Set to evaluation mode
        model.eval()

        print(f"Model loaded successfully, training epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"Best discriminator loss: {checkpoint.get('discriminator_loss', 'Unknown')}")

        return model

    def calculate_mmd(self, real_samples, generated_samples, kernel='rbf', gamma=None):
        """Calculate Maximum Mean Discrepancy (MMD)"""
        # Flatten data
        real_flat = real_samples.reshape(real_samples.shape[0], -1)
        gen_flat = generated_samples.reshape(generated_samples.shape[0], -1)

        # Calculate MMD
        n_real = real_flat.shape[0]
        n_gen = gen_flat.shape[0]

        # Calculate kernel matrix
        if gamma is None:
            gamma = 1.0 / real_flat.shape[1]

        def rbf_kernel(X, Y):
            """RBF kernel function"""
            sq_dist = torch.cdist(X, Y, p=2) ** 2
            return torch.exp(-gamma * sq_dist)

        # Convert to torch tensor
        real_tensor = torch.FloatTensor(real_flat).to(self.device)
        gen_tensor = torch.FloatTensor(gen_flat).to(self.device)

        # Calculate MMD
        K_rr = rbf_kernel(real_tensor, real_tensor)
        K_rg = rbf_kernel(real_tensor, gen_tensor)
        K_gg = rbf_kernel(gen_tensor, gen_tensor)

        mmd = (K_rr.mean() - 2 * K_rg.mean() + K_gg.mean()).item()

        return mmd

    def generate_samples(self, model, n_samples):
        """Generate samples"""
        model.eval()
        generated_samples = []

        with torch.no_grad():
            for i in range(n_samples):
                # Generate random noise
                noise = torch.randn(1, self.noise_dim).to(self.device)

                # Generate sample
                generated = model.generator(noise)
                generated_samples.append(generated.cpu().numpy())

        return np.concatenate(generated_samples, axis=0)

    def evaluate_classification(self, model, X_test, y_test):
        """Evaluate classification performance"""
        model.eval()
        predictions = []

        with torch.no_grad():
            for x in X_test:
                x_tensor = torch.FloatTensor(x).unsqueeze(0).to(self.device)

                # Get discriminator output
                _, disc_output = model(x_tensor)

                # Convert to predicted class
                pred_class = torch.argmax(disc_output, dim=1).item()
                predictions.append(pred_class)

        predictions = np.array(predictions)

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)

        return accuracy, mae, mse, predictions

    def plot_loss_curves(self):
        """Plot loss curves"""
        # Load training history
        history_path = os.path.join(self.model_dir, f'results_{self.config_name}', 'training_history.pkl')

        if not os.path.exists(history_path):
            print("Warning: Training history file does not exist, cannot plot loss curves")
            return

        with open(history_path, 'rb') as f:
            history = pickle.load(f)

        # Plot loss curves
        plt.figure(figsize=(12, 8))

        # Discriminator loss
        plt.subplot(2, 2, 1)
        plt.plot(history['discriminator_loss'], label='Discriminator Loss')
        plt.title('Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Generator loss
        plt.subplot(2, 2, 2)
        plt.plot(history['generator_loss'], label='Generator Loss')
        plt.title('Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Total loss
        plt.subplot(2, 2, 3)
        plt.plot(history['total_loss'], label='Total Loss')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Learning rate
        if 'learning_rate' in history:
            plt.subplot(2, 2, 4)
            plt.plot(history['learning_rate'], label='Learning Rate')
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('LR')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()

        # Save image
        loss_plot_path = os.path.join(self.results_dir, 'loss_curves.png')
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Loss curves saved to: {loss_plot_path}")

    def visualize_samples(self, real_samples, generated_samples, n_samples=5):
        """Visualize real and generated samples"""
        plt.figure(figsize=(15, 8))

        # Select first n_samples
        real_vis = real_samples[:n_samples]
        gen_vis = generated_samples[:n_samples]

        for i in range(n_samples):
            # Real sample
            plt.subplot(2, n_samples, i + 1)
            plt.plot(real_vis[i].T, alpha=0.7)
            plt.title(f'Real Sample {i + 1}')
            plt.xlabel('Time Points')
            plt.ylabel('Amplitude')

            # Generated sample
            plt.subplot(2, n_samples, n_samples + i + 1)
            plt.plot(gen_vis[i].T, alpha=0.7)
            plt.title(f'Generated Sample {i + 1}')
            plt.xlabel('Time Points')
            plt.ylabel('Amplitude')

        plt.tight_layout()

        # Save image
        sample_plot_path = os.path.join(self.results_dir, 'sample_comparison.png')
        plt.savefig(sample_plot_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Sample comparison plot saved to: {sample_plot_path}")

    def run_test(self):
        """Run complete test"""
        print("=" * 60)
        print(f"IGANEEG Model Test - {self.config_name}")
        print("=" * 60)

        try:
            # 1. Load test data
            X_test, y_test = self.load_test_data()

            # 2. Load trained model
            model = self.load_trained_model()

            # 3. Generate samples
            print("\nGenerating samples...")
            n_generate = len(X_test)
            generated_samples = self.generate_samples(model, n_generate)

            # 4. Calculate MMD
            print("\nCalculating MMD...")
            mmd_value = self.calculate_mmd(X_test, generated_samples)

            # 5. Evaluate classification performance
            print("\nEvaluating classification performance...")
            accuracy, mae, mse, predictions = self.evaluate_classification(model, X_test, y_test)

            # 6. Output results
            print("\n" + "=" * 60)
            print("Test Results:")
            print("=" * 60)
            print(f"Maximum Mean Discrepancy (MMD): {mmd_value:.6f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Test sample count: {len(X_test)}")
            print(f"Generated sample count: {len(generated_samples)}")

            # 7. Save results
            results = {
                'config_name': self.config_name,
                'window_size': self.window_size,
                'overlap': self.overlap,
                'mmd': mmd_value,
                'accuracy': accuracy,
                'mae': mae,
                'mse': mse,
                'n_test_samples': len(X_test),
                'n_generated_samples': len(generated_samples),
                'predictions': predictions.tolist(),
                'true_labels': y_test.tolist()
            }

            results_path = os.path.join(self.results_dir, 'test_results.pkl')
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)

            print(f"\nTest results saved to: {results_path}")

            # 8. Plot loss curves
            print("\nPlotting loss curves...")
            self.plot_loss_curves()

            # 9. Visualize samples
            print("\nVisualizing sample comparison...")
            self.visualize_samples(X_test, generated_samples)

            # 10. Save generated samples
            generated_path = os.path.join(self.results_dir, 'generated_samples.npy')
            np.save(generated_path, generated_samples)
            print(f"Generated samples saved to: {generated_path}")

            print("\n" + "=" * 60)
            print("Test completed!")
            print("=" * 60)

            return results

        except Exception as e:
            print(f"Error occurred during testing: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main function"""
    print("IGANEEG Model Test - Window 64, Overlap 32")
    print("=" * 60)

    # Create tester
    tester = IGANEEGTester("win64_overlap32")

    # Run test
    results = tester.run_test()

    if results is not None:
        print("\nTest completed successfully!")
        print(f"Results saved in: {tester.results_dir}")
    else:
        print("\nTest failed!")


if __name__ == "__main__":
    main()
