
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
import pywt
from sklearn.decomposition import FastICA


class L1FeatureSelector(nn.Module):
    """L1 Feature Selector"""

    def __init__(self, input_size: int, sparsity_ratio: float = 0.5):
        super(L1FeatureSelector, self).__init__()
        self.input_size = input_size
        self.sparsity_ratio = sparsity_ratio

        # Learnable weight parameters
        self.weights = nn.Parameter(torch.ones(input_size))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Calculate L1 regularization weights
        l1_weights = torch.abs(self.weights)

        # Determine number of features to retain
        k = max(1, int(self.input_size * self.sparsity_ratio))

        # Get top-k feature indices
        _, top_indices = torch.topk(l1_weights, k, largest=True)

        # Create mask
        mask = torch.zeros_like(l1_weights)
        mask[top_indices] = 1.0

        # Apply mask
        selected_features = x * mask

        return selected_features, mask


class WPTFeatureExtractor(nn.Module):
    """Wavelet Packet Transform Feature Extractor"""

    def __init__(self, wavelet='db4', level=3):

        super(WPTFeatureExtractor, self).__init__()
        self.wavelet = wavelet
        self.level = level

    def extract_wpt_features(self, eeg_signal: torch.Tensor) -> torch.Tensor:

        batch_size, time_points, channels = eeg_signal.shape
        wpt_features = []

        for b in range(batch_size):
            sample_features = []
            for ch in range(channels):
                signal_data = eeg_signal[b, :, ch].detach().cpu().numpy()

                # Wavelet packet decomposition
                wp = pywt.WaveletPacket(signal_data, wavelet=self.wavelet, mode='symmetric', maxlevel=self.level)

                # Extract all node energies
                nodes = [node.path for node in wp.get_level(self.level, 'natural')]
                node_features = []

                for node in nodes:
                    node_data = wp[node].data
                    energy = np.sum(node_data ** 2)
                    node_features.append(energy)

                sample_features.extend(node_features)

            wpt_features.append(sample_features)

        return torch.tensor(wpt_features, dtype=torch.float32, device=eeg_signal.device)


class ICAFeatureExtractor(nn.Module):
    """ICA Feature Extractor"""

    def __init__(self, n_components: int = 16):

        super(ICAFeatureExtractor, self).__init__()
        self.n_components = n_components
        self.ica = FastICA(n_components=n_components, random_state=42)

    def extract_ica_features(self, eeg_signal: torch.Tensor) -> torch.Tensor:

        batch_size, time_points, channels = eeg_signal.shape
        ica_features = []

        for b in range(batch_size):
            # Convert to numpy format
            signal_data = eeg_signal[b].detach().cpu().numpy()  # [time_points, channels]

            # Apply ICA
            try:
                # If channel count is greater than component count, use PCA for dimensionality reduction
                if channels > self.n_components:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=self.n_components)
                    signal_reduced = pca.fit_transform(signal_data)
                else:
                    signal_reduced = signal_data

                # Apply ICA
                ica_result = self.ica.fit_transform(signal_reduced)

                # Calculate statistical features
                features = []
                for comp in range(ica_result.shape[1]):
                    comp_signal = ica_result[:, comp]
                    features.extend([
                        np.mean(comp_signal),
                        np.std(comp_signal),
                        np.max(comp_signal),
                        np.min(comp_signal)
                    ])

                # Ensure consistent feature length
                target_length = 64  # Target feature length
                if len(features) < target_length:
                    features.extend([0.0] * (target_length - len(features)))
                else:
                    features = features[:target_length]

                ica_features.append(features)

            except Exception as e:
                print(f"ICA feature extraction failed: {e}")
                # Use zero features as fallback
                ica_features.append([0.0] * 64)

        return torch.tensor(ica_features, dtype=torch.float32, device=eeg_signal.device)


class RealFeatureCombinationPool:
    """Real Feature Combination Pool"""

    def __init__(self, max_size: int = 1000):

        self.max_size = max_size
        self.feature_pool = []
        self.current_size = 0

    def add_features(self, features: torch.Tensor):

        for i in range(features.shape[0]):
            if self.current_size < self.max_size:
                self.feature_pool.append(features[i].clone())
                self.current_size += 1
            else:
                # Random replacement
                idx = np.random.randint(0, self.max_size)
                self.feature_pool[idx] = features[i].clone()

    def sample_random_features(self, batch_size: int) -> torch.Tensor:

        if self.current_size == 0:
            raise ValueError("Feature pool is empty")

        indices = np.random.choice(self.current_size, batch_size, replace=True)
        sampled_features = torch.stack([self.feature_pool[i] for i in indices])

        return sampled_features

    def get_pool_size(self) -> int:
        """Get current pool size"""
        return self.current_size


class Discriminator(nn.Module):
    """Discriminator Module"""

    def __init__(self, input_size: int, wpt_feature_size: int = 64, ica_feature_size: int = 64):

        super(Discriminator, self).__init__()

        # Feature extractors
        self.wpt_extractor = WPTFeatureExtractor()
        self.ica_extractor = ICAFeatureExtractor()

        # Feature combination pool
        self.feature_pool = RealFeatureCombinationPool()

        # Combined feature dimension
        combined_size = wpt_feature_size + ica_feature_size

        # L1 feature selectors
        self.l1_selector1 = L1FeatureSelector(combined_size, sparsity_ratio=0.5)
        self.l1_selector2 = L1FeatureSelector(256, sparsity_ratio=0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(combined_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

        # Dropout layers
        self.dropout = nn.Dropout(0.3)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(64)

    def extract_features(self, eeg_signal: torch.Tensor) -> torch.Tensor:

        # Extract WPT features
        wpt_features = self.wpt_extractor.extract_wpt_features(eeg_signal)

        # Extract ICA features
        ica_features = self.ica_extractor.extract_ica_features(eeg_signal)

        # Combine features
        combined_features = torch.cat([wpt_features, ica_features], dim=1)

        return combined_features

    def forward(self, eeg_signal: torch.Tensor, training: bool = True) -> torch.Tensor:

        # Extract features
        combined_features = self.extract_features(eeg_signal)

        # Add to feature pool during training
        if training:
            self.feature_pool.add_features(combined_features)

        # First FC layer + L1 feature selection
        x1 = self.fc1(combined_features)
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1, 0.2)
        x1 = self.dropout(x1)

        selected_x1, mask1 = self.l1_selector1(x1)

        # Second FC layer + L1 feature selection
        x2 = self.fc2(selected_x1)
        x2 = self.bn2(x2)
        x2 = F.leaky_relu(x2, 0.2)
        x2 = self.dropout(x2)

        selected_x2, mask2 = self.l1_selector2(x2)

        # Third FC layer
        x3 = self.fc3(selected_x2)

        # Add retained features from first two layers to third layer output
        # Need to adjust dimensions to match
        if selected_x1.shape[1] == 2:
            x3 = x3 + selected_x1
        elif selected_x2.shape[1] == 2:
            x3 = x3 + selected_x2

        # Softmax output
        output = F.softmax(x3, dim=1)

        return output


class Generator(nn.Module):
    """Generator Module"""

    def __init__(self, noise_size: int = 64, feature_size: int = 128, output_size: int = 64):

        super(Generator, self).__init__()

        self.noise_size = noise_size
        self.feature_size = feature_size

        # L1 feature selectors
        self.l1_selector1 = L1FeatureSelector(noise_size + feature_size, sparsity_ratio=0.5)
        self.l1_selector2 = L1FeatureSelector(128, sparsity_ratio=0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(noise_size + feature_size, 128)  # Encoder layer 1
        self.fc2 = nn.Linear(128, 64)  # Encoder layer 2
        self.fc3 = nn.Linear(64, 128)  # Decoder
        self.fc4 = nn.Linear(128, output_size)  # Output layer

        # Dropout layers
        self.dropout = nn.Dropout(0.3)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, noise: torch.Tensor, real_features: torch.Tensor) -> torch.Tensor:

        # Concatenate noise and real features
        x = torch.cat([noise, real_features], dim=1)

        # First FC layer + L1 feature selection (Encoder layer 1)
        x1 = self.fc1(x)
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1, 0.2)
        x1 = self.dropout(x1)

        selected_x1, mask1 = self.l1_selector1(x1)

        # Second FC layer + L1 feature selection (Encoder layer 2)
        x2 = self.fc2(selected_x1)
        x2 = self.bn2(x2)
        x2 = F.leaky_relu(x2, 0.2)
        x2 = self.dropout(x2)

        selected_x2, mask2 = self.l1_selector2(x2)

        # Third FC layer (Decoder)
        x3 = self.fc3(selected_x2)
        x3 = self.bn3(x3)
        x3 = F.leaky_relu(x3, 0.2)

        # Add retained features from first two layers to third layer output
        if selected_x1.shape[1] == 128:
            x3 = x3 + selected_x1
        elif selected_x2.shape[1] == 128:
            x3 = x3 + selected_x2

        # Fourth FC layer (Output layer)
        generated_eeg = self.fc4(x3)

        return generated_eeg


class IGANEEG(nn.Module):
    """IGANEEG Model"""

    def __init__(self, noise_size: int = 64, feature_size: int = 128, output_size: int = 64):

        super(IGANEEG, self).__init__()

        self.discriminator = Discriminator(output_size)
        self.generator = Generator(noise_size, feature_size, output_size)

        # Loss function
        self.adversarial_loss = nn.BCELoss()

    def generate_noise(self, batch_size: int) -> torch.Tensor:

        return torch.randn(batch_size, self.generator.noise_size)

    def forward(self, batch_size: int, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:

        # Generate noise
        noise = self.generate_noise(batch_size)

        # Sample real features from discriminator's feature pool
        if self.discriminator.feature_pool.get_pool_size() > 0:
            real_features = self.discriminator.feature_pool.sample_random_features(batch_size)
        else:
            # If feature pool is empty, use zero features
            real_features = torch.zeros(batch_size, self.generator.feature_size)

        # Generate EEG signals
        generated_eeg = self.generator(noise, real_features)

        # Discriminator evaluation
        if training:
            # Use real EEG signals during training
            # Need to pass real EEG signal data here
            pass
        else:
            # Use generated EEG signals during testing
            # Need to convert generated_eeg to appropriate format
            # Assume generated_eeg is already in correct format here
            pass

        return generated_eeg


# Test code
if __name__ == "__main__":
    # Test model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = IGANEEG().to(device)

    # Test data
    batch_size = 8
    time_points = 128
    channels = 16

    # Create test EEG signal
    test_eeg = torch.randn(batch_size, time_points, channels).to(device)

    # Test discriminator
    print("Testing discriminator...")
    disc_output = model.discriminator(test_eeg, training=True)
    print(f"Discriminator output shape: {disc_output.shape}")

    # Test generator
    print("\nTesting generator...")
    noise = torch.randn(batch_size, 64).to(device)
    real_features = torch.randn(batch_size, 128).to(device)
    generated_eeg = model.generator(noise, real_features)
    print(f"Generator output shape: {generated_eeg.shape}")

    print("\nModel testing completed!")
