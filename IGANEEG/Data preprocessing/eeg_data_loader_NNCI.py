
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import glob


class EEGDataset(Dataset):
    """EEG signal dataset class"""

    def __init__(self, data_dir, sampling_rate=128, duration=120):

        self.data_dir = data_dir
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.expected_samples = sampling_rate * duration  # 15360 sample points
        self.num_electrodes = 16  # 16 electrode points

        # Find all CSV files
        self.csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

        if not self.csv_files:
            raise FileNotFoundError(f"No CSV files found in directory {data_dir}")

        print(f"Found {len(self.csv_files)} CSV files")

        # Preload all data
        self.data = []
        self._load_all_data()

    def _load_all_data(self):
        """Load all CSV file data"""
        for i, csv_file in enumerate(self.csv_files):
            try:
                # Read CSV file
                df = pd.read_csv(csv_file)

                # Check data dimensions
                if df.shape[1] != self.num_electrodes:
                    print(
                        f"Warning: File {os.path.basename(csv_file)} has {df.shape[1]} columns, expected {self.num_electrodes} columns")

                # Check sample count
                if df.shape[0] != self.expected_samples:
                    print(
                        f"Warning: File {os.path.basename(csv_file)} has {df.shape[0]} samples, expected {self.expected_samples} samples")

                # Convert to numpy array
                eeg_data = df.values.astype(np.float32)

                # Ensure correct shape (sample_count × electrodes)
                if eeg_data.shape[1] != self.num_electrodes:
                    eeg_data = eeg_data.T  # Transpose if dimensions are incorrect

                self.data.append(eeg_data)

                if (i + 1) % 10 == 0:
                    print(f"Loaded {i + 1}/{len(self.csv_files)} files")

            except Exception as e:
                print(f"Error loading file {csv_file}: {e}")

    def __len__(self):
        """Return dataset size"""
        return len(self.data)

    def __getitem__(self, idx):
        """Get single sample"""
        return torch.from_numpy(self.data[idx])

    def get_tensor(self):
        """Combine all data into one PyTorch tensor"""
        # Stack all samples into one tensor
        tensor_data = torch.stack([torch.from_numpy(sample) for sample in self.data])
        return tensor_data

    def get_statistics(self):
        """Get data statistics"""
        all_data = np.concatenate(self.data, axis=0)

        stats = {
            'num_samples': len(self.data),
            'total_data_points': all_data.shape[0] * all_data.shape[1],
            'shape_per_sample': f"({self.expected_samples}, {self.num_electrodes})",
            'mean': np.mean(all_data),
            'std': np.std(all_data),
            'min': np.min(all_data),
            'max': np.max(all_data)
        }

        return stats


def load_NNCI_eeg_data(data_dir):
    dataset = EEGDataset(data_dir)
    tensor_data = dataset.get_tensor()
    statistics = dataset.get_statistics()

    return dataset, tensor_data, statistics


if __name__ == "__main__":
    # Usage example
    data_directory = r"D:\data"  # Please modify to your actual path

    try:
        # Load data
        dataset, tensor_data, stats = load_NNCI_eeg_data(data_directory)

        # Print statistics
        print("\nData statistics:")
        print(f"Number of samples: {stats['num_samples']}")
        print(f"Shape per sample: {stats['shape_per_sample']}")
        print(f"Total data points: {stats['total_data_points']}")
        print(f"Data mean: {stats['mean']:.4f}")
        print(f"Data standard deviation: {stats['std']:.4f}")
        print(f"Data range: [{stats['min']:.4f}, {stats['max']:.4f}]")

        # Print tensor information
        print(f"\nPyTorch tensor shape: {tensor_data.shape}")
        print(f"Tensor data type: {tensor_data.dtype}")

        # Save tensor to file (optional)
        torch.save(tensor_data, 'eeg_data_tensor.pt')
        print("Tensor saved to 'eeg_data_tensor.pt'")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the data directory exists and contains CSV files")
        print("Please modify the data_directory variable to your actual path")
