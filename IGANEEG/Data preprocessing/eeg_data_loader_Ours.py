
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import glob


class OurEEGDataset(Dataset):
    """Our collected EEG signal dataset class"""

    def __init__(self, data_dir, sampling_rate=128, duration=120):

        self.data_dir = data_dir
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.expected_samples = sampling_rate * duration  # 15360 sample points

        # 16 electrode points to be retained from the original 64 electrode points
        self.selected_electrodes = ['O1', 'O2', 'P3', 'P4', 'Pz', 'T5', 'T6', 'C3',
                                    'C4', 'Cz', 'T3', 'T4', 'F3', 'F4', 'F7', 'F8']
        self.num_selected_electrodes = len(self.selected_electrodes)

        # Find all CSV files (limit 200 samples)
        all_csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        self.csv_files = all_csv_files[:200]  # Only take the first 200 files

        if not self.csv_files:
            raise FileNotFoundError(f"No CSV files found in directory {data_dir}")

        print(f"Found {len(self.csv_files)} CSV files (limited to 200 samples)")

        # Preload all data
        self.data = []
        self._load_all_data()

    def _load_all_data(self):
        """Load all CSV file data and filter electrode points"""
        for i, csv_file in enumerate(self.csv_files):
            try:
                # Read CSV file
                df = pd.read_csv(csv_file)

                # Check data dimensions
                if df.shape[1] < 64:
                    print(
                        f"Warning: File {os.path.basename(csv_file)} only has {df.shape[1]} columns, expected at least 64 columns")

                # Check sample count
                if df.shape[0] != self.expected_samples:
                    print(
                        f"Warning: File {os.path.basename(csv_file)} has {df.shape[0]} samples, expected {self.expected_samples} samples")

                # Get column names (electrode point names)
                column_names = df.columns.tolist()

                # Filter specified 16 electrode points
                selected_columns = []
                missing_electrodes = []

                for electrode in self.selected_electrodes:
                    # Try different column name matching methods
                    matching_columns = [col for col in column_names if electrode.lower() in col.lower()]
                    if matching_columns:
                        selected_columns.append(matching_columns[0])
                    else:
                        missing_electrodes.append(electrode)

                if missing_electrodes:
                    print(f"Warning: File {os.path.basename(csv_file)} missing electrodes: {missing_electrodes}")

                if len(selected_columns) != self.num_selected_electrodes:
                    print(
                        f"Warning: File {os.path.basename(csv_file)} only found {len(selected_columns)} target electrode points, expected {self.num_selected_electrodes}")

                # Filter data
                if selected_columns:
                    selected_data = df[selected_columns].values.astype(np.float32)

                    # Ensure correct shape (sample_count × electrodes)
                    if selected_data.shape[1] != len(selected_columns):
                        selected_data = selected_data.T

                    self.data.append(selected_data)
                else:
                    print(
                        f"Error: File {os.path.basename(csv_file)} did not find any target electrode points, skipping this file")

                if (i + 1) % 20 == 0:
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
        if not self.data:
            return {}

        all_data = np.concatenate(self.data, axis=0)

        stats = {
            'num_samples': len(self.data),
            'total_data_points': all_data.shape[0] * all_data.shape[1],
            'shape_per_sample': f"({self.expected_samples}, {self.num_selected_electrodes})",
            'mean': np.mean(all_data),
            'std': np.std(all_data),
            'min': np.min(all_data),
            'max': np.max(all_data),
            'selected_electrodes': self.selected_electrodes
        }

        return stats

    def get_electrode_statistics(self):
        """Get statistical information for each electrode point"""
        if not self.data:
            return {}

        all_data = np.concatenate(self.data, axis=0)
        electrode_stats = {}

        for i, electrode in enumerate(self.selected_electrodes):
            if i < all_data.shape[1]:
                electrode_data = all_data[:, i]
                electrode_stats[electrode] = {
                    'mean': np.mean(electrode_data),
                    'std': np.std(electrode_data),
                    'min': np.min(electrode_data),
                    'max': np.max(electrode_data)
                }

        return electrode_stats


def load_our_eeg_data(data_dir):
    dataset = OurEEGDataset(data_dir)
    tensor_data = dataset.get_tensor()
    statistics = dataset.get_statistics()
    electrode_statistics = dataset.get_electrode_statistics()

    return dataset, tensor_data, statistics, electrode_statistics


if __name__ == "__main__":
    # Usage example
    data_directory = r"D:\data"  # Please modify to your actual path

    try:
        # Load data
        dataset, tensor_data, stats, electrode_stats = load_our_eeg_data(data_directory)

        # Print statistics
        print("\nData statistics:")
        print(f"Number of samples: {stats['num_samples']}")
        print(f"Shape per sample: {stats['shape_per_sample']}")
        print(f"Total data points: {stats['total_data_points']}")
        print(f"Data mean: {stats['mean']:.4f}")
        print(f"Data standard deviation: {stats['std']:.4f}")
        print(f"Data range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"Selected electrodes: {stats['selected_electrodes']}")

        # Print electrode statistics
        print("\nElectrode statistics:")
        for electrode, e_stats in electrode_stats.items():
            print(f"{electrode}: mean={e_stats['mean']:.4f}, std={e_stats['std']:.4f}, "
                  f"range=[{e_stats['min']:.4f}, {e_stats['max']:.4f}]")

        # Print tensor information
        print(f"\nPyTorch tensor shape: {tensor_data.shape}")
        print(f"Tensor data type: {tensor_data.dtype}")

        # Save tensor to file (optional)
        torch.save(tensor_data, 'our_eeg_data_tensor.pt')
        print("Tensor saved to 'our_eeg_data_tensor.pt'")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the data directory exists and contains CSV files")
        print("Please modify the data_directory variable to your actual path")
