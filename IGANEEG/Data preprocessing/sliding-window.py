
import numpy as np
import torch
import os
from typing import Tuple, Dict, List
import pickle


class SlidingWindowProcessor:
    """Sliding window processor"""

    def __init__(self):
        # Window configurations
        self.window_configs = {
            'win64_overlap0': {'window_size': 64, 'overlap': 0},
            'win64_overlap32': {'window_size': 64, 'overlap': 32},
            'win128_overlap0': {'window_size': 128, 'overlap': 0},
            'win128_overlap32': {'window_size': 128, 'overlap': 32}
        }

        # Label configurations
        self.label_configs = {
            'NNCI': {
                'SZ_count': 45,  # First 45 samples are SZ (label 1)
                'HC_count': 39,  # Last 39 samples are HC (label 0)
                'total_samples': 84
            },
            'Ours': {
                'SZ_count': 100,  # First 100 samples are SZ (label 1)
                'HC_count': 100,  # Last 100 samples are HC (label 0)
                'total_samples': 200
            }
        }

    def sliding_window(self, signal: np.ndarray, window_size: int, overlap: int) -> np.ndarray:
        """Apply sliding window segmentation to signal"""
        if len(signal.shape) == 1:
            signal = signal.reshape(-1, 1)

        n_samples, n_channels = signal.shape

        # Calculate step size
        step_size = window_size - overlap

        # Calculate number of windows
        n_windows = max(0, (n_samples - window_size) // step_size + 1)

        if n_windows == 0:
            raise ValueError(f"Signal length({n_samples}) is less than window size({window_size})")

        # Create windows
        windows = []
        for i in range(n_windows):
            start = i * step_size
            end = start + window_size
            window = signal[start:end, :]
            windows.append(window)

        return np.array(windows)

    def create_labels_for_windows(self, original_labels: np.ndarray, n_windows_per_sample: int) -> np.ndarray:
        """Create labels for all windows"""
        window_labels = []
        for label in original_labels:
            # Assign same label to all windows of each sample
            sample_window_labels = np.full(n_windows_per_sample, label)
            window_labels.extend(sample_window_labels)

        return np.array(window_labels)

    def process_dataset(self, data: np.ndarray, dataset_type: str, config_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Process dataset using sliding window"""
        if dataset_type not in self.label_configs:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        if config_name not in self.window_configs:
            raise ValueError(f"Unsupported window configuration: {config_name}")

        # Get window configuration
        window_config = self.window_configs[config_name]
        window_size = window_config['window_size']
        overlap = window_config['overlap']

        # Get label configuration
        label_config = self.label_configs[dataset_type]

        # Validate data sample count
        n_samples = data.shape[0]
        expected_samples = label_config['total_samples']

        if n_samples != expected_samples:
            print(
                f"Warning: {dataset_type} dataset sample count({n_samples}) does not match expected({expected_samples})")

        # Create original sample labels
        original_labels = []

        # SZ samples (label 1)
        sz_count = min(label_config['SZ_count'], n_samples)
        original_labels.extend([1] * sz_count)

        # HC samples (label 0)
        hc_count = min(label_config['HC_count'], n_samples - sz_count)
        original_labels.extend([0] * hc_count)

        # If there are remaining samples, mark them as HC by default
        if len(original_labels) < n_samples:
            remaining = n_samples - len(original_labels)
            original_labels.extend([0] * remaining)

        original_labels = np.array(original_labels)

        # Apply sliding window segmentation to each sample
        all_windows = []
        n_windows_per_sample = None

        for i in range(n_samples):
            sample = data[i]  # Shape: [time_points, electrode_points]

            try:
                windows = self.sliding_window(sample, window_size, overlap)
                all_windows.append(windows)

                # Record number of windows per sample
                if n_windows_per_sample is None:
                    n_windows_per_sample = windows.shape[0]
                elif windows.shape[0] != n_windows_per_sample:
                    print(
                        f"Warning: Sample {i} window count({windows.shape[0]}) differs from other samples({n_windows_per_sample})")

            except ValueError as e:
                print(f"Sample {i} segmentation failed: {e}")
                continue

        if not all_windows:
            raise ValueError("No windows successfully segmented")

        # Combine all windows
        X_windows = np.concatenate(all_windows, axis=0)

        # Create window labels
        y_windows = self.create_labels_for_windows(original_labels, n_windows_per_sample)

        # Validate data consistency
        if X_windows.shape[0] != y_windows.shape[0]:
            raise ValueError(f"Window count({X_windows.shape[0]}) does not match label count({y_windows.shape[0]})")

        return X_windows, y_windows

    def process_all_datasets(self, nnci_data: np.ndarray, ours_data: np.ndarray) -> Dict[str, Dict]:
        """Process all datasets with all window configurations"""
        results = {}

        # Process NNCI data
        print("Processing NNCI data...")
        nnci_results = {}

        for config_name in self.window_configs.keys():
            print(f"  - Using configuration {config_name}")
            try:
                X_nnci, y_nnci = self.process_dataset(nnci_data, 'NNCI', config_name)
                nnci_results[config_name] = {
                    'X': X_nnci,
                    'y': y_nnci,
                    'config': self.window_configs[config_name]
                }
                print(f"    → Generated {X_nnci.shape[0]} windows")
            except Exception as e:
                print(f"    → Processing failed: {e}")

        results['NNCI'] = nnci_results

        # Process Ours data
        print("\nProcessing Ours data...")
        ours_results = {}

        for config_name in self.window_configs.keys():
            print(f"  - Using configuration {config_name}")
            try:
                X_ours, y_ours = self.process_dataset(ours_data, 'Ours', config_name)
                ours_results[config_name] = {
                    'X': X_ours,
                    'y': y_ours,
                    'config': self.window_configs[config_name]
                }
                print(f"    → Generated {X_ours.shape[0]} windows")
            except Exception as e:
                print(f"    → Processing failed: {e}")

        results['Ours'] = ours_results

        return results

    def save_results(self, results: Dict, output_dir: str = '.'):
        """Save processing results"""
        os.makedirs(output_dir, exist_ok=True)

        for dataset_name, dataset_results in results.items():
            for config_name, config_results in dataset_results.items():
                # Save data
                filename = f"{dataset_name}_{config_name}.pkl"
                filepath = os.path.join(output_dir, filename)

                with open(filepath, 'wb') as f:
                    pickle.dump(config_results, f)

                print(f"Saved: {filename}")

                # Also save in numpy format (optional)
                np_filename = f"{dataset_name}_{config_name}_X.npy"
                np_filepath = os.path.join(output_dir, np_filename)
                np.save(np_filepath, config_results['X'])

                np_y_filename = f"{dataset_name}_{config_name}_y.npy"
                np_y_filepath = os.path.join(output_dir, np_y_filename)
                np.save(np_y_filepath, config_results['y'])

        # Save configuration information
        config_info = {
            'window_configs': self.window_configs,
            'label_configs': self.label_configs
        }

        config_filepath = os.path.join(output_dir, 'processing_config.pkl')
        with open(config_filepath, 'wb') as f:
            pickle.dump(config_info, f)

        print(f"Configuration saved: processing_config.pkl")

    def print_statistics(self, results: Dict):
        """Print processing statistics"""
        print("\n=== Processing Results Statistics ===")

        for dataset_name, dataset_results in results.items():
            print(f"\n{dataset_name} Dataset:")

            for config_name, config_results in dataset_results.items():
                X = config_results['X']
                y = config_results['y']

                sz_count = np.sum(y == 1)
                hc_count = np.sum(y == 0)
                total_windows = len(y)

                print(f"  {config_name}:")
                print(f"    - Total windows: {total_windows}")
                print(f"    - SZ windows: {sz_count} ({sz_count / total_windows * 100:.1f}%)")
                print(f"    - HC windows: {hc_count} ({hc_count / total_windows * 100:.1f}%)")
                print(f"    - Window shape: {X.shape}")
                print(f"    - Label distribution: SZ={sz_count}, HC={hc_count}")


def main():
    """Main function"""

    # Load denoised data
    try:
        print("Loading denoised data...")

        # Check if files exist
        nnci_file = 'nnci_denoised.npy'
        ours_file = 'ours_denoised.npy'

        if not os.path.exists(nnci_file):
            print(f"Warning: File {nnci_file} does not exist")
            print("Please run Denoising.py first to generate denoised data")
            return

        if not os.path.exists(ours_file):
            print(f"Warning: File {ours_file} does not exist")
            print("Please run Denoising.py first to generate denoised data")
            return

        nnci_denoised = np.load(nnci_file)
        ours_denoised = np.load(ours_file)

        print(f"NNCI data shape: {nnci_denoised.shape}")
        print(f"Ours data shape: {ours_denoised.shape}")

    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Create processor
    processor = SlidingWindowProcessor()

    # Process all data
    results = processor.process_all_datasets(nnci_denoised, ours_denoised)

    # Print statistics
    processor.print_statistics(results)

    # Save results
    processor.save_results(results, 'sliding_window_results')

    print("\nProcessing completed! Results saved in 'sliding_window_results' directory")


if __name__ == "__main__":
    main()
