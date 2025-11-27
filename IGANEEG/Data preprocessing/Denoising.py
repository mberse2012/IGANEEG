
import numpy as np
import torch
import scipy.signal as signal
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from eeg_data_loader_NNCI import load_NNCI_eeg_data
from eeg_data_loader_Ours import load_our_eeg_data


class BayesianEEGDenoiser:
    """EEG signal denoiser based on Bayesian methods"""

    def __init__(self, sampling_rate=128):

        self.sampling_rate = sampling_rate

    def wavelet_transform(self, signal_data, wavelet='db4', level=5):

        try:
            import pywt
            coeffs = pywt.wavedec(signal_data, wavelet, level=level)
            return coeffs
        except ImportError:
            print("Warning: pywt library not installed, using alternative method")
            # Use simple Fourier transform as alternative
            fft_coeffs = np.fft.fft(signal_data)
            return [fft_coeffs]

    def inverse_wavelet_transform(self, coeffs, wavelet='db4'):

        try:
            import pywt
            reconstructed = pywt.waverec(coeffs, wavelet)
            return reconstructed
        except ImportError:
            # Use inverse Fourier transform as alternative
            reconstructed = np.fft.ifft(coeffs[0]).real
            return reconstructed

    def bayesian_threshold(self, coeffs, noise_std=None):

        if noise_std is None:
            # Estimate noise standard deviation using median absolute deviation
            if len(coeffs) > 1:
                detail_coeffs = coeffs[1]
                noise_std = np.median(np.abs(detail_coeffs)) / 0.6745
            else:
                noise_std = np.std(coeffs[0]) * 0.8

        thresholded_coeffs = []
        for i, level_coeffs in enumerate(coeffs):
            if i == 0:  # Approximation coefficients, usually preserved
                thresholded_coeffs.append(level_coeffs)
            else:  # Detail coefficients, apply Bayesian threshold
                # Bayesian threshold formula
                sigma_signal = np.sqrt(np.maximum(0, np.var(level_coeffs) - noise_std ** 2))
                threshold = noise_std ** 2 / sigma_signal if sigma_signal > 0 else 3 * noise_std

                # Soft thresholding
                thresholded = np.sign(level_coeffs) * np.maximum(np.abs(level_coeffs) - threshold, 0)
                thresholded_coeffs.append(thresholded)

        return thresholded_coeffs

    def kalman_filter(self, signal_data, process_noise=0.1, measurement_noise=1.0):

        n = len(signal_data)

        # Initialization
        x = signal_data[0]  # Initial state
        P = 1.0  # Initial covariance

        filtered_signal = np.zeros(n)
        filtered_signal[0] = x

        for i in range(1, n):
            # Prediction step
            x_pred = x  # State prediction
            P_pred = P + process_noise  # Covariance prediction

            # Update step
            K = P_pred / (P_pred + measurement_noise)  # Kalman gain
            x = x_pred + K * (signal_data[i] - x_pred)  # State update
            P = (1 - K) * P_pred  # Covariance update

            filtered_signal[i] = x

        return filtered_signal

    def empirical_bayes_denoise(self, signal_data):

        # Use wavelet transform + Bayesian threshold
        coeffs = self.wavelet_transform(signal_data)

        # Estimate noise standard deviation
        if len(coeffs) > 1:
            noise_std = np.median(np.abs(coeffs[1])) / 0.6745
        else:
            noise_std = np.std(coeffs[0]) * 0.8

        # Bayesian threshold processing
        thresholded_coeffs = self.bayesian_threshold(coeffs, noise_std)

        # Wavelet reconstruction
        denoised_signal = self.inverse_wavelet_transform(thresholded_coeffs)

        # Ensure consistent length
        if len(denoised_signal) != len(signal_data):
            denoised_signal = signal.resample(denoised_signal, len(signal_data))

        return denoised_signal

    def denoise_eeg_signal(self, eeg_signal, method='empirical_bayes'):

        if len(eeg_signal.shape) == 1:
            # 1D signal
            if method == 'empirical_bayes':
                return self.empirical_bayes_denoise(eeg_signal)
            elif method == 'kalman':
                return self.kalman_filter(eeg_signal)
            else:
                raise ValueError("Unsupported denoising method")
        else:
            # 2D signal (multi-channel)
            denoised_signals = []
            for channel in range(eeg_signal.shape[1]):
                channel_signal = eeg_signal[:, channel]
                if method == 'empirical_bayes':
                    denoised_channel = self.empirical_bayes_denoise(channel_signal)
                elif method == 'kalman':
                    denoised_channel = self.kalman_filter(channel_signal)
                else:
                    raise ValueError("Unsupported denoising method")
                denoised_signals.append(denoised_channel)

            return np.column_stack(denoised_signals)

    def denoise_eeg_dataset(self, dataset, method='empirical_bayes', progress=True):

        denoised_data = []

        for i, sample in enumerate(dataset):
            if isinstance(sample, torch.Tensor):
                sample_np = sample.numpy()
            else:
                sample_np = sample

            denoised_sample = self.denoise_eeg_signal(sample_np, method)
            denoised_data.append(denoised_sample)

            if progress and (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(dataset)} samples")

        return np.array(denoised_data)

    def calculate_snr_improvement(self, original_signal, denoised_signal):

        # Calculate noise (difference between original and denoised signal)
        noise = original_signal - denoised_signal

        # Calculate signal power and noise power
        signal_power = np.mean(denoised_signal ** 2)
        noise_power = np.mean(noise ** 2)

        # Avoid division by zero
        if noise_power == 0:
            return float('inf')

        # Calculate SNR (dB)
        snr_after = 10 * np.log10(signal_power / noise_power)

        # Original signal SNR (assuming noise power is part of total power)
        original_noise_power = np.var(original_signal) * 0.3  # Assume 30% is noise
        snr_before = 10 * np.log10(np.mean(original_signal ** 2) / original_noise_power)

        return snr_after - snr_before


def denoise_and_evaluate():
    """
    Denoise and evaluate two datasets
    """
    # Data paths
    nnci_data_dir = r"D:\data\NNCI"
    ours_data_dir = r"D:\data\Ours"

    denoiser = BayesianEEGDenoiser(sampling_rate=128)

    try:
        # Load NNCI data
        print("Loading NNCI data...")
        nnci_dataset, nnci_tensor, nnci_stats, _ = load_NNCI_eeg_data(nnci_data_dir)
        print(f"NNCI data loaded, number of samples: {len(nnci_dataset)}")

        # Load Ours data
        print("\nLoading Ours data...")
        ours_dataset, ours_tensor, ours_stats, _ = load_our_eeg_data(ours_data_dir)
        print(f"Ours data loaded, number of samples: {len(ours_dataset)}")

        # Denoise NNCI data
        print("\nApplying Bayesian denoising to NNCI data...")
        nnci_denoised = denoiser.denoise_eeg_dataset(nnci_dataset, method='empirical_bayes')

        # Denoise Ours data
        print("\nApplying Bayesian denoising to Ours data...")
        ours_denoised = denoiser.denoise_eeg_dataset(ours_dataset, method='empirical_bayes')

        # Calculate SNR improvement
        print("\nCalculating SNR improvement...")
        nnci_snr_improvement = []
        for i in range(min(10, len(nnci_dataset))):  # Calculate first 10 samples
            original = nnci_dataset[i].numpy()
            denoised = nnci_denoised[i]
            improvement = denoiser.calculate_snr_improvement(original.flatten(), denoised.flatten())
            nnci_snr_improvement.append(improvement)

        ours_snr_improvement = []
        for i in range(min(10, len(ours_dataset))):
            original = ours_dataset[i].numpy()
            denoised = ours_denoised[i]
            improvement = denoiser.calculate_snr_improvement(original.flatten(), denoised.flatten())
            ours_snr_improvement.append(improvement)

        # Output results
        print("\n=== Denoising Results ===")
        print(f"NNCI Dataset:")
        print(f"  - Number of samples: {len(nnci_dataset)}")
        print(f"  - Average SNR improvement: {np.mean(nnci_snr_improvement):.2f} dB")
        print(f"  - Denoised data shape: {nnci_denoised.shape}")

        print(f"\nOurs Dataset:")
        print(f"  - Number of samples: {len(ours_dataset)}")
        print(f"  - Average SNR improvement: {np.mean(ours_snr_improvement):.2f} dB")
        print(f"  - Denoised data shape: {ours_denoised.shape}")

        # Save denoised data
        np.save('nnci_denoised.npy', nnci_denoised)
        np.save('ours_denoised.npy', ours_denoised)
        print("\nDenoised data saved as: nnci_denoised.npy and ours_denoised.npy")

        return nnci_denoised, ours_denoised

    except Exception as e:
        print(f"Error during processing: {e}")
        return None, None


if __name__ == "__main__":
    # Run denoising and evaluation
    nnci_denoised, ours_denoised = denoise_and_evaluate()
