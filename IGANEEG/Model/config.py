
import torch


class Config:
    """IGANEEG model configuration class"""

    # Model parameters
    NOISE_SIZE = 64  # Noise vector size
    FEATURE_SIZE = 128  # Feature vector size
    OUTPUT_SIZE = 64  # Output EEG signal size

    # Discriminator parameters
    DISCRIMINATOR_FC_SIZES = [256, 64, 2]  # Discriminator fully connected layer sizes
    L1_SPARSITY_RATIO = 0.5  # L1 feature selection sparsity ratio

    # Generator parameters
    GENERATOR_FC_SIZES = [128, 64, 128, 64]  # Generator fully connected layer sizes

    # Training parameters
    BATCH_SIZE = 8  # Batch size
    LEARNING_RATE_D = 0.0001  # Discriminator learning rate
    LEARNING_RATE_G = 0.0001  # Generator learning rate
    BETA1 = 0.5  # Adam optimizer beta1
    BETA2 = 0.999  # Adam optimizer beta2
    EPOCHS = 300  # Training epochs

    # Learning rate scheduling parameters
    LR_STEP_SIZE = 50  # Learning rate decay step size
    LR_GAMMA = 0.95  # Learning rate decay factor

    # Regularization parameters
    DROPOUT_RATE = 0.3  # Dropout rate

    # Feature extraction parameters
    WPT_WAVELET = 'db4'  # Wavelet basis function
    WPT_LEVEL = 3  # Wavelet decomposition levels
    ICA_COMPONENTS = 16  # ICA components count

    # Feature pool parameters
    FEATURE_POOL_MAX_SIZE = 1000  # Feature pool maximum capacity

    # Data parameters
    TIME_POINTS = 128  # Time points count
    CHANNELS = 16  # Channels count

    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Path configuration
    DATA_PATH = "../Data Preprocessing/sliding_window_results"
    MODEL_SAVE_PATH = "./saved_models"
    LOG_PATH = "./logs"

    @classmethod
    def get_model_params(cls):
        """Get model parameters"""
        return {
            'noise_size': cls.NOISE_SIZE,
            'feature_size': cls.FEATURE_SIZE,
            'output_size': cls.OUTPUT_SIZE
        }

    @classmethod
    def get_training_params(cls):
        """Get training parameters"""
        return {
            'batch_size': cls.BATCH_SIZE,
            'learning_rate_d': cls.LEARNING_RATE_D,
            'learning_rate_g': cls.LEARNING_RATE_G,
            'beta1': cls.BETA1,
            'beta2': cls.BETA2,
            'epochs': cls.EPOCHS
        }

    @classmethod
    def get_feature_params(cls):
        """Get feature extraction parameters"""
        return {
            'wpt_wavelet': cls.WPT_WAVELET,
            'wpt_level': cls.WPT_LEVEL,
            'ica_components': cls.ICA_COMPONENTS
        }

