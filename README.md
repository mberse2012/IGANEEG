
IGANEEG: An IoMT-Enabled Generative Framework for Small-Sample Schizophrenia EEG Augmentation

PROJECT OVERVIEW:IGANEEG is an electroencephalogram (EEG) data augmentation framework based on an improved generative adversarial network, specifically designed to address the small-sample data problem in schizophrenia diagnosis. This project generates high-quality and high-fidelity synthetic EEG signals through the reconstruction loss of the autoencoder and the dynamic pause mechanism, effectively overcoming the mode collapse and gradient vanishing problems of traditional methods. It provides a reliable tool for intelligent diagnosis in the context of the Internet of Medical Things. 


DIRECTORY STRUCTURE:
├── Data Preprocessing/
│   ├── Denoising.py              # EEG signal denoising using Bayesian methods
│   ├── EEG_data_loader_NNCI.py   # NNCI dataset loader
│   ├── EEG_data_loader_Ours.py   # Custom dataset loader
│   └── Sliding_window.py         # Sliding window processing
├── Model/
│   ├── IGANEEG.py                # Main GAN model architecture
│   ├── config.py                 # Model configuration parameters
│   └── utils.py                  # Utility functions
├── Train/
│   ├── batch_train.py            # Batch training script
│   ├── train_win64_overlap0.py   # Training for window 64, overlap 0
│   ├── train_win64_overlap32.py  # Training for window 64, overlap 32
│   ├── train_win128_overlap0.py  # Training for window 128, overlap 0
│   └── train_win128_overlap32.py # Training for window 128, overlap 32
├── Test/
│   ├── batch_test.py             # Batch testing script
│   ├── test_win64_overlap0.py    # Testing for window 64, overlap 0
│   ├── test_win64_overlap32.py   # Testing for window 64, overlap 32
│   ├── test_win128_overlap0.py   # Testing for window 128, overlap 0
│   └── test_win128_overlap32.py  # Testing for window 128, overlap 32
├── Ablation/
│   ├── ablation_models.py      # Definition of models for ablation experiments
│   ├── ablation_training.py    # Training script for ablation experiments
│   ├── ablation_testing.py     # Testing script for ablation experiments
│   └── run_ablation_experiments.py  # Run complete ablation experiments

QUICK START:

1. DATA PREPROCESSING:
   - Run Denoising.py to clean EEG signals
   - Use Sliding_window.py to create windowed datasets
   - Configure window sizes (64/128) and overlaps (0/32)

2. TRAINING:
   - Use individual train scripts for specific configurations
   - Or run batch_train.py for all configurations
   - Models are saved in respective results directories

3. TESTING:
   - Use individual test scripts for specific configurations
   - Or run batch_test.py for comprehensive testing
   - Results include MMD, accuracy, MAE, and MSE metrics

4. Ablation Experiments
   - Run  Ablation/run_ablation_experiments.py to conduct a complete ablation experiment.
   - Alternatively, run ablation_training.py` and  ablation_testing.py separately. 



KEY FEATURES:
- L1 Feature Selection for sparse feature representation
- Wavelet Packet Transform for time-frequency analysis
- Independent Component Analysis for signal separation
- Real Feature Combination Pool for training stability
- Multiple window configurations for robust performance

DEPENDENCIES:
- Python 3.7+
- PyTorch 1.8+
- NumPy, SciPy, scikit-learn
- Matplotlib for visualization


CONFIGURATIONS SUPPORTED:
- Window 64, Overlap 0
- Window 64, Overlap 32  
- Window 128, Overlap 0
- Window 128, Overlap 32

RESULTS:
Each configuration generates:
- Training loss curves
- Sample comparisons (real vs generated)
- Performance metrics (MMD, accuracy, MAE, MSE)
- Model checkpoints

