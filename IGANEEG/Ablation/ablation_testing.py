"""
IGANEEG ablation experiment test script
It is used to test and evaluate the four trained model variants.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Model'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data Preprocessing'))

from ablation_models import AblationModelFactory, ABLATION_CONFIGS, ABLATION_NAMES
from utils import set_seed, load_sliding_window_data

class AblationTester:
    def __init__(self, ablation_type, data_config, device=None):
        self.ablation_type = ablation_type
        self.data_config = data_config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

        set_seed(42)
        

        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, 'Data Preprocessing')
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                       f'results_{ABLATION_CONFIGS[ablation_type]}_{data_config}')
        self.results_dir = self.model_dir

        print(f"Initializing the ablation experiment tester:")
        print(f"  Model type: {ABLATION_NAMES[ABLATION_CONFIGS[ablation_type]]}")
        print(f"  Data configuration: {data_config}")
        print(f"  Device: {self.device}")
        
    def load_test_data(self):

        print(f"Load the test data: {self.data_config}")
        

        data_path = os.path.join(self.data_dir, f'sliding_window_{self.data_config}.pkl')
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The data file does not exist: {data_path}")
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        X = data['X']  # [N, window_size, channels]
        y = data['y']  # [N]
        
        print(f"Data shape: X={X.shape}, y={y.shape}")
        

        total_samples = len(X)
        train_size = int(total_samples * 0.7)
        

        X_test = X[train_size:]
        y_test = y[train_size:]

        print(f"Size of the test set: {len(X_test)}")
        print(f"Class distribution in the test set: SZ={np.sum(y_test == 1)}, HC={np.sum(y_test == 0)}")
        
        # 转换为torch tensor
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)
        
        return X_test_tensor, y_test_tensor, X_test, y_test
    
    def load_trained_model(self):

        print(f"Load the trained model: {ABLATION_NAMES[ABLATION_CONFIGS[self.ablation_type]]}")

        _, _, X_train, _ = self.load_test_data()
        input_shape = X_train.shape[1:]  # [window_size, channels]
        

        input_size = np.prod(input_shape)
        output_size = input_size
        

        discriminator, generator = AblationModelFactory.create_model(
            ablation_type=ABLATION_CONFIGS[self.ablation_type],
            input_size=input_size,
            noise_dim=64,
            output_size=output_size
        )
        
        discriminator = discriminator.to(self.device)
        generator = generator.to(self.device)
        

        model_path = os.path.join(self.model_dir, 'best_model.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The model file does not exist: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        generator.load_state_dict(checkpoint['generator_state_dict'])
        

        discriminator.eval()
        generator.eval()

        print(f"Model loaded successfully. Number of training epochs: {checkpoint.get('epoch', 'Unknown')}")
        print(f"Best loss: {checkpoint.get('total_loss', 'Unknown')}")
        
        return discriminator, generator
    
    def generate_samples(self, generator, n_samples):

        generator.eval()
        generated_samples = []
        
        with torch.no_grad():
            for i in range(n_samples):

                noise = torch.randn(1, 64).to(self.device)
                

                if self.ablation_type in [2, 3, 4]:
                    generated, _ = generator(noise)
                else:
                    generated = generator(noise)
                
                generated_samples.append(generated.cpu().numpy())
        
        return np.concatenate(generated_samples, axis=0)
    
    def evaluate_classification(self, discriminator, X_test, y_test):

        discriminator.eval()
        predictions = []
        
        with torch.no_grad():
            for x in X_test:
                x_tensor = x.unsqueeze(0)
                

                if self.ablation_type in [2, 3, 4]:
                    disc_output, disc_prob, _ = discriminator(x_tensor)
                else:
                    disc_output, disc_prob = discriminator(x_tensor)
                

                pred_class = torch.argmax(disc_output, dim=1).item()
                predictions.append(pred_class)
        
        predictions = np.array(predictions)
        y_test_np = y_test.cpu().numpy()
        

        accuracy = accuracy_score(y_test_np, predictions)
        mae = mean_absolute_error(y_test_np, predictions)
        mse = mean_squared_error(y_test_np, predictions)
        
        return accuracy, mae, mse, predictions
    
    def calculate_mmd(self, real_samples, generated_samples):

        real_flat = real_samples.reshape(real_samples.shape[0], -1)
        gen_flat = generated_samples.reshape(generated_samples.shape[0], -1)
        

        real_tensor = torch.FloatTensor(real_flat).to(self.device)
        gen_tensor = torch.FloatTensor(gen_flat).to(self.device)
        
        # Calculate the Maximum Mean Discrepancy (MMD).
        def rbf_kernel(X, Y, gamma=1.0):

            sq_dist = torch.cdist(X, Y, p=2)**2
            return torch.exp(-gamma * sq_dist)
        

        gamma = 1.0 / real_flat.shape[1]
        K_rr = rbf_kernel(real_tensor, real_tensor, gamma)
        K_rg = rbf_kernel(real_tensor, gen_tensor, gamma)
        K_gg = rbf_kernel(gen_tensor, gen_tensor, gamma)
        
        mmd = (K_rr.mean() - 2*K_rg.mean() + K_gg.mean()).item()
        
        return mmd
    
    def visualize_samples(self, real_samples, generated_samples, n_samples=5):

        plt.figure(figsize=(15, 8))
        

        real_vis = real_samples[:n_samples]
        gen_vis = generated_samples[:n_samples]
        
        for i in range(n_samples):

            plt.subplot(2, n_samples, i+1)
            plt.plot(real_vis[i].T, alpha=0.7)
            plt.title(f'Real Sample {i+1}')
            plt.xlabel('Time Points')
            plt.ylabel('Amplitude')
            

            plt.subplot(2, n_samples, n_samples+i+1)
            plt.plot(gen_vis[i].T, alpha=0.7)
            plt.title(f'Generated Sample {i+1}')
            plt.xlabel('Time Points')
            plt.ylabel('Amplitude')
        
        plt.tight_layout()
        
        # 保存图像
        sample_plot_path = os.path.join(self.results_dir, 'sample_comparison.png')
        plt.savefig(sample_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"The sample comparison chart has been saved to: {sample_plot_path}")
    
    def run_test(self):

        print("=" * 60)
        print(f"Ablation experiment test - {ABLATION_NAMES[ABLATION_CONFIGS[self.ablation_type]]}")
        print(f"Data configuration: {self.data_config}")
        print("=" * 60)
        
        try:

            X_test_tensor, y_test_tensor, X_test_np, y_test_np = self.load_test_data()
            

            discriminator, generator = self.load_trained_model()
            

            print("\nGenerate samples....")
            n_generate = len(X_test_tensor)
            generated_samples = self.generate_samples(generator, n_generate)
            

            print("\nCalculate the MMD....")
            mmd_value = self.calculate_mmd(X_test_np, generated_samples)
            

            print("\nEvaluate the classification performance....")
            accuracy, mae, mse, predictions = self.evaluate_classification(
                discriminator, X_test_tensor, y_test_tensor
            )
            

            print("\n" + "="*60)
            print("Test results:")
            print("=" * 60)
            print(f"Maximum Mean Discrepancy (MMD): {mmd_value:.6f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Number of test samples: {len(X_test_np)}")
            print(f"Number of generated samples: {len(generated_samples)}")
            

            results = {
                'ablation_type': self.ablation_type,
                'ablation_name': ABLATION_NAMES[ABLATION_CONFIGS[self.ablation_type]],
                'data_config': self.data_config,
                'mmd': mmd_value,
                'accuracy': accuracy,
                'mae': mae,
                'mse': mse,
                'n_test_samples': len(X_test_np),
                'n_generated_samples': len(generated_samples),
                'predictions': predictions.tolist(),
                'true_labels': y_test_np.tolist(),
                'timestamp': datetime.now().isoformat()
            }
            
            results_path = os.path.join(self.results_dir, 'test_results.pkl')
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)
            
            print(f"\nThe test results have been saved to: {results_path}")
            

            print("\nVisualize the sample comparison....")
            self.visualize_samples(X_test_np, generated_samples)
            

            generated_path = os.path.join(self.results_dir, 'generated_samples.npy')
            np.save(generated_path, generated_samples)
            print(f"The generated samples have been saved to: {generated_path}")
            
            print("\n" + "="*60)
            print("The test is completed！")
            print("="*60)
            
            return results
            
        except Exception as e:
            print(f"An error occurred during the test.: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def run_single_ablation_test(ablation_type, data_config):

    print(f"\n{'='*80}")
    print(f"Start the ablation experiment test.: {ABLATION_NAMES[ABLATION_CONFIGS[ablation_type]]} - {data_config}")
    print(f"{'='*80}")
    
    tester = AblationTester(ablation_type, data_config)
    results = tester.run_test()
    
    if results is not None:
        print(f"✅ The ablation experiment test is completed: {ABLATION_NAMES[ABLATION_CONFIGS[ablation_type]]} - {data_config}")
        return results
    else:
        print(f"❌ The ablation experiment test has failed: {ABLATION_NAMES[ABLATION_CONFIGS[ablation_type]]} - {data_config}")
        return None

if __name__ == "__main__":

    ablation_type = 1
    data_config = "win64_overlap0"
    
    results = run_single_ablation_test(ablation_type, data_config)

    if results:
        print("The ablation experiment test has been successfully completed!")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"MAE: {results['mae']:.4f}")
        print(f"MSE: {results['mse']:.4f}")
    else:
        print("The ablation experiment test has failed!")