"""
IGANEEG ablation experiment training script
Train 4 model variants on 4 data configurations.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Model'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data Preprocessing'))

from ablation_models import AblationModelFactory, ABLATION_CONFIGS, ABLATION_NAMES
from utils import set_seed, load_sliding_window_data

class AblationTrainer:
    def __init__(self, ablation_type, data_config, device=None):
        self.ablation_type = ablation_type
        self.data_config = data_config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

        set_seed(42)
        

        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, 'Data Preprocessing')
        self.results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                       f'results_{ABLATION_CONFIGS[ablation_type]}_{data_config}')
        

        os.makedirs(self.results_dir, exist_ok=True)
        

        self.epochs = 100
        self.batch_size = 32
        self.learning_rate = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999

        print(f"Initializing the ablation experiment trainer:")
        print(f"  Model type: {ABLATION_NAMES[ABLATION_CONFIGS[ablation_type]]}")
        print(f"  Data configuration: {data_config}")
        print(f"  Device: {self.device}")
        
    def load_data(self):

        print(f"Load the data.: {self.data_config}")
        

        data_path = os.path.join(self.data_dir, f'sliding_window_{self.data_config}.pkl')
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The data file does not exist: {data_path}")
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        X = data['X']  # [N, window_size, channels]
        y = data['y']  # [N]
        
        print(f"data shape: X={X.shape}, y={y.shape}")
        

        total_samples = len(X)
        train_size = int(total_samples * 0.7)
        
        X_train = X[:train_size]
        y_train = y[:train_size]

        print(f"Size of the training set: {len(X_train)}")
        print(f"Class distribution in the training set: SZ={np.sum(y_train == 1)}, HC={np.sum(y_train == 0)}")
        

        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        
        return X_train_tensor, y_train_tensor, X_train.shape
    
    def create_models(self, input_shape):


        input_size = np.prod(input_shape)  # window_size * channels
        output_size = input_size

        print(f"Creating the model - Input size: {input_size}, Output size: {output_size}")
        

        discriminator, generator = AblationModelFactory.create_model(
            ablation_type=ABLATION_CONFIGS[self.ablation_type],
            input_size=input_size,
            noise_dim=64,
            output_size=output_size
        )
        
        discriminator = discriminator.to(self.device)
        generator = generator.to(self.device)
        

        total_params_d = sum(p.numel() for p in discriminator.parameters())
        total_params_g = sum(p.numel() for p in generator.parameters())


        print(f"Number of discriminator parameters: {total_params_d:,}")
        print(f"Number of generator parameters: {total_params_g:,}")

        return discriminator, generator
    
    def create_optimizers(self, discriminator, generator):

        d_optimizer = optim.Adam(
            discriminator.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2)
        )
        
        g_optimizer = optim.Adam(
            generator.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2)
        )
        
        return d_optimizer, g_optimizer
    
    def train_epoch(self, discriminator, generator, d_optimizer, g_optimizer, 
                    X_train, y_train, epoch):

        discriminator.train()
        generator.train()
        
        n_samples = len(X_train)
        n_batches = n_samples // self.batch_size
        
        total_d_loss = 0
        total_g_loss = 0
        total_l1_loss = 0
        
        for batch_idx in range(n_batches):

            start_idx = batch_idx * self.batch_size
            end_idx = start_idx + self.batch_size
            
            real_samples = X_train[start_idx:end_idx]
            real_labels = y_train[start_idx:end_idx]
            
            batch_size = real_samples.size(0)
            

            d_optimizer.zero_grad()
            

            if self.ablation_type in [2, 3, 4]:
                d_output_real, d_prob_real, l1_loss_real = discriminator(real_samples)
                d_loss_real = nn.CrossEntropyLoss()(d_output_real, real_labels) + l1_loss_real
            else:
                d_output_real, d_prob_real = discriminator(real_samples)
                d_loss_real = nn.CrossEntropyLoss()(d_output_real, real_labels)
            

            noise = torch.randn(batch_size, 64).to(self.device)
            if self.ablation_type in [2, 3, 4]:
                generated_samples, g_l1_loss = generator(noise)
            else:
                generated_samples = generator(noise)
                g_l1_loss = 0
            

            fake_labels = torch.zeros(batch_size, dtype=torch.long).to(self.device)
            
            if self.ablation_type in [2, 3, 4]:
                d_output_fake, d_prob_fake, l1_loss_fake = discriminator(generated_samples.detach())
                d_loss_fake = nn.CrossEntropyLoss()(d_output_fake, fake_labels) + l1_loss_fake
            else:
                d_output_fake, d_prob_fake = discriminator(generated_samples.detach())
                d_loss_fake = nn.CrossEntropyLoss()(d_output_fake, fake_labels)
            

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            

            g_optimizer.zero_grad()
            

            real_labels_for_g = torch.ones(batch_size, dtype=torch.long).to(self.device)
            
            if self.ablation_type in [2, 3, 4]:
                d_output_fake_g, d_prob_fake_g, l1_loss_fake_g = discriminator(generated_samples)
                g_loss = nn.CrossEntropyLoss()(d_output_fake_g, real_labels_for_g) + g_l1_loss + l1_loss_fake_g
            else:
                d_output_fake_g, d_prob_fake_g = discriminator(generated_samples)
                g_loss = nn.CrossEntropyLoss()(d_output_fake_g, real_labels_for_g)
            
            g_loss.backward()
            g_optimizer.step()
            

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            if isinstance(g_l1_loss, torch.Tensor):
                total_l1_loss += g_l1_loss.item()
        
        avg_d_loss = total_d_loss / n_batches
        avg_g_loss = total_g_loss / n_batches
        avg_l1_loss = total_l1_loss / n_batches
        
        return avg_d_loss, avg_g_loss, avg_l1_loss
    
    def train(self):

        print("="*60)
        print(f"Starting the ablation experiment training")
        print(f"Model: {ABLATION_NAMES[ABLATION_CONFIGS[self.ablation_type]]}")
        print(f"Data: {self.data_config}")
        print("=" * 60)
        
        try:

            X_train, y_train, input_shape = self.load_data()
            

            discriminator, generator = self.create_models(input_shape)
            

            d_optimizer, g_optimizer = self.create_optimizers(discriminator, generator)
            

            history = {
                'discriminator_loss': [],
                'generator_loss': [],
                'l1_loss': [],
                'total_loss': []
            }
            
            best_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            start_time = time.time()
            
            for epoch in range(self.epochs):

                d_loss, g_loss, l1_loss = self.train_epoch(
                    discriminator, generator, d_optimizer, g_optimizer,
                    X_train, y_train, epoch
                )
                
                total_loss = d_loss + g_loss
                

                history['discriminator_loss'].append(d_loss)
                history['generator_loss'].append(g_loss)
                history['l1_loss'].append(l1_loss)
                history['total_loss'].append(total_loss)
                

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}] - "
                          f"D_loss: {d_loss:.4f}, G_loss: {g_loss:.4f}, "
                          f"L1_loss: {l1_loss:.4f}, Total: {total_loss:.4f}")
                

                if total_loss < best_loss:
                    best_loss = total_loss
                    patience_counter = 0
                    

                    torch.save({
                        'epoch': epoch + 1,
                        'discriminator_state_dict': discriminator.state_dict(),
                        'generator_state_dict': generator.state_dict(),
                        'd_optimizer_state_dict': d_optimizer.state_dict(),
                        'g_optimizer_state_dict': g_optimizer.state_dict(),
                        'discriminator_loss': d_loss,
                        'generator_loss': g_loss,
                        'total_loss': total_loss,
                        'ablation_type': self.ablation_type,
                        'data_config': self.data_config
                    }, os.path.join(self.results_dir, 'best_model.pth'))
                else:
                    patience_counter += 1
                

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            end_time = time.time()
            training_time = end_time - start_time


            print(f"Training completed! Time taken: {training_time:.2f} seconds")
            print(f"Best loss: {best_loss:.4f}")


            with open(os.path.join(self.results_dir, 'training_history.pkl'), 'wb') as f:
                pickle.dump(history, f)
            

            self.plot_loss_curves(history)
            
            return history, training_time
            
        except Exception as e:
            print(f"An error occurred during the training process: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, 0
    
    def plot_loss_curves(self, history):

        plt.figure(figsize=(15, 5))
        

        plt.subplot(1, 3, 1)
        plt.plot(history['discriminator_loss'], label='Discriminator Loss')
        plt.title('Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        

        plt.subplot(1, 3, 2)
        plt.plot(history['generator_loss'], label='Generator Loss')
        plt.title('Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        

        plt.subplot(1, 3, 3)
        plt.plot(history['total_loss'], label='Total Loss')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        

        loss_plot_path = os.path.join(self.results_dir, 'loss_curves.png')
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"The loss curve has been saved to: {loss_plot_path}")

def run_single_ablation(ablation_type, data_config):

    print(f"\n{'='*80}")
    print(f"Start the ablation experiment: {ABLATION_NAMES[ABLATION_CONFIGS[ablation_type]]} - {data_config}")
    print(f"{'='*80}")
    
    trainer = AblationTrainer(ablation_type, data_config)
    history, training_time = trainer.train()
    
    if history is not None:
        print(f"✅ The ablation experiment is completed: {ABLATION_NAMES[ABLATION_CONFIGS[ablation_type]]} - {data_config}")
        print(f"⏱️  Time taken: {training_time:.2f}S")
        return True
    else:
        print(f"❌ The ablation experiment has failed: {ABLATION_NAMES[ABLATION_CONFIGS[ablation_type]]} - {data_config}")
        return False

if __name__ == "__main__":

    ablation_type = 1
    data_config = "win64_overlap0"
    
    success = run_single_ablation(ablation_type, data_config)

    if success:
        print("The training of the ablation experiment has been successfully completed!")
    else:
        print("The training of the ablation experiment has failed!")