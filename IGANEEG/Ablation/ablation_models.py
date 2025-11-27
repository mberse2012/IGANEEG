"""
IGANEEG ablation experiment model variants
It includes four different model configurations for the ablation experiment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
from sklearn.decomposition import FastICA
import warnings
warnings.filterwarnings('ignore')

class L1FeatureSelector(nn.Module):
   
    def __init__(self, input_dim, l1_lambda=0.01):
        super(L1FeatureSelector, self).__init__()
        self.input_dim = input_dim
        self.l1_lambda = l1_lambda
        
      
        self.feature_weights = nn.Parameter(torch.ones(input_dim))
        
    def forward(self, x):
     
        weights = torch.sigmoid(self.feature_weights)
        selected_features = x * weights
        
        l1_loss = self.l1_lambda * torch.sum(torch.abs(self.feature_weights))
        
        return selected_features, l1_loss, weights

class SimpleDiscriminator(nn.Module):

    def __init__(self, input_size, hidden_dims=[256, 64, 2]):
        super(SimpleDiscriminator, self).__init__()
        
        layers = []
        current_dim = input_size
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if hidden_dim != hidden_dims[-1]:  
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(0.3))
            current_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
     
        x_flat = x.view(x.size(0), -1)
        
        output = self.network(x_flat)
        
        probabilities = self.softmax(output)
        
        return output, probabilities

class SimpleGenerator(nn.Module):

    def __init__(self, noise_dim=64, output_size=1024, hidden_dims=[128, 64, 128, 64]):
        super(SimpleGenerator, self).__init__()
        
        layers = []
        current_dim = noise_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.3))
            current_dim = hidden_dim
        
    
        layers.append(nn.Linear(current_dim, output_size))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, noise):
        return self.network(noise)

class DiscriminatorNoL1(nn.Module):
    def __init__(self, input_size, hidden_dims=[256, 64, 2]):
        super(DiscriminatorNoL1, self).__init__()
        
        self.feature_layers = nn.ModuleList()
        current_dim = input_size
        
        for i, hidden_dim in enumerate(hidden_dims[:-1]):
            self.feature_layers.append(nn.Linear(current_dim, hidden_dim))
            self.feature_layers.append(nn.LeakyReLU(0.2))
            self.feature_layers.append(nn.Dropout(0.3))
            current_dim = hidden_dim
        
        self.final_layer = nn.Linear(current_dim, hidden_dims[-1])
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
   
        x_flat = x.view(x.size(0), -1)
        
        current = x_flat
        for layer in self.feature_layers:
            current = layer(current)
        
        output = self.final_layer(current)
        probabilities = self.softmax(output)
        
        return output, probabilities

class GeneratorNoAutoencoder(nn.Module):

    def __init__(self, noise_dim=64, output_size=1024, hidden_dims=[128, 64, 128, 64]):
        super(GeneratorNoAutoencoder, self).__init__()
        
  
        self.l1_selectors = nn.ModuleList()
        current_dim = noise_dim
        
    
        for i in range(2):
            self.l1_selectors.append(L1FeatureSelector(current_dim))
            current_dim = hidden_dims[i]
        
    
        self.layers = nn.ModuleList()
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(current_dim if i > 0 else noise_dim, hidden_dim))
            if i < len(hidden_dims) - 1:
                self.layers.append(nn.ReLU())
                self.layers.append(nn.BatchNorm1d(hidden_dim))
                self.layers.append(nn.Dropout(0.3))
            current_dim = hidden_dim
        
       
        self.output_layer = nn.Linear(current_dim, output_size)
        self.tanh = nn.Tanh()
        
    def forward(self, noise):
        current = noise
        total_l1_loss = 0
        
    
        for i, l1_selector in enumerate(self.l1_selectors):
            current, l1_loss, weights = l1_selector(current)
            total_l1_loss += l1_loss
            
        
            layer_idx = i * 2 
            current = self.layers[layer_idx](current)
            if layer_idx + 1 < len(self.layers):
                current = self.layers[layer_idx + 1](current)
        
    
        start_idx = len(self.l1_selectors) * 2
        for i in range(start_idx, len(self.layers)):
            current = self.layers[i](current)
        
    
        output = self.tanh(self.output_layer(current))
        
        return output, total_l1_loss

class DiscriminatorWithL1(nn.Module):

    def __init__(self, input_size, hidden_dims=[256, 64, 2]):
        super(DiscriminatorWithL1, self).__init__()
        
       
        self.l1_selectors = nn.ModuleList()
        current_dim = input_size
        
        for i in range(len(hidden_dims) - 1):
            self.l1_selectors.append(L1FeatureSelector(current_dim))
            current_dim = hidden_dims[i]
        
  
        self.layers = nn.ModuleList()
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(current_dim if i > 0 else input_size, hidden_dim))
            if i < len(hidden_dims) - 1:
                self.layers.append(nn.LeakyReLU(0.2))
                self.layers.append(nn.Dropout(0.3))
            current_dim = hidden_dim
        
   
        self.feature_fusion = nn.Linear(hidden_dims[-2] + hidden_dims[-2], hidden_dims[-2])
        self.final_layer = nn.Linear(hidden_dims[-2], hidden_dims[-1])
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        current = x_flat
        total_l1_loss = 0
        selected_features_list = []
        
      
        for i, l1_selector in enumerate(self.l1_selectors):
            current, l1_loss, weights = l1_selector(current)
            total_l1_loss += l1_loss
            selected_features_list.append(current)
            
         
            layer_idx = i * 2
            current = self.layers[layer_idx](current)
            if layer_idx + 1 < len(self.layers):
                current = self.layers[layer_idx + 1](current)
        
      
        if len(selected_features_list) >= 2:
            fused_features = self.feature_fusion(
                torch.cat([selected_features_list[-2], selected_features_list[-1]], dim=1)
            )
            current = fused_features
        
        output = self.final_layer(current)
        probabilities = self.softmax(output)
        
        return output, probabilities, total_l1_loss

class FullIGANEEGGenerator(nn.Module):

    def __init__(self, noise_dim=64, output_size=1024, hidden_dims=[128, 64, 128, 64]):
        super(FullIGANEEGGenerator, self).__init__()

        self.l1_selectors = nn.ModuleList()
        current_dim = noise_dim
        
        for i in range(2):
            self.l1_selectors.append(L1FeatureSelector(current_dim))
            current_dim = hidden_dims[i]
        
      
        self.layers = nn.ModuleList()
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(current_dim if i > 0 else noise_dim, hidden_dim))
            if i < len(hidden_dims) - 1:
                self.layers.append(nn.ReLU())
                self.layers.append(nn.BatchNorm1d(hidden_dim))
                self.layers.append(nn.Dropout(0.3))
            current_dim = hidden_dim
        
    
        self.encoder_output = hidden_dims[1]  
        self.decoder_input = hidden_dims[2]   
        
        self.feature_fusion = nn.Linear(self.encoder_output + self.decoder_input, self.decoder_input)
        
  
        self.output_layer = nn.Linear(current_dim, output_size)
        self.tanh = nn.Tanh()
        
    def forward(self, noise):
        current = noise
        total_l1_loss = 0
        selected_features_list = []
        
  
        for i, l1_selector in enumerate(self.l1_selectors):
            current, l1_loss, weights = l1_selector(current)
            total_l1_loss += l1_loss
            selected_features_list.append(current)
            
      
            layer_idx = i * 2
            current = self.layers[layer_idx](current)
            if layer_idx + 1 < len(self.layers):
                current = self.layers[layer_idx + 1](current)
        
 
        encoder_output = selected_features_list[-1]
        
     
        start_idx = len(self.l1_selectors) * 2
        decoder_input = self.layers[start_idx](current)
        
   
        fused_features = self.feature_fusion(
            torch.cat([encoder_output, decoder_input], dim=1)
        )
        
       
        current = fused_features
        for i in range(start_idx + 1, len(self.layers)):
            current = self.layers[i](current)
        
  
        output = self.tanh(self.output_layer(current))
        
        return output, total_l1_loss

class AblationModelFactory:

    
    @staticmethod
    def create_model(ablation_type, input_size=None, noise_dim=64, output_size=None):
             
        if ablation_type == "simple_gan":
     
            discriminator = SimpleDiscriminator(input_size)
            generator = SimpleGenerator(noise_dim, output_size)
            return discriminator, generator
            
        elif ablation_type == "no_l1":
    
            discriminator = DiscriminatorNoL1(input_size)
            generator = GeneratorNoAutoencoder(noise_dim, output_size)
            return discriminator, generator
            
        elif ablation_type == "no_autoencoder":
          
            discriminator = DiscriminatorWithL1(input_size)
            generator = GeneratorNoAutoencoder(noise_dim, output_size)
            return discriminator, generator
            
        elif ablation_type == "full_igan":
      
            discriminator = DiscriminatorWithL1(input_size)
            generator = FullIGANEEGGenerator(noise_dim, output_size)
            return discriminator, generator
            
        else:
            raise ValueError(f"Unknown ablation type: {ablation_type}")


ABLATION_CONFIGS = {
    1: "simple_gan",
    2: "no_l1", 
    3: "no_autoencoder",
    4: "full_igan"
}

ABLATION_NAMES = {
        "simple_gan": "Simple GAN (no L1, no autoencoder)",
        "no_l1": "Remove L1 feature selection",
        "no_autoencoder": "Remove the autoencoder",
        "full_igan": "Full IGANEEG"
}