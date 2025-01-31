import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
    
class SparseAutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_size = cfg['input_size']
        self.hidden_size = cfg['hidden_size']
        self.nonlinearity = cfg.get('NONLINEARITY', "ReLU")
        # Convert dtype string to torch dtype if necessary
        self.dtype = cfg['dtype'] if isinstance(cfg['dtype'], torch.dtype) else getattr(torch, cfg['dtype'])
        
        # Initialize layers with specified dtype
        self.encoder = nn.Linear(self.input_size, self.hidden_size, bias=True, dtype=self.dtype)
        self.decoder = nn.Linear(self.hidden_size, self.input_size, bias=True, dtype=self.dtype)
        self.norm_factor = 0.1

        if self.nonlinearity == "TopK":
            k = cfg["top_k"]
            self.activation = TopK(k=k)
        else:
            self.activation = nn.ReLU()

        # Move model to specified dtype
        self.to(self.dtype)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        """https://transformer-circuits.pub/2024/april-update/index.html#training-saes"""
        self.encoder.bias.data.zero_()
        self.decoder.bias.data.zero_()
        for i in range(self.hidden_size):
            col_norm = torch.norm(self.decoder.weight[:, i])
            self.decoder.weight[:, i] = self.norm_factor * (self.decoder.weight[:, i] / col_norm)
        self.encoder.weight.data = self.decoder.weight.data.T.clone()

    def forward(self, x):
        features = self.activation(self.encoder(x))
        x_reconstruct = self.decoder(features)
        reconstruct_loss = F.mse_loss(x_reconstruct, x)
        l2_penalty = torch.sum(features * torch.norm(self.encoder.weight, p=2, dim=-1))
        return reconstruct_loss, l2_penalty, x_reconstruct, features
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, features):
        return self.decoder(features)
    
    @torch.no_grad()
    def normalize_weights(self):
        """Ensures that the model has identical predictions but the decoder weight columns have L2 norm of 1"""
        # Compute the L2 norm of each column of the decoder weights
        W_dec_norm = torch.norm(self.decoder.weight, dim=0, keepdim=True)  # Shape: [1, 128]

        # Normalize decoder weights column-wise
        self.decoder.weight.data = self.decoder.weight / W_dec_norm  # Normalize each column of decoder weights
        
        # Apply the same normalization to the encoder weights (transpose it to match dimensions)
        self.encoder.weight.data = self.encoder.weight * W_dec_norm.T  # Adjust encoder weights accordingly

        self.encoder.weight.bias = self.decoder.bias * W_dec_norm.T

class TopK(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.post_act_fn = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get top-k values and indices
        top_k = torch.topk(x, k=self.k, dim=-1)
        values = self.post_act_fn(top_k.values)
        # Create a zero tensor and scatter top-k values back
        result = torch.zeros_like(x)
        result.scatter_(-1, top_k.indices, values)
        return result