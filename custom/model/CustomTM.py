import torch
import torch.nn as nn
from layers.Projections import Permutation, TimeProjection, Residual
from layers.Wavelets import WaveletDecomposition, WaveletReconstruction
from layers.Attention import VanillaAttention, GeometricAttention, SelfAttentionLayer, TopKAttention, CosineAttention
from layers.FeedForward import FeedForward

class CustomTM(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.variables          = configs.variables
        self.length             = configs.length
        self.pseudo_length      = configs.pseudo_length
        self.prediction_length  = configs.prediction_length
        self.dropout            = configs.dropout
        self.m                  = configs.m
        self.learnable_wavelets = configs.learnable_wavelets
        self.wv                 = configs.wv
        self.pad_mode           = configs.pad_mode
        self.inverted           = configs.inverted
        self.alpha              = configs.alpha
        self.scale              = configs.scale
        self.attention_dropout  = configs.attention_dropout
        self.normalize          = configs.normalize
        self.transformer_layers = configs.transformer_layers
        self.encoder_activation = configs.encoder_activation
        self.feedforward_dim    = configs.feedforward_dim
        self.attention_type     = configs.attention_type
        self.top_k              = configs.top_k
        
        # create model layers
        self.layers = self._create_layers()

    def _create_layers(self):

        # parse activation function
        if self.encoder_activation == "gelu":
            self.activation = nn.GELU
        elif self.encoder_activation == "relu":
            self.activation = nn.ReLU
        else:
            raise ValueError(f"Activation function {self.encoder_activation} not supported!")

        # store list of layers
        model_layers = []

        # original permutation
        orig_permutation = Permutation(0, 2, 1)
        model_layers.append(orig_permutation)

        # linear projection
        time_projection = TimeProjection(self.length, self.pseudo_length, self.dropout)
        model_layers.append(time_projection)

        # wavelet blocks
        wavelet_attention_layers = []
        for _ in range(self.transformer_layers):

            # wavelet decomposition
            wavelet_decomp = WaveletDecomposition(
                self.variables, self.m, self.learnable_wavelets, self.wv, self.pad_mode
            )
            wavelet_attention_layers.append(wavelet_decomp)

            if self.attention_type == "geometric":
                attention = GeometricAttention(self.scale, self.attention_dropout, self.alpha)
            elif self.attention_type == "cosine":
                attention = CosineAttention(self.scale, self.attention_dropout)
            elif self.attention_type == "topk":
                attention = TopKAttention(self.top_k, self.scale, self.attention_dropout)
            else:  # vanilla
                attention = VanillaAttention(self.scale, self.attention_dropout)
            attention_layer = SelfAttentionLayer(
                attention, self.pseudo_length, self.attention_dropout
            )
            wavelet_attention_layers.append(Residual(
                attention_layer, self.pseudo_length, self.attention_dropout
            ))

            # wavelet reconstruction
            wavelet_recon = WaveletReconstruction(
                self.variables, self.m, self.learnable_wavelets, self.wv, self.pad_mode
            )
            wavelet_attention_layers.append(wavelet_recon)

            # feedforward layer
            feedforward = FeedForward(self.pseudo_length, self.feedforward_dim, self.dropout, self.activation())
            wavelet_attention_layers.append(feedforward)
        
        model_layers += wavelet_attention_layers

        # swap linear projection and first wavelet decomposition
        if self.inverted:

            #TODO: remove hard-coded indices for projection/wavelet
            model_layers[1], model_layers[2] = model_layers[2], model_layers[1]
        
        # forecasting projection layer
        projection = nn.Linear(self.pseudo_length, self.prediction_length)
        model_layers.append(projection)

        # undo original permutation
        final_permutation = Permutation(0, 2, 1)
        model_layers.append(final_permutation)
        
        # combine all layers into Sequential
        return nn.Sequential(*model_layers)
    
    def forward(self, x : torch.Tensor):
        if self.normalize:
            x_mean = x.mean(1, keepdim=True).detach()
            x_std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = (x - x_mean) / x_std
        
        output = self.layers(x)

        if self.normalize:
            std_readd = x_std.repeat(1, self.prediction_length, 1)
            mean_readd = x_mean.repeat(1, self.prediction_length, 1)
            output = (output * std_readd) + mean_readd
        
        return output