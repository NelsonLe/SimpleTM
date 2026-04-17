import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionProjection(nn.Module):
    def __init__(self, kernel_size, channels):
        super(ConvolutionProjection, self).__init__()

        self.kernel_size = kernel_size
        self.channels = channels

        self.conv_weights = nn.Parameter(torch.Tensor(1, 1, self.kernel_size))
        nn.init.xavier_uniform_(self.conv_weights)
    
    def forward(self, x):
        weights_to_use = self.conv_weights.expand(self.channels, 1, self.kernel_size)
        return F.conv1d(x, weights_to_use, stride=self.kernel_size, groups=self.channels)

class DataEmbedding_inverted(nn.Module):
    def __init__(self, length, pseudo_length, embed_type='fixed', freq='h', dropout=0.1,
                 use_projection="linear", channels=None):
        
        super(DataEmbedding_inverted, self).__init__()

        self.use_projection = use_projection

        # linear projection
        if use_projection == "linear":
            self.value_embedding = nn.Linear(length, pseudo_length)
        else:
            self.kernel_size = length // pseudo_length
        
            # convolutional projection
            if use_projection == "convolutional":
                if channels is None:
                    raise ValueError("Must specify channel number for convolutional projection.")
                self.channels = channels

                self.value_embedding = ConvolutionProjection(self.kernel_size, self.channels)
            
            # average pooled projection
            else:
                if use_projection != "pooling":
                    raise ValueError(f"Invalid projection mode {use_projection} specified."
                                      "Must be one of 'linear', 'convolutional', or 'pooling'") 
                self.value_embedding = nn.AvgPool1d(kernel_size=self.kernel_size)

        # dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):

        x = x.permute(0, 2, 1)

        if x_mark is None:
            x = self.value_embedding(x)
        else:
            raise NotImplementedError("Cannot use x_mark values!")
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) 

        return self.dropout(x)