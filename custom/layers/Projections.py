import torch.nn as nn

class Permutation(nn.Module):
    def __init__(self, *permutation_dims):
        super().__init__()
        self.permutation_dims = permutation_dims
    
    def forward(self, x):
        return x.permute(self.permutation_dims)

class TimeProjection(nn.Module):
    def __init__(self, length, pseudo_length, dropout=0.1):
        super().__init__()

        self.length = length
        self.pseudo_length = pseudo_length
        self.dropout = dropout

        # linear and dropout layers
        self.linear_layer = nn.Linear(self.length, self.pseudo_length)
        self.dropout_layer = nn.Dropout(self.dropout)
    
    def forward(self, x):
        return self.dropout_layer(self.linear_layer(x))