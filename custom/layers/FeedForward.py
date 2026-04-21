import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, pseudo_length, hidden_length=None, dropout=0.1, activation=None):
        super().__init__()

        self.pseudo_length = pseudo_length
        self.hidden_length = hidden_length or 4 * self.pseudo_length
        self.dropout = dropout

        # linear layers
        self.lin1 = nn.Linear(self.pseudo_length, self.hidden_length)
        self.lin2 = nn.Linear(self.hidden_length, self.pseudo_length)

        # activation layer
        self.activation = activation or nn.GELU()

        # dropout layer
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # layernorm
        self.layernorm = nn.LayerNorm(self.pseudo_length)
    
    def forward(self, x):

        # x: BxCxL'
        # first linear output: BxCxL''
        out1 = self.dropout1(self.activation(self.lin1(x)))

        # second linear output: BxCxL'
        out2 = self.dropout2(self.lin2(out1))

        # residual connection and layernorm
        out2 += x
        norm_out = self.layernorm(out2)

        return norm_out