import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class SelfAttentionLayer(nn.Module):
  def __init__(self, attention, pseudo_length, dropout=0.1):
    super().__init__()

    self.attention = attention
    self.pseudo_length = pseudo_length
    self.dropout = dropout

    # query, key, and values projections
    self.query_projection = nn.Sequential(
      nn.Linear(pseudo_length, pseudo_length),
      nn.Dropout(self.dropout)
    )

    self.key_projection = nn.Sequential(
      nn.Linear(pseudo_length, pseudo_length),
      nn.Dropout(self.dropout)
    )

    self.value_projection = nn.Sequential(
      nn.Linear(pseudo_length, pseudo_length),
      nn.Dropout(self.dropout)
    )

    # linear out projection
    self.out_projection = nn.Linear(self.pseudo_length, self.pseudo_length)

  def forward(self, x):

    # x: BxCx(M+1)xL'
    # queries, keys, values: BxL'x(M+1)xC
    queries = self.query_projection(x).permute(0, 3, 2, 1)
    keys = self.key_projection(x).permute(0, 3, 2, 1)
    values = self.value_projection(x).permute(0, 3, 2, 1)
    
    # attention outputs: BxL'x(M+1)xC
    attn_out = self.attention(queries, keys, values)

    # layer outputs: BxCx(M+1)xL'
    projection_out = self.out_projection(attn_out.permute(0, 3, 2, 1))

    return projection_out

class VanillaAttention(nn.Module):
  def __init__(self, scale=None, dropout=0.1):
    super().__init__()

    self.scale = scale
    self.dropout = nn.Dropout(dropout)

  def forward(self, queries, keys, values):

    # queries, keys, values: BxL'x(M+1)xC
    E = queries.shape[3]

    # scale attention outputs appropriately
    scale = self.scale or 1.0 / sqrt(E)

    # dot product attention matrix: Bx(M+1)xL'xL'
    scores = torch.einsum("blhe,bshe->bhls",queries,keys) * scale
    A = self.dropout(torch.softmax(scores,dim=-1))

    # attention dot values: BxL'x(M+1)xC
    attn_out = torch.einsum("bhls,bshd->blhd", A, values)

    return attn_out.contiguous()
  
class GeometricAttention(nn.Module):
  def __init__(self, scale=None, dropout=0.1, alpha=1.0):
    super().__init__()

    self.scale = scale
    self.dropout=nn.Dropout(dropout)
    self.alpha = alpha

  def forward(self, queries, keys, values):

    # queries: BxLxHxE
    # values: BxSxHxE
    # keys: BxSxHxD
    # all equiv. BxL'x(M+1)xC
    E = queries.shape[3]

    # scale attention outputs appropriately
    scale = self.scale or 1.0 / sqrt(E)

    # dot product attention matrix: Bx(M+1)xL'xL'
    dot_product = torch.einsum("blhe,bshe->bhls", queries, keys)

    # queries norm: Bx(M+1)xL'x1
    # keys norm: Bx(M+1)x1xL'
    queries_norm2 = torch.sum(queries**2, dim=-1).permute(0, 2, 1).unsqueeze(-1)
    keys_norm2 = torch.sum(keys**2, dim=-1).permute(0, 2, 1).unsqueeze(-2)

    # wedge_norm: Bx(M+1)xL'xL'
    wedge_norm2 = queries_norm2 * keys_norm2 - dot_product **2
    wedge_norm2 = F.relu(wedge_norm2)
    wedge_norm=torch.sqrt(wedge_norm2 + 1e-8)

    # geometric attention matrix: Bx(M+1)xL'xL'
    scores= ((1 - self.alpha) * dot_product + self.alpha * wedge_norm) * scale
    A = self.dropout(torch.softmax(scores, dim=-1))

    # attention dot values: BxL'x(M+1)xC
    attn_out = torch.einsum("bhls,bshd->blhd", A, values)
    
    return attn_out.contiguous()