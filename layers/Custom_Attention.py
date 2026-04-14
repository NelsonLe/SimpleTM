import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class VanillaAttention(nn.Module):
  def __init__(self,scale=None,mask_flag=False,attention_dropout=0.1,output_attention=False):
    super().__init__()
    self.scale=scale
    self.mask_flag=mask_flag
    self.output_attention=output_attention
    self.dropout=nn.Dropout(attention_dropout)
  def forward(self,queries,keys,values,attn_mask=None):
    B,L,H,E=queries.shape
    _,S,_,_=values.shape
    scale=self.scale or 1.0/sqrt(E)
    scores=torch.einsum("blhe,bshe->bhls",queries,keys)*scale
    if self.mask_flag:
      if attn_mask is None:
        attn_mask=torch.tril(torch.ones(L,S,device=scores.device))
      scores=scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0)==0,float("-inf"))
    A=self.dropout(torch.softmax(scores,dim=-1))
    V=torch.einsum("bhls,bshd->blhd",A,values)
    if self.output_attention:
      return V.contiguous(),A
    return V.contiguous(), None

class GeometricAttention(nn.Module):
  def __init__(self,scale=None,mask_flag=False,attention_dropout=0.1,output_attention=False,alpha=1.0):
    super().__init__()
    self.scale=scale
    self.mask_flag=mask_flag
    self.output_attention=output_attention
    self.dropout=nn.Dropout(attention_dropout)
    self.alpha=alpha
  def forward(self,queries,keys,values,attn_mask=None):
    B,L,H,E=queries.shape
    _,S,_,_=values.shape
    scale=self.scale or 1.0/sqrt(E)
    dot_product=torch.einsum("blhe,bshe->bhls",queries,keys)
    queries_norm2=torch.sum(queries**2,dim=-1).permute(0,2,1).unsqueeze(-1)
    keys_norm2=torch.sum(keys**2,dim=-1).permute(0,2,1).unsqueeze(-2)
    wedge_norm2=queries_norm2*keys_norm2-dot_product**2
    wedge_norm2=F.relu(wedge_norm2)
    wedge_norm=torch.sqrt(wedge_norm2+1e-8)
    scores=((1-self.alpha)*dot_product+self.alpha*wedge_norm)*scale
    if self.mask_flag:
      if attn_mask is None:
        attn_mask=torch.tril(torch.ones(L,S,device=scores.device))
      scores=scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0)==0,float("-inf"))
    A=self.dropout(torch.softmax(scores,dim=-1))
    V=torch.einsum("bhls,bshd->blhd",A,values)
    if self.output_attention:
      return V.contiguous(),A
    return V.contiguous(), None

class CustomAttentionLayer(nn.Module):
  def __init__(self,attention,d_model,projection_dropout=0.0,):
    super().__init__()
    self.inner_attention=attention
    self.query_projection=nn.Sequential(nn.Linear(d_model, d_model),nn.Dropout(projection_dropout))
    self.key_projection=nn.Sequential(nn.Linear(d_model, d_model),nn.Dropout(projection_dropout))
    self.value_projection=nn.Sequential(nn.Linear(d_model, d_model),nn.Dropout(projection_dropout))
    self.out_projection=nn.Linear(d_model, d_model)
  def forward(self,queries,keys,values,attn_mask=None,tau=None,delta=None):
    B,L,D=queries.shape
    _,S,_=keys.shape
    queries=self.query_projection(queries).view(B,L,1,D)
    keys=self.key_projection(keys).view(B,S,1,D)
    values=self.value_projection(values).view(B,S,1,D)
    out,attn=self.inner_attention(queries,keys,values,attn_mask=attn_mask)
    out=out.reshape(B,L,D)
    out=self.out_projection(out)
    return out,attn
