import torch
import torch.nn as nn
from models.attention import attention, multi_head_attention, make_window_mask
from models.layers import SwiGLU
from models.layers import WindowedSelfAttention

"""
class WSA(nn.Module):
  def __init__(self, d_model, num_heads):
    super().__init__()
    self.num_heads = num_heads
    self.head_dim = d_model // num_heads 
    self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
    self.out_proj = nn.Linear(d_model, d_model, bias=False)

  def forward(self, x, mask=None):
    x = self.qkv_proj(x) 
    Q, K, V = torch.chunk(x, 3, dim=-1) 
    out, attn_weights = multi_head_attention(Q, K, V, num_heads=self.num_heads, mask=mask)
    out = self.out_proj(out)
    return out, attn_weights



class WSALayer(nn.Module):
  def __init__(self, d_model, num_heads, dropout):
    super().__init__()
    self.wsa = WSA(d_model, num_heads)
    self.ffn = SwiGLU(d_model, dropout=dropout)
    self.norm1 = nn.LayerNorm(d_model, bias=True) 
    self.norm2 = nn.LayerNorm(d_model, bias=True)
    self.dropout = nn.Dropout(dropout) 

  def forward(self, x, mask=None):
    attn_out, attn_weight = self.wsa(self.norm1(x), mask)
    x = x + self.dropout(attn_out)
    x = x + self.dropout(self.ffn(self.norm2(x)))
    return x, attn_weight
"""


class TemporalBlock(nn.Module):
  def __init__(self, d_model, num_heads, dropout, window_sizes):
    super().__init__()
    self.layers = nn.ModuleList(
        WSALayer(d_model, num_heads, dropout)
        for _ in range(len(window_sizes))
    )
    self.window_sizes = window_sizes

  def forward(self, x):
    seq_len = x.shape[1]
    attn_maps = []
    for layer, w in zip(self.layers, self.window_sizes):
      mask = None if w == 0 else make_window_mask(seq_len, w)
      x, attn = layer(x, mask)
      attn_maps.append(attn)

    return x, attn_maps




class CrossModalAttention(nn.Module):
  def __init__(self, d_model, num_heads):
    super().__init__()
    self.num_heads = num_heads
    self.head_dim = d_model // num_heads
    self.q_proj = nn.Linear(d_model, d_model, bias=False)
    self.k_proj = nn.Linear(d_model, d_model, bias=False)
    self.v_proj = nn.Linear(d_model, d_model, bias=False)
    self.out_proj = nn.Linear(d_model, d_model, bias=False)

  def forward(self, query, context):
    Q = self.q_proj(query)
    K = self.k_proj(context)
    V = self.v_proj(context)

    out, attn_weights = multi_head_attention(Q, K, V, num_heads=self.num_heads,
                                             mask=None)
    out = self.out_proj(out)
    return out, attn_weights



class CrossModalAttentionLayer(nn.Module):
  def __init__(self, d_model, num_heads, dropout):
    super().__init__()
    self.cma = CrossModalAttention(d_model, num_heads)
    self.ffn = SwiGLU(d_model, dropout)
    self.norm_q = nn.LayerNorm(d_model, bias=True)
    self.norm_c = nn.LayerNorm(d_model, bias=True) 
    self.norm2 = nn.LayerNorm(d_model, bias=True)
    self.dropout = nn.Dropout(dropout)

  def forward(self, query, context):
    attn_out, attn_weights = self.cma(self.norm_q(query), self.norm_c(context))
    query = query + self.dropout(attn_out)
    ffn_out = self.ffn(self.norm2(query))
    query = query + self.dropout(ffn_out)

    return query, attn_weights



class CrossModalFusionBlock(nn.Module):
  def __init__(self, d_model, num_heads, dropout):
    super().__init__()
    self.num_heads = num_heads 
    self.cma_layer = CrossModalAttentionLayer(d_model, num_heads, dropout) 

  def forward(self, fusion, visual, text, audio):
    query = fusion
    context = torch.stack((visual, text, audio), dim = 2)
    B, T, D = query.shape
    N = context.shape[2] 
    query_reshaped = query.reshape(B * T, 1, D)
    context_reshaped = context.reshape(B * T, N, D)
    query_update, attn_weights = self.cma_layer(query_reshaped, context_reshaped)
    fusion_update = query_update.reshape(B, T, D) # geri aç
    attn_weights = attn_weights.squeeze(2).view(B, T , self.num_heads, N)

    return fusion_update, attn_weights


"""
class MSTBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout, window_size):
      super().__init__()
      self.wsa_layer = WSALayer(d_model=d_model, num_heads=num_heads, dropout=dropout)
      self.window_size = window_size

    def forward(self, fusion, visual, text, audio, mask=None):
      seq_len = fusion.shape[1]
      window_mask = None if self.window_size == 0 else make_window_mask(seq_len, self.window_size)
      fusion, _ = self.wsa_layer(fusion, window_mask)
      visual, _ = self.wsa_layer(visual, window_mask)
      text, _ = self.wsa_layer(text, window_mask)
      audio, _ = self.wsa_layer(audio, window_mask)

      return fusion, visual, text, audio

"""

# bizim yazdığımız wsa ile paper'daki wsa arasında ufak fark var. orjinalden devam..
# Windowed Self-Attention Layer
class WindowedSelfAttentionLayer(nn.Module):
    def __init__(self, input_dim, num_heads, dropout, window_size):
        super().__init__()
        self.wsa = WindowedSelfAttention(
            input_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            window_size=window_size,
        )
        self.ffn = SwiGLU(input_dim=input_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.wsa(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        return x

# Multi-Scale Temporal Block
class MultiScaleTemporalBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dropout, window_size):
        super().__init__()
        self.wsa_layer = WindowedSelfAttentionLayer(
            input_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            window_size=window_size
        )

    def forward(self, fusion, visual, text, audio, mask=None):
        fusion = self.wsa_layer(fusion, mask)
        visual = self.wsa_layer(visual, mask)
        text = self.wsa_layer(text, mask)
        audio = self.wsa_layer(audio, mask)
        return fusion, visual, text, audio