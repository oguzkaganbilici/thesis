import torch

def attention(Q, K, V, mask=None):
  scores = Q @ K.transpose(-2, -1) / (Q.shape[-1]**0.5)

  if mask is not None:
    scores = scores.masked_fill(~mask, float("-inf"))

  attn_weights = torch.softmax(scores, dim=-1)
  out = attn_weights @ V
  return out, attn_weights



def make_window_mask(seq_len, window_size):
  radius = window_size // 2 
  i = torch.arange(seq_len).view(-1, 1) 
  j = torch.arange(seq_len).view(1, -1) 
  dist = torch.abs(i - j) 

  return dist <= radius 



def multi_head_attention(Q, K, V, num_heads, mask=None):
  batch_size, seq_len, d_model = Q.shape
  d_head = d_model // num_heads

  kv_len = K.shape[1]
  q_len =  Q.shape[1]

  qH = Q.reshape(batch_size, q_len, num_heads, d_head)
  qH = qH.transpose(-2, -3)

  kH = K.reshape(batch_size, kv_len, num_heads, d_head)
  kH = kH.transpose(-2, -3)

  vH = V.reshape(batch_size, kv_len, num_heads, d_head)
  vH = vH.transpose(-2, -3)

  out_h, attn_weights_H = attention(qH, kH, vH, mask)

  outH = out_h.transpose(1, 2).reshape(batch_size, q_len, d_model)

  return outH, attn_weights_H