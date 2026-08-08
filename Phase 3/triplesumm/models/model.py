import torch
import torch.nn as nn
from models.layers import SinusoidalTemporalPE, ModalityEmbedding
from models.blocks import MultiScaleTemporalBlock, CrossModalFusionBlock

class TripleSumm(nn.Module):
    def __init__(self, visual_dim = 1024, text_dim = 768,
                 audio_dim = 768, d_model = 128, hidden_dim = 192,
                 num_heads = 4, num_layers = 4, window_size = [5, 15, 45, 0],
                 dropout = 0.1, max_seq_len = 10_000):
        super().__init__()


        self.num_model_layers = 2
        self.num_mst_layers = 2
        self.num_cmf_layers = 2
        self.get_attn_weights = False


        # projections
        self.visual_proj = nn.Linear(visual_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.audio_proj = nn.Linear(audio_dim, d_model)

        # layernorms
        self.visual_ln = nn.LayerNorm(d_model)
        self.text_ln = nn.LayerNorm(d_model)
        self.audio_ln = nn.LayerNorm(d_model)

        # PE and modality embedding

        self.temporal_pe = SinusoidalTemporalPE(d_model=d_model, max_seq_len=max_seq_len, dropout=dropout)
        self.modality_embedding = ModalityEmbedding(d_model=d_model)

        # temporal block
        self.temporal_block = nn.ModuleList(
            [
                MultiScaleTemporalBlock(input_dim=d_model, num_heads=num_heads, dropout=dropout, window_size=window_size[i]) 
                for i in range(num_layers)
            ]
        )

        # modality block
        self.modality_block = nn.ModuleList(
            [
                CrossModalFusionBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, visual, text, audio, mask=None):
        visual = self.visual_ln(self.visual_proj(visual))
        text = self.text_ln(self.text_proj(text))
        audio = self.audio_ln(self.audio_proj(audio))

        fusion = (visual + text + audio) / 3

        fusion = self.temporal_pe(fusion)
        visual = self.temporal_pe(visual)
        text = self.temporal_pe(text)
        audio = self.temporal_pe(audio)
        
        fusion = fusion + self.modality_embedding(fusion, modality_index=0)
        visual = visual + self.modality_embedding(visual, modality_index=1)
        text = text + self.modality_embedding(text, modality_index=2)
        audio = audio + self.modality_embedding(audio, modality_index=3)
        
        attn_weights_list = []
        for i in range(self.num_model_layers):
            # Shared Multi-Scale Temporal block
            for j in range(self.num_mst_layers):
                fusion, visual, text, audio = self.temporal_block[i * self.num_mst_layers + j](fusion, visual, text, audio, mask)
            
            # Cross-Modal Fusion block
            for j in range(self.num_cmf_layers):
                fusion, attn_weights = self.modality_block[i * self.num_cmf_layers + j](fusion, visual, text, audio)
            
            if self.get_attn_weights:
                attn_weights_list.append(attn_weights.detach())
        
        if mask is not None:
            fusion = fusion * mask.unsqueeze(-1).float()

        out = self.head(fusion).squeeze(-1)
        return out, attn_weights_list
