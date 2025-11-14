# model/hftt.py - ORIGINAL VERSION
import torch
import torch.nn as nn
import torch.nn.functional as F
from BEATs.BEATs import BEATs, BEATsConfig

class FrequencyTemporalAttention(nn.Module):
    """
    HFTT's frequency-temporal attention for multi-scale features
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        """
        x: [B, N, D]
        """
        B, N, D = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D_h]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        
        return x


class MultiScaleTemporalPooling(nn.Module):
    """
    HFTT's multi-scale temporal pooling
    Capture both short-term (crackles) and long-term (wheezes) patterns
    """
    def __init__(self, dim, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.dim = dim
        
        # Attention pooling for each scale
        self.attention_pools = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.Tanh(),
                nn.Linear(dim // 4, 1)
            )
            for _ in scales
        ])
        
        # Cross-scale fusion
        self.fusion = nn.Sequential(
            nn.Linear(dim * len(scales), dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        """
        x: [B, N, D] features from BEATs
        """
        B, N, D = x.shape
        scale_features = []
        
        for scale, attn_pool in zip(self.scales, self.attention_pools):
            if scale > 1:
                # Average pooling for different temporal scales
                x_scaled = x.view(B, N // scale, scale, D).mean(dim=2)  # [B, N/scale, D]
            else:
                x_scaled = x
            
            # Attention-based pooling
            attn_weights = attn_pool(x_scaled)  # [B, N/scale, 1]
            attn_weights = F.softmax(attn_weights, dim=1)
            pooled = torch.sum(x_scaled * attn_weights, dim=1)  # [B, D]
            
            scale_features.append(pooled)
        
        # Fuse multi-scale features
        fused = torch.cat(scale_features, dim=-1)  # [B, 3*D]
        fused = self.fusion(fused)  # [B, D]
        
        return fused


class HybridHFTT(nn.Module):
    """
    Hybrid HFTT: BEATs encoder (pretrained) + HFTT improvements
    
    Architecture:
    1. BEATs encoder (frozen or fine-tuned) - provides audio representations
    2. Frequency-Temporal attention - exploit spectrogram structure
    3. Multi-scale temporal pooling - capture crackles & wheezes
    """
    def __init__(
        self,
        num_target_classes: int = 4,
        beats_model_path: str = "./pretrained_models/BEATs_iter3_plus_AS2M.pt",
        freeze_encoder: bool = False,
        use_multi_scale: bool = True,
        spec_transform=None,
    ):
        super().__init__()
        
        # 1. Load pretrained BEATs encoder
        checkpoint = torch.load(beats_model_path, map_location='cpu')
        cfg = BEATsConfig({
            **checkpoint["cfg"],
            "predictor_class": num_target_classes,
            "finetuned_model": False,
            "spec_transform": spec_transform
        })
        
        self.beats = BEATs(cfg)
        self.beats.load_state_dict(checkpoint["model"])
        
        # Freeze BEATs encoder if needed
        if freeze_encoder:
            for param in self.beats.parameters():
                param.requires_grad = False
        
        self.encoder_dim = cfg.encoder_embed_dim  # Usually 768
        self.final_feat_dim = self.encoder_dim
        
        # 2. HFTT improvements
        self.use_multi_scale = use_multi_scale
        
        if use_multi_scale:
            # Frequency-temporal attention
            self.freq_temp_attn = FrequencyTemporalAttention(
                dim=self.encoder_dim,
                num_heads=12,
                dropout=0.1
            )
            
            # Multi-scale temporal pooling
            self.multi_scale_pool = MultiScaleTemporalPooling(
                dim=self.encoder_dim,
                scales=[1, 2, 4]  # Short, medium, long patterns
            )
        else:
            # Simple attention pooling (baseline)
            self.attention_pool = nn.Sequential(
                nn.Linear(self.encoder_dim, self.encoder_dim // 4),
                nn.Tanh(),
                nn.Linear(self.encoder_dim // 4, 1)
            )
    
    def forward(self, source, padding_mask=None, training=False):
        """
        source: [B, T] raw audio waveform
        Returns: [B, 1, D] or [B, D] features
        """
        # BEATs encoder
        if padding_mask is not None:
            x, _ = self.beats.extract_features(source, padding_mask)
        else:
            x, _ = self.beats.extract_features(source)
        
        # x: [B, N, D] where N is number of temporal frames
        
        if self.use_multi_scale:
            # Apply frequency-temporal attention
            x_attn = self.freq_temp_attn(x)  # [B, N, D]
            x = x + x_attn  # Residual
            
            # Multi-scale temporal pooling
            x_pooled = self.multi_scale_pool(x)  # [B, D]
            
            # Expand for compatibility with PAFA
            x_pooled = x_pooled.unsqueeze(1)  # [B, 1, D]
            
            return x_pooled
        else:
            # Simple pooling (for comparison)
            attn_weights = self.attention_pool(x)  # [B, N, 1]
            attn_weights = F.softmax(attn_weights, dim=1)
            x_pooled = torch.sum(x * attn_weights, dim=1)  # [B, D]
            
            return x_pooled.unsqueeze(1)  # [B, 1, D]


# Factory function
def get_hftt_model(
    beats_model_path="./pretrained_models/BEATs_iter3_plus_AS2M.pt",
    freeze_encoder=False,
    spec_transform=None
):
    """
    Create Hybrid HFTT model
    
    Args:
        beats_model_path: Path to pretrained BEATs checkpoint
        freeze_encoder: If True, freeze BEATs encoder (only train HFTT head)
        spec_transform: SpecAugment transform
    """
    model = HybridHFTT(
        num_target_classes=4,
        beats_model_path=beats_model_path,
        freeze_encoder=freeze_encoder,
        use_multi_scale=True,
        spec_transform=spec_transform
    )
    return model

# # model/hftt.py - REGULARIZED VERSION
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from BEATs.BEATs import BEATs, BEATsConfig

# class FrequencyTemporalAttention(nn.Module):
#     """
#     Regularized frequency-temporal attention
#     """
#     def __init__(self, dim, num_heads=8, qkv_bias=True, dropout=0.3):  # CHANGED: 0.3
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
        
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.proj = nn.Linear(dim, dim)
#         self.dropout = nn.Dropout(dropout)
#         self.norm = nn.LayerNorm(dim)
        
#     def forward(self, x):
#         """
#         x: [B, N, D]
#         """
#         B, N, D = x.shape
        
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
#         qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D_h]
#         q, k, v = qkv[0], qkv[1], qkv[2]
        
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.dropout(attn)
        
#         x = (attn @ v).transpose(1, 2).reshape(B, N, D)
#         x = self.proj(x)
#         x = self.norm(x)  # ADDED: LayerNorm after projection
        
#         return x


# class MultiScaleTemporalPooling(nn.Module):
#     """
#     Regularized multi-scale temporal pooling
#     """
#     def __init__(self, dim, scales=[1, 2, 4]):
#         super().__init__()
#         self.scales = scales
#         self.dim = dim
        
#         # Simplified attention pooling with regularization
#         self.attention_pools = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(dim, dim // 8),  # CHANGED: Smaller hidden size
#                 nn.Dropout(0.2),  # ADDED
#                 nn.Linear(dim // 8, 1)
#             )
#             for _ in scales
#         ])
        
#         # Cross-scale fusion with regularization
#         self.fusion = nn.Sequential(
#             nn.Linear(dim * len(scales), dim),
#             nn.LayerNorm(dim),
#             nn.GELU(),
#             nn.Dropout(0.3)  # CHANGED: 0.3
#         )
        
#     def forward(self, x):
#         """
#         x: [B, N, D] features from BEATs
#         """
#         B, N, D = x.shape
#         scale_features = []
        
#         for scale, attn_pool in zip(self.scales, self.attention_pools):
#             if scale > 1:
#                 # Average pooling for different temporal scales
#                 x_scaled = x.view(B, N // scale, scale, D).mean(dim=2)  # [B, N/scale, D]
#             else:
#                 x_scaled = x
            
#             # Attention-based pooling
#             attn_weights = attn_pool(x_scaled)  # [B, N/scale, 1]
#             attn_weights = F.softmax(attn_weights, dim=1)
#             attn_weights = F.dropout(attn_weights, p=0.2, training=self.training)  # ADDED
#             pooled = torch.sum(x_scaled * attn_weights, dim=1)  # [B, D]
            
#             scale_features.append(pooled)
        
#         # Fuse multi-scale features
#         fused = torch.cat(scale_features, dim=-1)  # [B, 3*D]
#         fused = self.fusion(fused)  # [B, D]
        
#         return fused


# class HybridHFTT(nn.Module):
#     """
#     Regularized Hybrid HFTT
#     """
#     def __init__(
#         self,
#         num_target_classes: int = 4,
#         beats_model_path: str = "./pretrained_models/BEATs_iter3_plus_AS2M.pt",
#         freeze_encoder: bool = False,
#         use_multi_scale: bool = True,
#         spec_transform=None,
#     ):
#         super().__init__()
        
#         # 1. Load pretrained BEATs encoder
#         checkpoint = torch.load(beats_model_path, map_location='cpu')
#         cfg = BEATsConfig({
#             **checkpoint["cfg"],
#             "predictor_class": num_target_classes,
#             "finetuned_model": False,
#             "spec_transform": spec_transform
#         })
        
#         self.beats = BEATs(cfg)
#         self.beats.load_state_dict(checkpoint["model"])
        
#         # Freeze BEATs encoder if needed
#         if freeze_encoder:
#             for param in self.beats.parameters():
#                 param.requires_grad = False
        
#         self.encoder_dim = cfg.encoder_embed_dim  # Usually 768
#         self.final_feat_dim = self.encoder_dim
        
#         # 2. HFTT improvements
#         self.use_multi_scale = use_multi_scale
        
#         if use_multi_scale:
#             # Frequency-temporal attention
#             self.freq_temp_attn = FrequencyTemporalAttention(
#                 dim=self.encoder_dim,
#                 num_heads=12,
#                 dropout=0.3  # CHANGED: 0.3
#             )
            
#             # Multi-scale temporal pooling
#             self.multi_scale_pool = MultiScaleTemporalPooling(
#                 dim=self.encoder_dim,
#                 scales=[1, 2, 4]
#             )
#         else:
#             # Simple attention pooling (baseline)
#             self.attention_pool = nn.Sequential(
#                 nn.Linear(self.encoder_dim, self.encoder_dim // 4),
#                 nn.Tanh(),
#                 nn.Linear(self.encoder_dim // 4, 1)
#             )
    
#     def forward(self, source, padding_mask=None, training=False):
#         """
#         source: [B, T] raw audio waveform
#         Returns: [B, 1, D] or [B, D] features
#         """
#         # BEATs encoder
#         if padding_mask is not None:
#             x, _ = self.beats.extract_features(source, padding_mask)
#         else:
#             x, _ = self.beats.extract_features(source)
        
#         # x: [B, N, D] where N is number of temporal frames
        
#         if self.use_multi_scale:
#             # Apply frequency-temporal attention
#             x_attn = self.freq_temp_attn(x)  # [B, N, D]
#             x = x + x_attn  # Residual
#             x = F.dropout(x, p=0.1, training=self.training)  # ADDED: Dropout on residual
            
#             # Multi-scale temporal pooling
#             x_pooled = self.multi_scale_pool(x)  # [B, D]
            
#             # Expand for compatibility with PAFA
#             x_pooled = x_pooled.unsqueeze(1)  # [B, 1, D]
            
#             return x_pooled
#         else:
#             # Simple pooling (for comparison)
#             attn_weights = self.attention_pool(x)  # [B, N, 1]
#             attn_weights = F.softmax(attn_weights, dim=1)
#             x_pooled = torch.sum(x * attn_weights, dim=1)  # [B, D]
            
#             return x_pooled.unsqueeze(1)  # [B, 1, D]


# # Factory function
# def get_hftt_model(
#     beats_model_path="./pretrained_models/BEATs_iter3_plus_AS2M.pt",
#     freeze_encoder=False,
#     spec_transform=None
# ):
#     """
#     Create Regularized Hybrid HFTT model
#     """
#     model = HybridHFTT(
#         num_target_classes=4,
#         beats_model_path=beats_model_path,
#         freeze_encoder=freeze_encoder,
#         use_multi_scale=True,
#         spec_transform=spec_transform
#     )
#     return model