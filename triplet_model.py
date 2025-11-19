import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor


class GeM(nn.Module):
    def __init__(self, p=3.0, trainable=True, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p) if trainable else p
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1. / self.p)

    def __repr__(self):
        if isinstance(self.p, torch.nn.Parameter):
            return f"GeM(p={self.p.data.tolist()[0]:.4f}, trainable={self.p.requires_grad})"
        else:
            return f"GeM(p={self.p:.4f}, trainable=False)"

def interpolate_pos_embed(model, checkpoint_model):
    # Extract pos_embed from checkpoint
    if "pos_embed" not in checkpoint_model:
        return

    pos_embed_checkpoint = checkpoint_model["pos_embed"]
    embedding_size = pos_embed_checkpoint.shape[-1]  # typically 768
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches

    # Original shape
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    new_size = int(num_patches ** 0.5)

    if orig_size != new_size:
        print(f"Interpolating position embeddings from {orig_size}x{orig_size} to {new_size}x{new_size}")

        # Separate class token / register tokens if any
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]

        pos_tokens = pos_tokens.reshape(1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = F.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)

        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model["pos_embed"] = new_pos_embed