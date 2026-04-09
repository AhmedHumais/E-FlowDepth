import torch
import torch.nn as nn
# import torch.nn.functional as F

from mamba_ssm import Mamba
from .ssm_block import Block
from einops import rearrange
# from timm.models.layers import DropPath

class MambaBiDir(nn.Module):
    def __init__(self, dim, h=60, w=80, pos_enc = False):
        super().__init__()
        num_layers=2
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(Block(dim, Mamba))
        self.norm_f = nn.LayerNorm(dim)
        self.height = h
        self.width = w
        # self.sp_idcs = MambaBiDir.generate_spiral_indices(h, w)
        self.pos_enc = pos_enc
        if pos_enc:
            self.pos_embed = nn.Parameter(torch.zeros(1, h*w, 1))
    
    @staticmethod
    def generate_spiral_indices(H, W):
        """Generate spiral indices starting from the center outwards."""
        center = (H // 2 -1, W // 2 -1)
        visited = torch.zeros(H, W, dtype=torch.bool)
        spiral_order = []

        # Directions: right, down, left, up
        directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        dir_idx = 0

        y, x = center
        step_size = 1
        turns = 0

        while len(spiral_order) < H * W:
            for _ in range(step_size):
                if 0 <= y < H and 0 <= x < W and not visited[y, x]:
                    visited[y, x] = True
                    spiral_order.append((y, x))
                y += directions[dir_idx][0]
                x += directions[dir_idx][1]
            dir_idx = (dir_idx + 1) % 4
            turns += 1
            if turns % 2 == 0:
                step_size += 1

        spiral_order_tensor = torch.tensor(spiral_order, dtype=torch.long)
        return spiral_order_tensor  # shape (L, 2)

    def spiral_flatten(self, x, spiral_indices):
        """
        x: (B, D, H, W)
        spiral_indices: (L, 2) where L = H * W
        Returns: (B, L, D)
        """
        B, D, H, W = x.shape
        L = H * W
        assert (H,W) == (self.height, self.width)
        
        # Flatten spatial dimension and transpose to (B, H, W, D)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, D)
        x = x.reshape(B, H * W, D)  # still in raster scan order

        # Convert spiral_indices (y, x) to flat indices
        flat_indices = spiral_indices[:, 0] * W + spiral_indices[:, 1]  # (L,)
        
        # Apply the spiral index to get (B, L, D)
        x_spiral = x[:, flat_indices]  # indexing is differentiable
        return x_spiral

    def spiral_unflatten(self, x_seq, spiral_indices):
        """
        x_seq: (B, L, D)
        spiral_indices: (L, 2)
        Returns: (B, D, H, W)
        """
        H, W = self.height, self.width
        B, L, D = x_seq.shape
        device = x_seq.device

        # Convert spiral indices to flat indices
        flat_indices = spiral_indices[:, 0] * W + spiral_indices[:, 1]  # (L,)

        # Prepare an empty raster (B, H*W, D)
        x_raster = torch.zeros(B, H * W, D, device=device)

        # Expand flat indices to shape (B, L)
        # flat_indices_exp = flat_indices.unsqueeze(0).expand(B, -1)  # (B, L)

        # Use scatter_add for differentiability
        # x_raster = x_raster.scatter_add(1, flat_indices_exp.unsqueeze(-1).expand(-1, -1, D), x_seq)
        x_raster[:, flat_indices] = x_seq

        # Reshape back to (B, D, H, W)
        x_raster = x_raster.view(B, H, W, D).permute(0, 3, 1, 2)  # (B, D, H, W)
        return x_raster.contiguous()

    # def forward(self, x, residual=None):
    #     # B, C, H, W = x.shape
    #     x = self.spiral_flatten(x, self.sp_idcs)
    #     if self.pos_enc:
    #         x = x+self.pos_embed
            
    #     x, residual = self.blocks[0](x)

    #     # Reverse sequence
    #     x = x+residual
    #     x = x.flip(1)

    #     # Second Mamba block
    #     x, residual = self.blocks[1](x)

    #     # Reverse back
    #     residual = residual + x
    #     x = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
    #     # x = x.flip(1)        
    #     x = self.spiral_unflatten(x, self.sp_idcs.flip(0))
    #     # x  = rearrange(x, 'b (h w) c -> b c h w', h=H)
        
    #     return x
    @torch._dynamo.disable
    def forward(self, x, residual=None):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        if self.pos_enc:
            x = x+self.pos_embed

        # First Mamba block
        x, residual = self.blocks[0](x)

        # Reverse sequence
        x = x+residual
        x = x.flip(1)

        # Second Mamba block
        x, residual = self.blocks[1](x)

        # Reverse back
        residual = residual + x
        x = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        x = x.flip(1)        
        
        x  = rearrange(x, 'b (h w) c -> b c h w', h=H)
        return x