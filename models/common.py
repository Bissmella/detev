# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from typing import Type


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
def load_checkpoint_SAMEncoder(model, filename, map_location="cpu", strict=False):
    checkpoint = torch.load(filename, map_location=map_location)

    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")
    # get state_dict from checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    if any([k.startswith("image_encoder.") for k in state_dict.keys()]):
        print("Remove encoder. prefix")
        state_dict = {
            k.replace("image_encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("image_encoder.")
        }
    
    if state_dict.get("absolute_pos_embed") is not None:
        absolute_pos_embed = state_dict["absolute_pos_embed"]
        N1, L, C1 = absolute_pos_embed.size()
        N2, C2, H, W = model.absolute_pos_embed.size()
        if N1 != N2 or C1 != C2 or L != H * W:
            print("Warning: Error in loading absolute_pos_embed, pass")
        else:
            state_dict["absolute_pos_embed"] = absolute_pos_embed.view(
                N2, H, W, C2
            ).permute(0, 3, 1, 2)

    # load state_dict
    msg = model.load_state_dict(state_dict, strict=strict)
    print(msg)
    return checkpoint