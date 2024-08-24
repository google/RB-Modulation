# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math

from modules.common import FeedForwardBlock, LayerNorm2d, ResBlock, TimestepBlock
from modules.controlnet import ControlNetDeliverer
from modules.stage_c import UpDownBlock2d
import numpy as np
import torch
import torch.nn as nn
from utils import AttnBlock


class StageCRBM(nn.Module):
  """
  RB-modulated StageC model in StableCascade.
  Modified from https://github.com/Stability-AI/StableCascade/blob/master/modules/stage_c.py
  """

  def __init__(
      self,
      c_in: int = 16,
      c_out: int = 16,
      c_r: int = 64,
      patch_size: int = 1,
      c_cond: int = 2048,
      c_hidden: list[int] = [2048, 2048],
      nhead: list[int] = [32, 32],
      blocks: list[list[int]] = [[8, 24], [24, 8]],
      block_repeat: list[list[int]] = [[1, 1], [1, 1]],
      level_config: list[str] = ["CTA", "CTA"],
      c_clip_text: int = 1280,
      c_clip_text_pooled: int = 1280,
      c_clip_img: int = 768,
      c_clip_seq: int = 4,
      kernel_size: int = 3,
      dropout: list[float] = [0.1, 0.1],
      self_attn: bool = True,
      t_conds: list[str] = ["sca", "crp"],
      switch_level: list[bool] = [False],
  ):
    """Create a StageCRBM model.

    Args:
      c_in: Number of input channels.
      c_out: Number of output channels.
      c_r: Dimensionality for the positional embedding.
      patch_size: Size of the patch for pixel unshuffling.
      c_cond: Dimensionality for the conditioning embeddings.
      c_hidden: List of hidden dimensions for each level.
      nhead: List of number of attention heads for each level.
      blocks: Number of blocks for each level.
      block_repeat: Number of block repeats for each level.
      level_config: Configuration of blocks for each level.
      c_clip_text: Dimensionality of the clip text embedding.
      c_clip_text_pooled: Dimensionality of the pooled clip text embedding.
      c_clip_img: Dimensionality of the clip image embedding.
      c_clip_seq: Sequence length for clip embeddings.
      kernel_size: Kernel size for convolution operations.
      dropout: Dropout rates for each level.
      self_attn: Whether to use self-attention in the attention block.
      t_conds: List of timestep conditions.
      switch_level: Whether to switch the level during upsampling.
    """
    super().__init__()
    self.c_r = c_r
    self.t_conds = t_conds
    self.c_clip_seq = c_clip_seq
    if not isinstance(dropout, list):
      dropout = [dropout] * len(c_hidden)
    if not isinstance(self_attn, list):
      self_attn = [self_attn] * len(c_hidden)

    # CONDITIONING
    self.clip_txt_mapper = nn.Linear(c_clip_text, c_cond)
    self.clip_txt_pooled_mapper = nn.Linear(
        c_clip_text_pooled, c_cond * c_clip_seq
    )
    self.clip_img_mapper = nn.Linear(c_clip_img, c_cond * c_clip_seq)
    self.clip_norm = nn.LayerNorm(c_cond, elementwise_affine=False, eps=1e-6)

    self.embedding = nn.Sequential(
        nn.PixelUnshuffle(patch_size),
        nn.Conv2d(c_in * (patch_size**2), c_hidden[0], kernel_size=1),
        LayerNorm2d(c_hidden[0], elementwise_affine=False, eps=1e-6),
    )

    def get_block(
        block_type: str,
        c_hidden: int,
        nhead: int,
        c_skip: int = 0,
        dropout: float = 0,
        self_attn: bool = True,
    ) -> nn.Module:
      """Returns a block of the specified type.

      Args:
          block_type: Type of the block ('C', 'A', 'F', 'T').
          c_hidden: Number of hidden channels.
          nhead: Number of attention heads.
          c_skip: Number of skip channels.
          dropout: Dropout rate.
          self_attn: Whether to use self-attention.

      Returns:
          Instantiated block module.

      Raises:
          ValueError: If the block type is not supported.
      """
      if block_type == "C":
        return ResBlock(
            c_hidden, c_skip, kernel_size=kernel_size, dropout=dropout
        )
      elif block_type == "A":
        return AttnBlock(
            c_hidden, c_cond, nhead, self_attn=self_attn, dropout=dropout
        )
      elif block_type == "F":
        return FeedForwardBlock(c_hidden, dropout=dropout)
      elif block_type == "T":
        return TimestepBlock(c_hidden, c_r, conds=t_conds)
      else:
        raise ValueError(f"Block type {block_type} not supported")

    # BLOCKS
    # -- down blocks
    self.down_blocks = nn.ModuleList()
    self.down_downscalers = nn.ModuleList()
    self.down_repeat_mappers = nn.ModuleList()
    for i in range(len(c_hidden)):
      if i > 0:
        self.down_downscalers.append(
            nn.Sequential(
                LayerNorm2d(
                    c_hidden[i - 1], elementwise_affine=False, eps=1e-6
                ),
                UpDownBlock2d(
                    c_hidden[i - 1],
                    c_hidden[i],
                    mode="down",
                    enabled=switch_level[i - 1],
                ),
            )
        )
      else:
        self.down_downscalers.append(nn.Identity())
      down_block = nn.ModuleList()
      for _ in range(blocks[0][i]):
        for block_type in level_config[i]:
          block = get_block(
              block_type,
              c_hidden[i],
              nhead[i],
              dropout=dropout[i],
              self_attn=self_attn[i],
          )
          down_block.append(block)
      self.down_blocks.append(down_block)
      if block_repeat is not None:
        block_repeat_mappers = nn.ModuleList()
        for _ in range(block_repeat[0][i] - 1):
          block_repeat_mappers.append(
              nn.Conv2d(c_hidden[i], c_hidden[i], kernel_size=1)
          )
        self.down_repeat_mappers.append(block_repeat_mappers)

    # -- up blocks
    self.up_blocks = nn.ModuleList()
    self.up_upscalers = nn.ModuleList()
    self.up_repeat_mappers = nn.ModuleList()
    for i in reversed(range(len(c_hidden))):
      if i > 0:
        self.up_upscalers.append(
            nn.Sequential(
                LayerNorm2d(c_hidden[i], elementwise_affine=False, eps=1e-6),
                UpDownBlock2d(
                    c_hidden[i],
                    c_hidden[i - 1],
                    mode="up",
                    enabled=switch_level[i - 1],
                ),
            )
        )
      else:
        self.up_upscalers.append(nn.Identity())
      up_block = nn.ModuleList()
      for j in range(blocks[1][::-1][i]):
        for k, block_type in enumerate(level_config[i]):
          c_skip = c_hidden[i] if i < len(c_hidden) - 1 and j == k == 0 else 0
          block = get_block(
              block_type,
              c_hidden[i],
              nhead[i],
              c_skip=c_skip,
              dropout=dropout[i],
              self_attn=self_attn[i],
          )
          up_block.append(block)
      self.up_blocks.append(up_block)
      if block_repeat is not None:
        block_repeat_mappers = nn.ModuleList()
        for _ in range(block_repeat[1][::-1][i] - 1):
          block_repeat_mappers.append(
              nn.Conv2d(c_hidden[i], c_hidden[i], kernel_size=1)
          )
        self.up_repeat_mappers.append(block_repeat_mappers)

    # OUTPUT
    self.clf = nn.Sequential(
        LayerNorm2d(c_hidden[0], elementwise_affine=False, eps=1e-6),
        nn.Conv2d(c_hidden[0], c_out * (patch_size**2), kernel_size=1),
        nn.PixelShuffle(patch_size),
    )

    # --- WEIGHT INIT ---
    self.apply(self._init_weights)  # General init
    nn.init.normal_(self.clip_txt_mapper.weight, std=0.02)  # conditionings
    nn.init.normal_(
        self.clip_txt_pooled_mapper.weight, std=0.02
    )  # conditionings
    nn.init.normal_(self.clip_img_mapper.weight, std=0.02)  # conditionings
    torch.nn.init.xavier_uniform_(self.embedding[1].weight, 0.02)  # inputs
    nn.init.constant_(self.clf[1].weight, 0)  # outputs

    # blocks
    for level_block in self.down_blocks + self.up_blocks:
      for block in level_block:
        if isinstance(block, ResBlock) or isinstance(block, FeedForwardBlock):
          block.channelwise[-1].weight.data *= np.sqrt(1 / sum(blocks[0]))
        elif isinstance(block, TimestepBlock):
          for layer in block.modules():
            if isinstance(layer, nn.Linear):
              nn.init.constant_(layer.weight, 0)

  def _init_weights(self, m: nn.Module) -> None:
    """Initializes the weights of the module.

    Args:
        m: The module to initialize.
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
      torch.nn.init.xavier_uniform_(m.weight)
      if m.bias is not None:
        nn.init.constant_(m.bias, 0)

  def gen_r_embedding(
      self, r: torch.Tensor, max_positions: int = 10000
  ) -> torch.Tensor:
    """Generates a positional embedding for the given input tensor.

    Args:
        r: Input tensor for positional embedding.
        max_positions: Maximum number of positions.

    Returns:
        Generated positional embedding.
    """
    r = r * max_positions
    half_dim = self.c_r // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
    emb = r[:, None] * emb[None, :]
    emb = torch.cat([emb.sin(), emb.cos()], dim=1)
    if self.c_r % 2 == 1:  # zero pad
      emb = nn.functional.pad(emb, (0, 1), mode="constant")
    return emb

  def gen_c_embeddings(
      self,
      clip_txt: torch.Tensor,
      clip_txt_pooled: torch.Tensor,
      clip_img: torch.Tensor,
      clip_style: torch.Tensor = None,
  ) -> torch.Tensor:
    """Generates conditional embeddings for the input clip embeddings.

    Args:
        clip_txt: Clip text embedding.
        clip_txt_pooled: Pooled clip text embedding.
        clip_img: Clip image embedding.
        clip_style: Clip style embedding.

    Returns:
        Generated conditional embedding.
    """
    clip_txt = self.clip_txt_mapper(clip_txt)
    if len(clip_txt_pooled.shape) == 2:
      clip_txt_pool = clip_txt_pooled.unsqueeze(1)
    if len(clip_img.shape) == 2:
      clip_img = clip_img.unsqueeze(1)

    clip_txt_pool = self.clip_txt_pooled_mapper(clip_txt_pooled).view(
        clip_txt_pooled.size(0), clip_txt_pooled.size(1) * self.c_clip_seq, -1
    )
    clip_img = self.clip_img_mapper(clip_img).view(
        clip_img.size(0), clip_img.size(1) * self.c_clip_seq, -1
    )

    if clip_style is not None:
      if len(clip_style.shape) == 2:
        clip_style = clip_style.unsqueeze(1)
      clip_style = self.clip_img_mapper(clip_style).view(
          clip_style.size(0), clip_style.size(1) * self.c_clip_seq, -1
      )
      clip = torch.cat([clip_txt, clip_txt_pool, clip_img, clip_style], dim=1)
    else:
      clip = torch.cat([clip_txt, clip_txt_pool, clip_img], dim=1)
    clip = self.clip_norm(clip)
    return clip

  def _down_encode(
      self,
      x: torch.Tensor,
      r_embed: torch.Tensor,
      clip: torch.Tensor,
      cnet: nn.Module = None,
      style: bool = False,
      img_style: bool = False,
  ) -> list[torch.Tensor]:
    """Encodes the input tensor through the down-sampling blocks.

    Args:
        x: Input tensor.
        r_embed: Positional embedding.
        clip: Conditional embedding.
        cnet: Control network module.
        style: Style flag.
        img_style: Image style flag.

    Returns:
        List of outputs from each level.
    """
    level_outputs = []
    block_group = zip(
        self.down_blocks, self.down_downscalers, self.down_repeat_mappers
    )
    for down_block, downscaler, repmap in block_group:
      x = downscaler(x)
      for i in range(len(repmap) + 1):
        for block in down_block:
          if isinstance(block, ResBlock) or (
              hasattr(block, "_fsdp_wrapped_module")
              and isinstance(block._fsdp_wrapped_module, ResBlock)
          ):
            if cnet is not None:
              next_cnet = cnet()
              if next_cnet is not None:
                x = x + nn.functional.interpolate(
                    next_cnet,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
            x = block(x)
          elif isinstance(block, AttnBlock) or (
              hasattr(block, "_fsdp_wrapped_module")
              and isinstance(block._fsdp_wrapped_module, AttnBlock)
          ):
            x = block(x, clip, style, img_style)
          elif isinstance(block, TimestepBlock) or (
              hasattr(block, "_fsdp_wrapped_module")
              and isinstance(block._fsdp_wrapped_module, TimestepBlock)
          ):
            x = block(x, r_embed)
          else:
            x = block(x)
        if i < len(repmap):
          x = repmap[i](x)
      level_outputs.insert(0, x)
    return level_outputs

  def _up_decode(
      self,
      level_outputs: list[torch.Tensor],
      r_embed: torch.Tensor,
      clip: torch.Tensor,
      cnet: nn.Module = None,
      style: bool = False,
      img_style: bool = False,
  ) -> torch.Tensor:
    """Decodes the input tensor through the up-sampling blocks.

    Args:
        level_outputs: List of outputs from down-sampling blocks.
        r_embed: Positional embedding.
        clip: Conditional embedding.
        cnet: Control network module.
        style: Style flag.
        img_style: Image style flag.

    Returns:
        Final output tensor after up-sampling.
    """
    x = level_outputs[0]
    block_group = zip(self.up_blocks, self.up_upscalers, self.up_repeat_mappers)
    for i, (up_block, upscaler, repmap) in enumerate(block_group):
      for j in range(len(repmap) + 1):
        for k, block in enumerate(up_block):
          if isinstance(block, ResBlock) or (
              hasattr(block, "_fsdp_wrapped_module")
              and isinstance(block._fsdp_wrapped_module, ResBlock)
          ):
            skip = level_outputs[i] if k == 0 and i > 0 else None
            if skip is not None and (
                x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2)
            ):
              x = torch.nn.functional.interpolate(
                  x.float(),
                  skip.shape[-2:],
                  mode="bilinear",
                  align_corners=True,
              )
            if cnet is not None:
              next_cnet = cnet()
              if next_cnet is not None:
                x = x + nn.functional.interpolate(
                    next_cnet,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
            x = block(x, skip)
          elif isinstance(block, AttnBlock) or (
              hasattr(block, "_fsdp_wrapped_module")
              and isinstance(block._fsdp_wrapped_module, AttnBlock)
          ):
            x = block(x, clip, style, img_style)
          elif isinstance(block, TimestepBlock) or (
              hasattr(block, "_fsdp_wrapped_module")
              and isinstance(block._fsdp_wrapped_module, TimestepBlock)
          ):
            x = block(x, r_embed)
          else:
            x = block(x)
        if j < len(repmap):
          x = repmap[j](x)
      x = upscaler(x)
    return x

  def forward(
      self,
      x: torch.Tensor,
      r: torch.Tensor,
      clip_text: torch.Tensor,
      clip_text_pooled: torch.Tensor,
      clip_img: torch.Tensor = None,
      clip_style: torch.Tensor = None,
      clip_img_style: torch.Tensor = None,
      cnet: nn.Module = None,
      style: bool = False,
      img_style: bool = False,
      **kwargs,
  ) -> torch.Tensor:
    """Forward pass of the StageCRBM module.

    Args:
        x: Input tensor.
        r: Input tensor for positional embedding.
        clip_text: Clip text embedding.
        clip_text_pooled: Pooled clip text embedding.
        clip_img: Clip content image embedding.
        clip_style: Clip style embedding.
        clip_img_style: Clip image style embedding.
        cnet: Control network module.
        style: Style flag.
        img_style: Image style flag.
        **kwargs: Additional keyword arguments for timestep conditions.

    Returns:
        Output tensor.
    """
    # Process the conditioning embeddings.
    r_embed = self.gen_r_embedding(r)
    for c in self.t_conds:
      t_cond = kwargs.get(c, torch.zeros_like(r))
      r_embed = torch.cat([r_embed, self.gen_r_embedding(t_cond)], dim=1)

    if clip_style is not None:
      clip = self.gen_c_embeddings(clip_text, clip_text_pooled, clip_style)
      style = True
    elif clip_img_style is not None:
      clip = self.gen_c_embeddings(
          clip_text, clip_text_pooled, clip_img, clip_img_style
      )
      img_style = True
    else:
      clip = self.gen_c_embeddings(clip_text, clip_text_pooled, clip_img)

    # Model Blocks.
    x = self.embedding(x)
    if cnet is not None:
      cnet = ControlNetDeliverer(cnet)
    level_outputs = self._down_encode(
        x, r_embed, clip, cnet, style=style, img_style=img_style
    )
    x = self._up_decode(
        level_outputs, r_embed, clip, cnet, style=style, img_style=img_style
    )
    return self.clf(x)
