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


from CSD.model import CSD_CLIP
from CSD.utils import convert_state_dict
from modules.common import LayerNorm2d, Linear
import numpy as np
import torch
import torch.nn as nn
from train import WurstCoreC


def setup_csd(device: str = "cpu") -> nn.Module:
  """Sets up the CSD model.

  Args:
      device: The device to load the model onto.

  Returns:
      The initialized CSD model.
  """
  model_path = "third_party/CSD/checkpoint.pth"
  model = CSD_CLIP("vit_large", "default")
  checkpoint = torch.load(model_path, map_location=device)
  state_dict = convert_state_dict(checkpoint["model_state_dict"])
  print("CSD model loading ...")
  msg = model.load_state_dict(state_dict, strict=True)
  print(msg)
  model.eval()
  return model


class WurstCoreCRBM(WurstCoreC):

  def extract_conditions(
      self,
      batch: dict,
      models: WurstCoreC.Models,
      extras: WurstCoreC.Extras,
      is_eval: bool = False,
      is_unconditional: bool = False,
      eval_image_embeds: bool = False,
      eval_style: bool = False,
      eval_style_pooled: bool = False,
      eval_subject_style: bool = False,
      eval_csd: bool = False,
      return_fields: list[str] = None,
  ) -> dict:
    """Extracts conditions from the input batch.

    Args:
        batch: Input batch of data.
        models: Models for extraction.
        extras: Extra utilities for preprocessing.
        is_eval: Flag for evaluation mode.
        is_unconditional: Flag for unconditional generation.
        eval_image_embeds: Flag for evaluating image embeddings.
        eval_style: Flag for evaluating style embeddings.
        eval_style_pooled: Flag for evaluating pooled style embeddings.
        eval_subject_style: Flag for evaluating subject style embeddings.
        eval_csd: Flag for evaluating CSD model.
        return_fields: list[str] - List of fields to return.

    Returns:
        Extracted conditions.
    """

    if return_fields is None:
      return_fields = ["clip_text", "clip_text_pooled", "clip_img"]

    captions = batch.get("captions", None)
    images = batch.get("images", None)
    style = batch.get("style", None)
    batch_size = len(captions)

    text_embeddings = None
    text_pooled_embeddings = None
    if "clip_text" in return_fields or "clip_text_pooled" in return_fields:
      if is_eval:
        if is_unconditional:
          captions_unpooled = ["" for _ in range(batch_size)]
        else:
          captions_unpooled = captions
      else:
        rand_idx = np.random.rand(batch_size) > 0.05
        captions_unpooled = [
            str(c) if keep else "" for c, keep in zip(captions, rand_idx)
        ]
      clip_tokens_unpooled = models.tokenizer(
          captions_unpooled,
          truncation=True,
          padding="max_length",
          max_length=models.tokenizer.model_max_length,
          return_tensors="pt",
      ).to(self.device)
      text_encoder_output = models.text_model(
          **clip_tokens_unpooled, output_hidden_states=True
      )
      if "clip_text" in return_fields:
        text_embeddings = text_encoder_output.hidden_states[-1]
      if "clip_text_pooled" in return_fields:
        text_pooled_embeddings = text_encoder_output.text_embeds.unsqueeze(1)

    return_fields_dict = {
        "clip_text": text_embeddings,
        "clip_text_pooled": text_pooled_embeddings,
    }

    style_embeddings = None
    if "clip_style" in return_fields:
      style_embeddings = torch.zeros(batch_size, 768, device=self.device)
      if style is not None:
        style = style.to(self.device)
        if is_eval:
          if not is_unconditional and eval_image_embeds and eval_style:
            if eval_csd:
              # AFA w/ CSD is more efficient.
              csd_model = setup_csd(device=self.device)
              bb_feats1, content_embeddings, style_embeddings = csd_model(
                  extras.clip_preprocess(style)
              )
            else:
              style_embeddings = models.image_model(
                  extras.clip_preprocess(style)
              ).image_embeds
        else:
          rand_idx = np.random.rand(batch_size) > 0.9
          if any(rand_idx):
            style_embeddings[rand_idx] = models.image_model(
                extras.clip_preprocess(style[rand_idx])
            ).image_embeds
      style_embeddings = style_embeddings.unsqueeze(1)
      return_fields_dict["clip_style"] = style_embeddings

      if "clip_style_pooled" in return_fields and eval_style_pooled:
        return_fields_dict["clip_style_pooled"] = torch.cat(
            [style_embeddings.mean(axis=0).unsqueeze(dim=0)] * batch_size, dim=0
        )
    else:
      image_embeddings = None
      image_embeddings = torch.zeros(batch_size, 768, device=self.device)
      if images is not None:
        images = images.to(self.device)
        if is_eval:
          if not is_unconditional and eval_image_embeds:
            image_embeddings = models.image_model(
                extras.clip_preprocess(images)
            ).image_embeds
        else:
          rand_idx = np.random.rand(batch_size) > 0.9
          if any(rand_idx):
            image_embeddings[rand_idx] = models.image_model(
                extras.clip_preprocess(images[rand_idx])
            ).image_embeds
      image_embeddings = image_embeddings.unsqueeze(1)
      return_fields_dict["clip_img"] = image_embeddings

      style_embeddings = None
      if "clip_img_style" in return_fields:
        style_embeddings = torch.zeros(batch_size, 768, device=self.device)
        if style is not None:
          style = style.to(self.device)
          if is_eval:
            if (
                not is_unconditional
                and eval_image_embeds
                and eval_subject_style
            ):
              if eval_csd:
                # AFA w/ csd.
                csd_model = setup_csd(device=self.device)
                bb_feats1, content_embeddings, style_embeddings = csd_model(
                    extras.clip_preprocess(style)
                )
              else:
                style_embeddings = models.image_model(
                    extras.clip_preprocess(style)
                ).image_embeds
          else:
            rand_idx = np.random.rand(batch_size) > 0.9
            if any(rand_idx):
              style_embeddings[rand_idx] = models.image_model(
                  extras.clip_preprocess(style[rand_idx])
              ).image_embeds
        style_embeddings = style_embeddings.unsqueeze(1)
        return_fields_dict["clip_img_style"] = style_embeddings
    return return_fields_dict

  def get_conditions(
      self,
      batch: dict,
      models: WurstCoreC.Models,
      extras: WurstCoreC.Extras,
      is_eval: bool = False,
      is_unconditional: bool = False,
      eval_image_embeds: bool = False,
      eval_style: bool = False,
      eval_style_pooled: bool = False,
      eval_subject_style: bool = False,
      eval_csd: bool = False,
      return_fields: list[str] = None,
  ) -> dict:
    """Retrieves conditions from the input batch.

    Args:
        batch: Input batch of data.
        models: Models for extraction.
        extras: Extra utilities for preprocessing.
        is_eval: Flag for evaluation mode.
        is_unconditional: Flag for unconditional generation.
        eval_image_embeds: Flag for evaluating image embeddings.
        eval_style: Flag for evaluating style embeddings.
        eval_style_pooled: Flag for evaluating pooled style embeddings.
        eval_subject_style: Flag for evaluating subject style embeddings.
        eval_csd: Flag for evaluating CSD model.
        return_fields: List of fields to return.

    Returns:
        Extracted conditions.
    """
    if eval_style:
      if eval_style_pooled:
        conditions = self.extract_conditions(
            batch,
            models,
            extras,
            is_eval,
            is_unconditional,
            eval_image_embeds,
            eval_style=eval_style,
            eval_style_pooled=eval_style_pooled,
            eval_csd=eval_csd,
            return_fields=return_fields
            or [
                "clip_text",
                "clip_text_pooled",
                "clip_style",
                "clip_style_pooled",
            ],
        )
      else:
        conditions = self.extract_conditions(
            batch,
            models,
            extras,
            is_eval,
            is_unconditional,
            eval_image_embeds,
            eval_style=eval_style,
            eval_csd=eval_csd,
            return_fields=return_fields
            or ["clip_text", "clip_text_pooled", "clip_style"],
        )
    elif eval_subject_style:
      conditions = self.extract_conditions(
          batch,
          models,
          extras,
          is_eval,
          is_unconditional,
          eval_image_embeds,
          eval_subject_style=eval_subject_style,
          eval_csd=eval_csd,
          return_fields=return_fields
          or ["clip_text", "clip_text_pooled", "clip_img", "clip_img_style"],
      )
    else:
      conditions = self.extract_conditions(
          batch,
          models,
          extras,
          is_eval,
          is_unconditional,
          eval_image_embeds,
          return_fields=return_fields
          or ["clip_text", "clip_text_pooled", "clip_img"],
      )

    return conditions


class Attention2D(nn.Module):
  """Attention2D module with Attention Feature Aggregation (AFA)."""

  def __init__(self, c: int, nhead: int, dropout: float = 0.0) -> None:
    """Creates the Attention2D module.

    Args:
        c: Number of channels.
        nhead: Number of attention heads.
        dropout: Dropout rate.
    """
    super().__init__()
    self.attn = nn.MultiheadAttention(
        c, nhead, dropout=dropout, bias=True, batch_first=True
    )

  def forward(
      self,
      x: torch.Tensor,
      kv: torch.Tensor,
      self_attn: bool = False,
      style: bool = False,
      img_style: bool = False,
      clip_size: int = 4,
  ) -> torch.Tensor:
    """Forward pass of the Attention2D module.

    Args:
        x: Input tensor.
        kv: Key-value tensor.
        self_attn: Flag for self-attention.
        style: Flag for style attention.
        img_style: Flag for content style attention.
        clip_size: Size of the clip.

    Returns:
        Output tensor.
    """
    att_map = None
    orig_shape = x.shape
    x = x.view(x.size(0), x.size(1), -1).permute(
        0, 2, 1
    )  # Bx4xHxW -> Bx(HxW)x4

    if self_attn:
      kv = torch.cat([x, kv], dim=1)
    if style:
      mean = kv[:, -clip_size:, :].mean(axis=1).unsqueeze(dim=1)
      ## for style only
      kv[:, -clip_size:, :] = torch.cat([mean] * (clip_size), dim=1)

      # KV for text only to better align with the prompt
      x_txt = self.attn(
          x, kv[:, :-clip_size, :], kv[:, :-clip_size, :], need_weights=False
      )[0]

      # KV for text+reference_style(img)
      x_txt_style = self.attn(x, kv, kv, need_weights=False)[0]

      # KV for reference_style(img)
      kv[:, -2 * clip_size : -clip_size, :] = kv[:, -clip_size:, :]
      x_style = self.attn(
          x, kv[:, :-clip_size, :], kv[:, :-clip_size, :], need_weights=False
      )[0]

      ## simple average
      x = (x_txt + x_txt_style + x_style) / 3

    elif img_style:
      mean = kv[:, -clip_size:, :].mean(axis=1).unsqueeze(dim=1)

      ## for txt, helps in extreme style transfer
      x_txt = self.attn(
          x,
          kv[:, : -2 * clip_size, :],
          kv[:, : -2 * clip_size, :],
          need_weights=False,
      )[0]

      ## for sub
      x_sub = self.attn(
          x, kv[:, :-clip_size, :], kv[:, :-clip_size, :], need_weights=False
      )[0]

      ## for sub_style
      kv[:, -clip_size:, :] = torch.cat([mean] * (clip_size), dim=1)
      x_sub_style, att_map = self.attn(x, kv, kv, need_weights=True)

      ## for style
      kv[:, -2 * clip_size : -clip_size, :] = torch.cat(
          [mean] * (clip_size), dim=1
      )
      x_style = self.attn(
          x, kv[:, :-clip_size, :], kv[:, :-clip_size, :], need_weights=False
      )[0]

      ## simple averaging
      x = (x_txt + x_sub + x_style + x_sub_style) / 4
    else:
      x = self.attn(x, kv, kv, need_weights=False)[0]
    x = x.permute(0, 2, 1).view(*orig_shape)
    return x


class AttnBlock(nn.Module):
  """Attention block with Attention Feature Aggregation (AFA)."""

  def __init__(
      self,
      c: int,
      c_cond: int,
      nhead: int,
      self_attn: bool = True,
      dropout: float = 0.0,
  ) -> None:
    """Initializes the AttnBlock module.

    Args:
        c: Number of channels.
        c_cond: Number of conditional channels.
        nhead: Number of attention heads.
        self_attn: Flag for self-attention.
        dropout: Dropout rate.
    """
    super().__init__()
    self.self_attn = self_attn
    self.norm = LayerNorm2d(c, elementwise_affine=False, eps=1e-6)
    self.attention = Attention2D(c, nhead, dropout)
    self.kv_mapper = nn.Sequential(nn.SiLU(), Linear(c_cond, c))

  def forward(
      self,
      x: torch.Tensor,
      kv: torch.Tensor,
      style: bool = False,
      img_style: bool = False,
  ) -> torch.Tensor:
    """Forward pass of the AttnBlock module.

    Args:
        x: Input tensor.
        kv: Key-value tensor.
        style: Flag for style attention.
        img_style: Flag for content style attention.

    Returns:
        Output tensor.
    """
    kv = self.kv_mapper(kv)
    x_out = self.attention(
        self.norm(x),
        kv,
        self_attn=self.self_attn,
        style=style,
        img_style=img_style,
    )
    x = x + x_out
    return x
