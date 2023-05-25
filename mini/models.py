from pathlib import Path
from typing import Tuple, List

import torch
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from safetensors.torch import save_file
from torch import nn
from transformers import CLIPTokenizer, CLIPTextModel

from mini.inject import search_module_for_class, KOHYA_UNET_INJECTION_TARGETS, KOHYA_TEXT_ENCODER_INJECTION_TARGETS
from mini.lora import InjectedLinearLora


def load_models(pretrained_model_name_or_path: str, pretrained_vae_name_or_path: str | None) -> Tuple[
  CLIPTokenizer,
  CLIPTextModel,
  UNet2DConditionModel,
  AutoencoderKL,
  DDPMScheduler
]:
  tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer"
  )
  text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="text_encoder",
  )
  unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="unet",
  )
  if pretrained_vae_name_or_path is not None:
    vae = AutoencoderKL.from_pretrained(
      pretrained_vae_name_or_path,
    )
  else:
    vae = AutoencoderKL.from_pretrained(
      pretrained_model_name_or_path,
      subfolder="vae",
    )
  noise_scheduler = DDPMScheduler.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="scheduler"
  )

  return (
    tokenizer,
    text_encoder,
    unet,
    vae,
    noise_scheduler,
  )


def save_safetensors(
  unet: UNet2DConditionModel,
  text_encoder: CLIPTextModel,
  output_dir: str,
  output_name: str
):
  metadata = {}
  weights = {}

  def get_lora_weights_for_module(module: nn.Module, only_children_of: List[str], prefix: str):
    tmp_weights = {}
    for res in search_module_for_class(
      module,
      [InjectedLinearLora],
      only_children_of=only_children_of,
      exclude_children_of=None
    ):
      lora: InjectedLinearLora = res.module
      lora_name = f"{prefix}_{res.full_key}".replace(".", "_")
      up, down = lora.get_weights_as_tensor()
      tmp_weights[f"{lora_name}.lora_up.weight"] = up
      tmp_weights[f"{lora_name}.lora_down.weight"] = down
      tmp_weights[f"{lora_name}.alpha"] = torch.tensor(lora.rank)
    return tmp_weights

  tmp_unet = get_lora_weights_for_module(unet, KOHYA_UNET_INJECTION_TARGETS, "lora_unet")
  tmp_text_encoder = get_lora_weights_for_module(text_encoder, KOHYA_TEXT_ENCODER_INJECTION_TARGETS, "lora_te")
  weights.update(tmp_unet)
  weights.update(tmp_text_encoder)

  # Create output dir if it doesn't exist.
  Path(output_dir).mkdir(parents=True, exist_ok=True)
  output_path = f"{output_dir}/{output_name}.safetensors"
  print(f"Saving safetensors to {output_path}")
  save_file(weights, output_path, metadata=metadata)
