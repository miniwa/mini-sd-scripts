import torch
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from torch import Tensor
from torch.nn.functional import mse_loss
from transformers import CLIPTextModel

from mini.asserts import Asserts
from mini.dataloader import DataLoaderBatch, DreamboothDataLoaderBatch


def loss_step(
  batch: DataLoaderBatch | DreamboothDataLoaderBatch,
  unet: UNet2DConditionModel,
  text_encoder: CLIPTextModel,
  vae: AutoencoderKL,
  noise_scheduler: DDPMScheduler,
):
  # Fairly certain this could be found in one of the model configs somewhere.
  LATENT_SCALE_FACTOR = 0.18215
  Asserts.check(noise_scheduler.config["prediction_type"] == "epsilon")
  is_dreambooth = isinstance(batch, DreamboothDataLoaderBatch)
  if is_dreambooth:
    db_batch = Asserts.cast_to(batch, DreamboothDataLoaderBatch)
    batch_size = db_batch.batch_stacked_image_pixels.shape[0]
    image_pixels = db_batch.batch_stacked_image_pixels
    image_caption_ids = db_batch.batch_stacked_captions_ids
  else:
    normal_batch = Asserts.cast_to(batch, DataLoaderBatch)
    batch_size = normal_batch.batch_image_pixels.shape[0]
    image_pixels = normal_batch.batch_image_pixels
    image_caption_ids = normal_batch.batch_image_caption_ids

  with torch.no_grad():
    latents = vae.encode(image_pixels).latent_dist.sample()
    latents = latents * LATENT_SCALE_FACTOR

    # Latents with noise.
    noise = torch.randn_like(latents)
    max_training_timesteps = noise_scheduler.config["num_train_timesteps"]
    timesteps = torch.randint(0, max_training_timesteps, (batch_size,), device=unet.device)
    latents_with_noise = noise_scheduler.add_noise(latents, noise, timesteps)

  # Encoding of caption.
  encoder_hidden_states = text_encoder(
    image_caption_ids.to(text_encoder.device)
  )[0]

  # UNet output.
  pred: Tensor = unet(latents_with_noise, timesteps, encoder_hidden_states).sample
  target = noise
  if is_dreambooth:
    instance_pred, reg_pred = torch.chunk(pred, 2)
    instance_target, reg_target = torch.chunk(target, 2)

    instance_loss = mse_loss(instance_pred, instance_target, reduction="mean")
    reg_loss = mse_loss(reg_pred, reg_target, reduction="mean")
    loss = instance_loss + reg_loss
    return loss
  else:
    loss = mse_loss(pred, target, reduction="mean")
    return loss
