from itertools import chain

import fire
import torch
from accelerate import Accelerator
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from torch import optim
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import CLIPTextModel, get_scheduler

from mini.asserts import Asserts
from mini.dataloader import build_dataloader, DataLoaderBatch
from mini.dataset import LoraDataset
from mini.inject import KOHYA_UNET_INJECTION_TARGETS, \
  KOHYA_TEXT_ENCODER_INJECTION_TARGETS, inject_trainable_linear_lora
from mini.models import load_models, save_safetensors
from mini.rng import set_seed


def train(
  pretrained_model_name_or_path: str,
  pretrained_vae_name_or_path: str | None,
  dataset_instance_images_dir: str,
  output_dir: str,
  output_name: str,
  save_every_n_steps: int,
  train_max_steps: int,
  train_batch_size: int,
  train_learning_rate: float,
  train_unet_learning_rate: float,
  train_text_encoder_learning_rate: float,
  train_scale_lr_by_batch: bool,
  train_scheduler_name: str,
  train_scheduler_num_warmup_steps: int,
  optim_enable_gradient_checkpointing: bool,
  optim_train_fp16: bool,
  optim_gradient_accumulation_steps: int,
  lora_rank: int,
  seed: int | None = None,
):
  # Set seed as early as possible.
  if seed is not None:
    set_seed(seed)

  accelerator = Accelerator(mixed_precision="fp16" if optim_train_fp16 else None)
  print("Accelerator initialized with device:", accelerator.device)
  print("Loading models..")
  tokenizer, text_encoder, unet, vae, noise_scheduler = load_models(
    pretrained_model_name_or_path,
    pretrained_vae_name_or_path
  )
  print("Preparing dataset..")
  dataset = LoraDataset(size=(512, 512), instance_images_dir=dataset_instance_images_dir, tokenizer=tokenizer)
  dataloader = build_dataloader(
    dataset,
    batch_size=train_batch_size,
    tokenizer=tokenizer,
    cache_dataset=True,
    accelerator_device=accelerator.device
  )

  print("Injecting lora layers..")
  unet_params, unet_keys = inject_trainable_linear_lora(
    unet,
    KOHYA_UNET_INJECTION_TARGETS,
    lora_rank=lora_rank
  )
  te_params, te_keys = inject_trainable_linear_lora(
    text_encoder,
    KOHYA_TEXT_ENCODER_INJECTION_TARGETS,
    lora_rank=lora_rank
  )
  _all_trainable_params = chain(unet_params, te_params)
  print(f"Injected {len(unet_keys)} linear lora layers into UNet model.")
  print(f"Injected {len(te_keys)} linear lora layers into text encoder model.")

  # Put into train mode and freeze the weights that are not lora layers.
  unet.train()
  text_encoder.train()
  unet.requires_grad_(False)
  text_encoder.requires_grad_(False)
  vae.requires_grad_(False)

  # Only train the linear lora layers.
  for param in _all_trainable_params:
    param.requires_grad_(True)

  if optim_enable_gradient_checkpointing:
    unet.enable_gradient_checkpointing()
    text_encoder.gradient_checkpointing_enable()

  # Scale learning rate by batch size.
  if train_scale_lr_by_batch:
    train_learning_rate *= train_batch_size * optim_gradient_accumulation_steps
    train_unet_learning_rate *= train_batch_size * optim_gradient_accumulation_steps
    train_text_encoder_learning_rate *= train_batch_size * optim_gradient_accumulation_steps

  _params = [{
    "params": unet_params,
    "lr": train_unet_learning_rate,
  },
    {
      "params": te_params,
      "lr": train_text_encoder_learning_rate,
    }
  ]
  optimizer = optim.AdamW(params=_params, lr=train_learning_rate)
  lr_scheduler = get_scheduler(
    train_scheduler_name,
    optimizer=optimizer,
    num_warmup_steps=train_scheduler_num_warmup_steps,
    num_training_steps=train_max_steps
  )

  # Setup accelerator.
  print("Preparing models, dataset, optimizer and schedulers..")
  unet, text_encoder, vae, dataloader, optimizer, lr_scheduler = accelerator.prepare(
    unet,
    text_encoder,
    vae,
    dataloader,
    optimizer,
    lr_scheduler
  )

  print("Starting training..")
  progress_bar = tqdm(range(train_max_steps))
  progress_bar.set_description("Steps")
  step_count = 1
  sum_loss = torch.Tensor([0.0])
  sum_loss.requires_grad_(False)
  acc_steps = optim_gradient_accumulation_steps
  while step_count < train_max_steps:
    for batch in dataloader:
      loss = loss_step(
        batch=batch,
        unet=unet,
        text_encoder=text_encoder,
        vae=vae,
        noise_scheduler=noise_scheduler,
      )
      with torch.no_grad():
        sum_loss += loss.detach().to(sum_loss.device)
        mean_loss = sum_loss.item() / step_count

      loss = loss / acc_steps
      accelerator.backward(loss)
      clip_grad_norm_(_all_trainable_params, 1.0)
      should_update_weights = step_count % acc_steps == 0
      if should_update_weights:
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
      progress_bar.update(1)
      last_lr = lr_scheduler.get_last_lr()
      _logs = {
        "mean_loss": mean_loss,
        "unet_lr": last_lr[0],
        "te_lr": last_lr[1],
      }
      progress_bar.set_postfix(_logs)
      step_count += 1
      if step_count % save_every_n_steps == 0:
        _name = f"{output_name}-{step_count:05d}"
        _unwrapped_unet = accelerator.unwrap_model(unet)
        _unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
        save_safetensors(_unwrapped_unet, _unwrapped_text_encoder, output_dir, _name)
      if step_count >= train_max_steps:
        break

  # Save final model.
  _unwrapped_unet = accelerator.unwrap_model(unet)
  _unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
  save_safetensors(_unwrapped_unet, _unwrapped_text_encoder, output_dir, output_name)
  print("Training complete.")


def loss_step(
  batch: DataLoaderBatch,
  unet: UNet2DConditionModel,
  text_encoder: CLIPTextModel,
  vae: AutoencoderKL,
  noise_scheduler: DDPMScheduler,
):
  # Fairly certain this could be found in one of the model configs somewhere.
  LATENT_SCALE_FACTOR = 0.18215
  Asserts.check(noise_scheduler.config["prediction_type"] == "epsilon")

  with torch.no_grad():
    batch_size = batch.batch_image_pixels.shape[0]
    latents = vae.encode(batch.batch_image_pixels).latent_dist.sample()
    latents = latents * LATENT_SCALE_FACTOR

    # Latents with noise.
    noise = torch.randn_like(latents)
    max_training_timesteps = noise_scheduler.config["num_train_timesteps"]
    timesteps = torch.randint(0, max_training_timesteps, (batch_size,), device=unet.device)
    latents_with_noise = noise_scheduler.add_noise(latents, noise, timesteps)

  # Encoding of caption.
  encoder_hidden_states = text_encoder(
    batch.batch_image_caption_ids.to(text_encoder.device)
  )[0]

  # UNet output.
  pred = unet(latents_with_noise, timesteps, encoder_hidden_states).sample
  target = noise
  loss = mse_loss(pred, target, reduction="mean")
  return loss


if __name__ == "__main__":
  fire.Fire(train)
