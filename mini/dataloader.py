from typing import List, NamedTuple

import torch
from diffusers import AutoencoderKL
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTokenizer

from mini.dataset import Example, LoraDataset


class DataLoaderBatch(NamedTuple):
  batch_image_pixels: Tensor
  batch_image_caption_ids: Tensor

  def pin_memory(self):
    self.batch_image_pixels = self.batch_image_pixels.pin_memory()
    self.batch_image_caption_ids = self.batch_image_caption_ids.pin_memory()
    return self


class Collater:
  def __init__(self, tokenizer: CLIPTokenizer):
    self.tokenizer = tokenizer

  def __call__(self, examples: List[Example]) -> DataLoaderBatch:
    image_pixels = [example.instance_image_pixels for example in examples]
    image_caption_ids = [example.instance_image_caption_ids for example in examples]
    batch_image_pixels = torch.stack(image_pixels).to(memory_format=torch.contiguous_format)
    batch_input_encodings = self.tokenizer.pad(
      {"input_ids": image_caption_ids},
      padding="max_length",
      max_length=self.tokenizer.model_max_length,
      return_tensors="pt"
    )
    return DataLoaderBatch(batch_image_pixels=batch_image_pixels,
                           batch_image_caption_ids=batch_input_encodings.input_ids)


def build_dataloader(
  dataset: LoraDataset,
  batch_size: int,
  tokenizer: CLIPTokenizer,
  cache_dataset: bool,
  accelerator_device: str
) -> DataLoader[DataLoaderBatch]:
  # def collate_fn(examples: List[Example]) -> DataLoaderBatch:
  #   image_pixels = [example.instance_image_pixels for example in examples]
  #   image_caption_ids = [example.instance_image_caption_ids for example in examples]
  #   batch_image_pixels = torch.stack(image_pixels).to(memory_format=torch.contiguous_format)
  #   batch_input_encodings = tokenizer.pad(
  #     {"input_ids": image_caption_ids},
  #     padding="max_length",
  #     max_length=tokenizer.model_max_length,
  #     return_tensors="pt"
  #   )
  #   return DataLoaderBatch(batch_image_pixels=batch_image_pixels,
  #                          batch_image_caption_ids=batch_input_encodings.input_ids)

  if cache_dataset:
    cached = []
    for example in dataset:
      cached.append(example)
    return DataLoader(
      cached,
      batch_size=batch_size,
      shuffle=True,
      collate_fn=Collater(tokenizer),
      pin_memory=True,
      num_workers=4,
      persistent_workers=True,
    )
  return DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=Collater(tokenizer),
    pin_memory=True,
    num_workers=4,
    persistent_workers=True
  )
