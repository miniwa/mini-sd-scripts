from typing import List, NamedTuple

import torch
from diffusers import AutoencoderKL
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTokenizer

from mini.asserts import Asserts
from mini.dataset import Example, LoraDataset, DreamboothExample, DreamboothDataset


class DataLoaderBatch(NamedTuple):
  batch_image_pixels: Tensor
  batch_image_caption_ids: Tensor

  def pin_memory(self):
    self.batch_image_pixels = self.batch_image_pixels.pin_memory()
    self.batch_image_caption_ids = self.batch_image_caption_ids.pin_memory()
    return self


class DreamboothDataLoaderBatch(NamedTuple):
  batch_stacked_image_pixels: Tensor
  """
  Stacked image pixels of shape (batch_size * 2, 512, 512).
  The first half of the batch is the instance image pixels.
  The second half of the batch is the reg image pixels.
  """
  batch_stacked_captions_ids: Tensor
  """
  Stacked caption ids of shape (batch_size * 2, 77).
  The first half of the batch is the instance image caption ids.
  The second half of the batch is the reg image caption ids.
  """

  def pin_memory(self):
    self.batch_stacked_image_pixels = self.batch_stacked_image_pixels.pin_memory()
    self.batch_stacked_captions_ids = self.batch_stacked_captions_ids.pin_memory()
    return self


class Collator:
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


class DreamboothCollator:
  def __init__(self, tokenizer: CLIPTokenizer, instance_caption: str, reg_caption: str, batch_size: int):
    self.tokenizer = tokenizer
    self.batch_size = batch_size
    instance_ids: List[int] = tokenizer(
      instance_caption,
      padding="do_not_pad",
      truncation=True,
      max_length=self.tokenizer.model_max_length
    ).input_ids
    instance_ids_batched = [instance_ids for _ in range(batch_size)]
    reg_ids: List[int] = tokenizer(
      reg_caption,
      padding="do_not_pad",
      truncation=True,
      max_length=self.tokenizer.model_max_length
    ).input_ids
    reg_ids_batched = [reg_ids for _ in range(batch_size)]
    combined_ids = instance_ids_batched + reg_ids_batched

    # Will always be the same for any given instance and reg image.
    self.cached_batch_caption_ids: Tensor = tokenizer.pad(
      {"input_ids": combined_ids},
      padding="max_length",
      max_length=self.tokenizer.model_max_length,
      return_tensors="pt"
    ).input_ids

  def __call__(self, examples: List[DreamboothExample]) -> DreamboothDataLoaderBatch:
    instance_image_pixels = [example.instance_image_pixels for example in examples]
    reg_image_pixels = [example.reg_image_pixels for example in examples]
    combined_image_pixels = instance_image_pixels + reg_image_pixels

    batch_stacked_image_pixels = torch.stack(combined_image_pixels).to(memory_format=torch.contiguous_format)
    image_pixels_size = batch_stacked_image_pixels.shape[0]
    caption_ids_size = self.cached_batch_caption_ids.shape[0]
    Asserts.check(image_pixels_size == self.batch_size * 2, "Image pixels not aligned to batch size.")
    Asserts.check(caption_ids_size == self.batch_size * 2, "Caption ids not aligned to batch size.")
    return DreamboothDataLoaderBatch(
      batch_stacked_image_pixels=batch_stacked_image_pixels,
      batch_stacked_captions_ids=self.cached_batch_caption_ids
    )


def build_dataloader(
  dataset: LoraDataset,
  batch_size: int,
  tokenizer: CLIPTokenizer,
  cache_dataset: bool
) -> DataLoader[DataLoaderBatch]:
  if cache_dataset:
    cached = []
    for example in dataset:
      cached.append(example)
    return DataLoader(
      cached,
      batch_size=batch_size,
      shuffle=True,
      collate_fn=Collator(tokenizer),
      pin_memory=True,
      num_workers=4,
      persistent_workers=True,
    )
  return DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=Collator(tokenizer),
    pin_memory=True,
    num_workers=4,
    persistent_workers=True
  )


def build_dreambooth_dataloader(
  dataset: DreamboothDataset,
  batch_size: int,
  tokenizer: CLIPTokenizer,
  instance_caption: str,
  reg_caption: str,
):
  return DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=DreamboothCollator(tokenizer, instance_caption, reg_caption, batch_size),
    pin_memory=True,
    #num_workers=4,
    #persistent_workers=True
  )
