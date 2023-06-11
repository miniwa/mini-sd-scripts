from typing import List, NamedTuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer

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
  def __init__(self, tokenizer: CLIPTokenizer, instance_caption: str, reg_caption: str):
    self.tokenizer = tokenizer
    self.instance_tokens = tokenizer(
      instance_caption,
      padding="do_not_pad",
      truncation=True,
      max_length=self.tokenizer.model_max_length
    )

    self.reg_tokens = tokenizer(
      reg_caption,
      padding="do_not_pad",
      truncation=True,
      max_length=self.tokenizer.model_max_length
    )

  def __call__(self, examples: List[DreamboothExample]) -> DreamboothDataLoaderBatch:
    instance_image_pixels = [example.instance_image_pixels for example in examples]
    reg_image_pixels = [example.reg_image_pixels for example in examples]
    combined_image_pixels = instance_image_pixels + reg_image_pixels

    batch_stacked_image_pixels = torch.stack(combined_image_pixels).to(memory_format=torch.contiguous_format)
    nr_examples = len(examples)
    batch_caption_ids = self.captions_for(nr_examples)

    return DreamboothDataLoaderBatch(
      batch_stacked_image_pixels=batch_stacked_image_pixels,
      batch_stacked_captions_ids=batch_caption_ids
    )

  def captions_for(self, image_count: int) -> Tensor:
    instance_tokens_batched = [self.instance_tokens for _ in range(image_count)]
    reg_tokens_batched = [self.reg_tokens for _ in range(image_count)]
    combined_ids = instance_tokens_batched + reg_tokens_batched

    batch_tokens = self.tokenizer.pad(
      combined_ids,
      padding="max_length",
      max_length=self.tokenizer.model_max_length,
      return_tensors="pt"
    )
    return batch_tokens.input_ids


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
  cache_dataset: bool
) -> DataLoader[DreamboothDataLoaderBatch]:
  if cache_dataset:
    cached = []
    for example in dataset:
      cached.append(example)
    _dataset = cached
  else:
    _dataset = dataset

  return DataLoader(
    _dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=DreamboothCollator(tokenizer, instance_caption, reg_caption),
    pin_memory=True,
    num_workers=1,
    prefetch_factor=10,
    persistent_workers=True
  )
