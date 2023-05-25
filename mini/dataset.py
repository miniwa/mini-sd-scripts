import logging
from glob import glob
from pathlib import Path
from typing import Tuple, NamedTuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer

from mini.asserts import Asserts


class Example(NamedTuple):
  instance_image_pixels: Tensor
  instance_image_caption_ids: Tensor


class ImageNameWithCaption(NamedTuple):
  image_name: str
  caption: str


class LoraDataset(Dataset[Example]):
  def __init__(self, size: Tuple[int, int], instance_images_dir: str, tokenizer: CLIPTokenizer):
    self.tokenizer = tokenizer
    self.size = size

    Asserts.check(size == (512, 512), f"Only {(512, 512)} is supported.")
    _instance_images_dir = Path(instance_images_dir)
    Asserts.check(_instance_images_dir.exists(), f"Instance images directory  '{instance_images_dir}' does not exist.")

    source_image_names = glob(f"{_instance_images_dir}/*.jpg") + glob(f"{_instance_images_dir}/*.png") + glob(
      f"{_instance_images_dir}/*.jpeg")
    caption_names = glob(f"{_instance_images_dir}/*.txt")
    Asserts.check(len(source_image_names) > 0, f"No source images found in '{instance_images_dir}'.")
    Asserts.check(len(source_image_names) == len(caption_names), f"Number of source images and captions do not match.")
    logging.info("Found %d source images.", len(source_image_names))

    self.image_names_with_captions = []
    for source_image_name in source_image_names:
      _image_path = Path(source_image_name)
      caption_name = str(_image_path.parent / f"{_image_path.stem}.txt")
      caption = Path(caption_name).read_text().strip()
      _parsed = ImageNameWithCaption(source_image_name, caption)
      logging.debug("Parsed %s", _parsed)
      self.image_names_with_captions.append(_parsed)

    # Transforms.
    self.resize_and_normalize = transforms.Compose([
      transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

  def __len__(self) -> int:
    return len(self.image_names_with_captions)

  def __getitem__(self, index: int) -> Example:
    instance = self.image_names_with_captions[index]
    logging.debug("Getting %s", instance)

    image = Image.open(instance.image_name)
    if image.mode != "RGB":
      image = image.convert("RGB")

    # Transforms.
    min_size = min(image.size)
    preprocess_transforms = transforms.Compose([
      transforms.CenterCrop(min_size),
      self.resize_and_normalize
    ])
    example_pixels = preprocess_transforms(image)
    example_caption_ids = self.tokenizer(
      instance.caption,
      padding="do_not_pad",
      truncation=True,
      max_length=self.tokenizer.model_max_length
    ).input_ids
    return Example(example_pixels, example_caption_ids)
