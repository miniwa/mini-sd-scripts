import logging
from glob import glob
from pathlib import Path
from typing import Tuple, NamedTuple, List

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


class DreamboothExample(NamedTuple):
  instance_image_pixels: Tensor
  reg_image_pixels: Tensor


class ImageNameWithCaption(NamedTuple):
  image_name: str
  caption: str


class LoraDataset(Dataset[Example]):
  def __init__(self, size: Tuple[int, int], instance_images_dir: str, tokenizer: CLIPTokenizer):
    self.tokenizer = tokenizer
    self.size = size

    Asserts.check(size == (512, 512), f"Only size {(512, 512)} is supported.")
    _instance_images_dir = Path(instance_images_dir)
    Asserts.check(_instance_images_dir.exists(), f"Instance images directory  '{instance_images_dir}' does not exist.")

    source_image_names = find_image_names(_instance_images_dir)
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
      self.image_names_with_captions.append(_parsed)

    self.transform = build_image_transforms(size)

  def __len__(self) -> int:
    return len(self.image_names_with_captions)

  def __getitem__(self, index: int) -> Example:
    instance = self.image_names_with_captions[index]
    example_pixels = open_and_transform(instance.image_name, self.transform)
    example_caption_ids = self.tokenizer(
      instance.caption,
      padding="do_not_pad",
      truncation=True,
      max_length=self.tokenizer.model_max_length
    ).input_ids
    return Example(example_pixels, example_caption_ids)


class DreamboothDataset(Dataset[DreamboothExample]):
  def __init__(
    self,
    instance_images_dir: str,
    reg_images_dir: str,
    size: Tuple[int, int]
  ):
    self.size = size
    Asserts.check(size == (512, 512), f"Only size {(512, 512)} is supported.")
    _instance_path = Path(instance_images_dir)
    _reg_image_path = Path(reg_images_dir)
    Asserts.check(_instance_path.exists(), f"Instance images directory  '{instance_images_dir}' does not exist.")
    Asserts.check(_reg_image_path.exists(), f"Regular images directory  '{reg_images_dir}' does not exist.")

    self.instance_image_names = find_image_names(_instance_path)
    self.reg_image_names = find_image_names(_reg_image_path)
    Asserts.check(len(self.instance_image_names) > 0, f"No source images found in '{instance_images_dir}'.")
    Asserts.check(len(self.reg_image_names) > 0, f"No regularization images found in '{reg_images_dir}'.")

    self._len = len(self.instance_image_names) * len(self.reg_image_names)
    self.image_transforms = build_image_transforms(size)

  def __len__(self) -> int:
    return self._len

  def __getitem__(self, index: int) -> DreamboothExample:
    instance_idx = index % len(self.instance_image_names)
    reg_idx = index // len(self.instance_image_names)
    instance_image_name = self.instance_image_names[instance_idx]
    reg_image_name = self.reg_image_names[reg_idx]
    instance_image_pixels = open_and_transform(instance_image_name, self.image_transforms)
    reg_image_pixels = open_and_transform(reg_image_name, self.image_transforms)
    return DreamboothExample(instance_image_pixels, reg_image_pixels)


def open_and_transform(image_name: str, _transforms: transforms.Compose) -> Tensor:
  image = Image.open(image_name)
  if image.mode != "RGB":
    image = image.convert("RGB")
  return _transforms(image)


def find_image_names(image_dir: Path) -> List[str]:
  image_names =  glob(f"{image_dir}/*.jpg") +\
                 glob(f"{image_dir}/*.png")+\
                 glob(f"{image_dir}/*.jpeg")
  return image_names


def build_image_transforms(size: Tuple[int, int]) -> transforms.Compose:
  def _resize_to_max_square(image: Image.Image) -> Image.Image:
    min_size = min(image.size)
    return transforms.CenterCrop(min_size)(image)

  return transforms.Compose([
    transforms.Lambda(_resize_to_max_square),
    transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
  ])
