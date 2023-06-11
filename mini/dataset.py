import logging
from glob import glob
from math import sqrt
from pathlib import Path
from typing import Tuple, NamedTuple, List

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
  def __init__(
    self,
    size: Tuple[int, int],
    instance_images_dir: str,
    tokenizer: CLIPTokenizer,
    aug_color_jitter: bool,
    aug_horizontal_flip: bool,
    aug_random_crop: bool,
    aug_random_crop_cover_percent: float
  ):
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
    self.transform = build_image_transforms(size, aug_color_jitter, aug_horizontal_flip, aug_random_crop, aug_random_crop_cover_percent)

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
  """
  Note: This dataset will not generate all possible pairs between instance and regularization images.
  Instead, one image is chosen for each regularization image. This is to prevent duplicate regularization images.
  """
  def __init__(
    self,
    instance_images_dir: str,
    reg_images_dir: str,
    size: Tuple[int, int],
    aug_color_jitter: bool,
    aug_horizontal_flip: bool,
    aug_random_crop: bool,
    aug_random_crop_cover_percent: float
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
    self.instance_transform = build_image_transforms(
      size,
      aug_color_jitter,
      aug_horizontal_flip,
      aug_random_crop,
      aug_random_crop_cover_percent
    )
    self.regularization_transform = build_image_transforms(size, False, False, False, 1.0)

  def __len__(self) -> int:
    return len(self.reg_image_names)

  def __getitem__(self, index: int) -> DreamboothExample:
    instance_idx = index % len(self.instance_image_names)
    reg_idx = index
    instance_image_name = self.instance_image_names[instance_idx]
    reg_image_name = self.reg_image_names[reg_idx]
    instance_image_pixels = open_and_transform(instance_image_name, self.instance_transform)
    reg_image_pixels = open_and_transform(reg_image_name, self.regularization_transform)
    return DreamboothExample(instance_image_pixels, reg_image_pixels)


def open_and_transform(image_name: str, _transforms: transforms.Compose) -> Tensor:
  image = Image.open(image_name)
  if image.mode != "RGB":
    image = image.convert("RGB")
  return _transforms(image)


def find_image_names(image_dir: Path) -> List[str]:
  image_names = glob(f"{image_dir}/*.jpg") +\
                glob(f"{image_dir}/*.png") +\
                glob(f"{image_dir}/*.jpeg")
  return image_names


def build_image_transforms(
  size: Tuple[int, int],
  aug_color_jitter: bool,
  aug_horizontal_flip: bool,
  aug_random_crop: bool,
  aug_random_crop_cover_percent: float
) -> transforms.Compose:
  def _resize_to_max_square_and_random_crop(image: Image.Image) -> Image.Image:
    min_size = min(image.size)
    square = transforms.CenterCrop(min_size)(image)
    if aug_random_crop:
      scale = sqrt(aug_random_crop_cover_percent)
      scaled_size = int(min_size * scale)
      return transforms.RandomCrop(scaled_size)(square)
    else:
      return square

  if aug_color_jitter:
    color_jitter = transforms.ColorJitter(0.05, 0.05)
  else:
    color_jitter = transforms.Lambda(lambda x: x)

  if aug_horizontal_flip:
    horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
  else:
    horizontal_flip = transforms.Lambda(lambda x: x)

  return transforms.Compose([
    horizontal_flip,
    color_jitter,
    transforms.Lambda(_resize_to_max_square_and_random_crop),
    transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
  ])
