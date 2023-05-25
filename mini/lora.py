from typing import Tuple

import torch.nn as nn
from torch import Tensor

from mini.asserts import Asserts


class InjectedLinearLora(nn.Module):
  def __init__(self, nr_in_features: int, nr_out_features: int, bias: bool, rank: int, scale: float = 1.0):
    super().__init__()
    Asserts.check(rank <= min(nr_in_features, nr_out_features),
                  f"Rank must less than or equal to {min(nr_in_features, nr_out_features)}.")
    self.rank = rank
    self.scale = scale
    self.linear = nn.Linear(nr_in_features, nr_out_features, bias=bias)
    self.lora_down = nn.Linear(nr_in_features, rank, bias=False)
    self.lora_up = nn.Linear(rank, nr_out_features, bias=False)
    nn.init.normal_(self.lora_down.weight, std=1 / rank)
    nn.init.zeros_(self.lora_up.weight)

  def forward(self, _input: Tensor) -> Tensor:
    return self.linear(_input) + self.lora_up(self.lora_down(_input)) * self.scale

  def get_weights_as_tensor(self) -> Tuple[Tensor, Tensor]:
    return self.lora_up.weight.data, self.lora_down.weight.data
