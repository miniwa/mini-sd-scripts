from typing import List, Type, NamedTuple

from torch import nn

from mini.asserts import Asserts
from mini.lora import InjectedLinearLora

# Copied from https://github.com/cloneofsimo/lora
DEFAULT_UNET_INJECTION_TARGETS = ["CrossAttention", "Attention", "GEGLU"]
DEFAULT_TEXT_ENCODER_INJECTION_TARGETS = ["CLIPAttention"]

# Copied from kohya sd scripts
KOHYA_UNET_INJECTION_TARGETS = ["Transformer2DModel", "Attention"]
KOHYA_TEXT_ENCODER_INJECTION_TARGETS = ["CLIPAttention", "CLIPMLP"]
# These are the numbers of injections in kohya sd scripts.
# The number is different from the numbers I get with equal injection targets for unknown reasons.
# This may or may not be a problem.
KOHYA_NR_UNET_INJECTIONS = 192
KOHYA_NR_TEXT_ENCODER_INJECTIONS = 72


class SearchModuleForClassRes(NamedTuple):
  full_key: str
  parent_module: nn.Module
  module_name: str
  module: nn.Module


def search_module_for_class(_module: nn.Module, search_class: List[Type[nn.Module]],
                            only_children_of: list[str] | None,
                            exclude_children_of: list[Type[nn.Module]] | None):
  # Expect this to be true for all modules. Else might cause problems.
  _module_modules_len = len(list(_module.modules()))
  _module_named_modules_len = len(list(_module.named_modules()))
  Asserts.check(_module_modules_len == _module_named_modules_len, "Module and named_modules mismatch.")
  if only_children_of is not None:
    _ancestor_modules = [_name_module_pair for _name_module_pair in _module.named_modules() if
                         _name_module_pair[1].__class__.__name__ in only_children_of]
  else:
    _ancestor_modules = list(_module.named_modules())

  for _ancestor_name, _ancestor_module in _ancestor_modules:
    for name, child_module in _ancestor_module.named_modules():
      matches_search_class = any([isinstance(child_module, _class) for _class in search_class])
      if matches_search_class:
        *parent_names, child_name = name.split(".")
        parent = _ancestor_module
        while len(parent_names) > 0:
          parent = parent.get_submodule(parent_names.pop(0))
        # We check if the direct parent is an instance of any of the classes in exclude_children_of.
        if exclude_children_of is not None and any(
          [isinstance(parent, _class) for _class in exclude_children_of]):
          continue
        full_key = ".".join([_ancestor_name, name])
        yield SearchModuleForClassRes(full_key=full_key, parent_module=parent, module=child_module,
                                      module_name=child_name)


def inject_trainable_linear_lora(
  _module: nn.Module,
  injection_targets: List[str] | None,
  lora_rank: int,
  dropout_percent: float
):
  keys = []
  trainable_parameters = []
  # We search for all linear layers in the module, that is a child of any of the injection targets.
  # We also EXCLUDE any linear layer that is a direct child of InjectedLinearLora.
  # Otherwise, we would inject into the same layer over and over again.
  for result in search_module_for_class(
    _module,
    search_class=[nn.Linear],
    only_children_of=injection_targets,
    exclude_children_of=[InjectedLinearLora]
  ):
    full_key = result.full_key
    parent_module = result.parent_module
    name = result.module_name
    # noinspection PyTypeChecker
    module_to_replace: nn.Linear = result.module

    # Do not inject into layers that are already injected.
    Asserts.check(not isinstance(parent_module, InjectedLinearLora), f"Already injected into {full_key}.")

    old_weight = module_to_replace.weight
    old_bias = module_to_replace.bias
    _device = old_weight.device
    _dtype = old_weight.dtype

    lora = InjectedLinearLora(
      module_to_replace.in_features,
      module_to_replace.out_features,
      bias=old_bias is not None,
      rank=lora_rank,
      dropout_percent=dropout_percent
    )
    lora.linear.weight = old_weight
    if old_bias is not None:
      lora.linear.bias = old_bias

    # Convert to device and dtype of module_to_replace.
    lora.to(device=_device, dtype=_dtype)
    parent_module.add_module(name, lora)

    # We need to add the parameters of the lora module to the list of trainable parameters.
    keys.append(full_key)
    trainable_parameters.extend(lora.lora_up.parameters())
    trainable_parameters.extend(lora.lora_down.parameters())
  return trainable_parameters, keys
