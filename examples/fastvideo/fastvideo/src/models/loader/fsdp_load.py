from typing import Any, Callable, Optional

import mindspore as ms
from mindspore import nn


def maybe_load_fsdp_model(
    model_cls: type[nn.Cell],
    init_params: dict[str, Any],
    weight_dir_list: list[str],
    hsdp_replicate_dim: int,
    hsdp_shard_dim: int,
    param_dtype: ms.Type,
    reduce_dtype: ms.Type,
    cpu_offload: bool = False,
    fsdp_inference: bool = False,
    output_dtype: Optional[ms.Type] = None,
    training_mode: bool = True,
    pin_cpu_memory: bool = True,
) -> nn.Cell:
    """
    Load the model with FSDP if is training, else load the model without FSDP.
    """
    model = model_cls(**init_params)

    return model


def shard_model(
    model,
    *,
    cpu_offload: bool,
    reshard_after_forward: bool = True,
    mp_policy: None = None,
    mesh: None = None,
    fsdp_shard_conditions: list[Callable[[str, nn.Cell], bool]] = [],
    pin_cpu_memory: bool = True,
) -> None:
    pass
