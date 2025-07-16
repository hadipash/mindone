# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.src.configs.models.base import ArchConfig, ModelConfig
from fastvideo.src.platforms.interface import AttentionBackendEnum


@dataclass
class DiTArchConfig(ArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=list)
    _compile_conditions: list = field(default_factory=list)
    param_names_mapping: dict = field(default_factory=dict)
    reverse_param_names_mapping: dict = field(default_factory=dict)
    lora_param_names_mapping: dict = field(default_factory=dict)
    _supported_attention_backends: tuple[AttentionBackendEnum, ...] = (
        AttentionBackendEnum.SLIDING_TILE_ATTN,
        AttentionBackendEnum.SAGE_ATTN,
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.VIDEO_SPARSE_ATTN,
    )

    hidden_size: int = 0
    num_attention_heads: int = 0
    num_channels_latents: int = 0
    exclude_lora_layers: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self._compile_conditions:
            self._compile_conditions = self._fsdp_shard_conditions.copy()


@dataclass
class DiTConfig(ModelConfig):
    arch_config: DiTArchConfig = field(default_factory=DiTArchConfig)

    # FastVideoDiT-specific parameters
    prefix: str = ""
