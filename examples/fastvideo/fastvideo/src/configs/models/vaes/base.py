# SPDX-License-Identifier: Apache-2.0
import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Union

from fastvideo.src.configs.models.base import ArchConfig, ModelConfig

from mindspore import Tensor


@dataclass
class VAEArchConfig(ArchConfig):
    scaling_factor: Union[float, Tensor] = 0

    temporal_compression_ratio: int = 4
    spatial_compression_ratio: int = 8


@dataclass
class VAEConfig(ModelConfig):
    """
    Configuration for Variational Autoencoder (VAE) models.

    Attributes:
        arch_config: Architecture configuration for the VAE
        load_encoder: Whether to load the VAE encoder (default: True)
        load_decoder: Whether to load the VAE decoder (default: True)
        tile_sample_min_height: Minimum height for tiling (default: 256)
        tile_sample_min_width: Minimum width for tiling (default: 256)
        tile_sample_min_num_frames: Minimum number of frames for tiling (default: 16)
        tile_sample_stride_height: Stride height for tiling (default: 192)
        tile_sample_stride_width: Stride width for tiling (default: 192)
        tile_sample_stride_num_frames: Frame stride for tiling (default: 12)
        blend_num_frames: Number of frames to blend for tiling (default: 0)
        use_tiling: Enable spatial tiling (default: True)
        use_temporal_tiling: Enable temporal tiling (default: True)
        use_parallel_tiling: Enable parallel tiling (default: True)
        use_temporal_scaling_frames: Enable temporal scaling (default: True)
    """

    arch_config: VAEArchConfig = field(default_factory=VAEArchConfig)

    # FastVideoVAE-specific parameters
    load_encoder: bool = True
    load_decoder: bool = True

    tile_sample_min_height: int = 256
    tile_sample_min_width: int = 256
    tile_sample_min_num_frames: int = 16
    tile_sample_stride_height: int = 192
    tile_sample_stride_width: int = 192
    tile_sample_stride_num_frames: int = 12
    blend_num_frames: int = 0

    use_tiling: bool = True
    use_temporal_tiling: bool = True
    use_parallel_tiling: bool = True
    use_temporal_scaling_frames: bool = True

    def __post_init__(self):
        self.blend_num_frames = self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "VAEConfig":
        kwargs = {}
        for attr in dataclasses.fields(cls):
            value = getattr(args, attr.name, None)
            if value is not None:
                kwargs[attr.name] = value
        return cls(**kwargs)
