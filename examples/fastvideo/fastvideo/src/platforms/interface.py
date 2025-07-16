# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/platforms/interface.py

import enum


class AttentionBackendEnum(enum.Enum):
    FLASH_ATTN = enum.auto()
    SLIDING_TILE_ATTN = enum.auto()
    SAGE_ATTN = enum.auto()
    VIDEO_SPARSE_ATTN = enum.auto()
    NO_ATTENTION = enum.auto()
