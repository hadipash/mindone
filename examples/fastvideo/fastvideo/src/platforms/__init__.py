# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/platforms/__init__.py

import traceback
from typing import Union

# imported by other files, do not remove
from fastvideo.src.utils import resolve_obj_by_qualname


def cuda_platform_plugin() -> Union[str, None]:
    return None


def cpu_platform_plugin() -> Union[str, None]:
    """Detect if CPU platform should be used."""
    # CPU is always available as a fallback
    return "fastvideo.v1.platforms.cpu.CpuPlatform"


builtin_platform_plugins = {
    "cuda": cuda_platform_plugin,
    "cpu": cpu_platform_plugin,
}


def resolve_current_platform_cls_qualname() -> str:
    # TODO(will): if we need to support other platforms, we should consider if
    # vLLM's plugin architecture is suitable for our needs.

    # Fall back to CUDA
    platform_cls_qualname = cuda_platform_plugin()
    if platform_cls_qualname is not None:
        return platform_cls_qualname

    # Fall back to CPU as last resort
    platform_cls_qualname = cpu_platform_plugin()
    if platform_cls_qualname is not None:
        return platform_cls_qualname

    raise RuntimeError("No platform plugin found. Please check your installation.")


_current_platform = None
_init_trace: str = ""


def __getattr__(name: str):
    if name == "current_platform":
        # lazy init current_platform.
        # 1. out-of-tree platform plugins need `from fastvideo.platforms import
        #    Platform` so that they can inherit `Platform` class. Therefore,
        #    we cannot resolve `current_platform` during the import of
        #    `fastvideo.platforms`.
        # 2. when users use out-of-tree platform plugins, they might run
        #    `import fastvideo`, some fastvideo internal code might access
        #    `current_platform` during the import, and we need to make sure
        #    `current_platform` is only resolved after the plugins are loaded
        #    (we have tests for this, if any developer violate this, they will
        #    see the test failures).
        global _current_platform
        if _current_platform is None:
            platform_cls_qualname = resolve_current_platform_cls_qualname()
            _current_platform = resolve_obj_by_qualname(platform_cls_qualname)()
            global _init_trace
            _init_trace = "".join(traceback.format_stack())
        return _current_platform
    elif name in globals():
        return globals()[name]
    else:
        raise AttributeError(f"No attribute named '{name}' exists in {__name__}.")


__all__ = ["_init_trace"]
