# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

from loguru import logger


@dataclass
class SamplingParam:
    """
    Sampling parameters for video generation.

    Attributes:
        data_type: "video" or "image" based on num_frames. Default: "video"
        image_path: Optional path to input image for image-to-video generation. Default: None
        prompt: Text prompt(s) for video generation. Default: None
        negative_prompt: Optional negative prompt. Default: None
        prompt_path: Optional path to prompts file. Default: None
        output_path: Directory to save output videos. Default: "outputs".
        num_videos_per_prompt: Number of videos to generate per prompt. Default: 1
        seed: Random seed. Default: 1024
        num_frames: Number of frames to generate. Default: 125
        num_frames_round_down: Round frames down for NPU divisibility. Default: False
        height: Output height in pixels. Default: 720
        width: Output width in pixels. Default: 1280
        fps: Frames per second. Default: 24
        num_inference_steps: Number of denoising steps. Default: 50
        guidance_scale: Classifier-free guidance scale. Default: 1.0
        guidance_rescale: Guidance rescale factor. Default: 0.0
        enable_teacache: Enable TeaCache optimization. Default: False
        save_video: Whether to save output video. Default: True
        return_frames: Whether to return the raw frames. Default: False

    """

    # All fields below are copied from ForwardBatch
    data_type: Literal["image", "video"] = "video"

    # Image inputs
    image_path: Optional[str] = None

    # Text inputs
    prompt: Optional[Union[str, list[str]]] = None
    negative_prompt: Optional[str] = None
    prompt_path: Optional[str] = None
    output_path: str = "outputs/"

    # Batch info
    num_videos_per_prompt: int = 1
    seed: int = 1024

    # Original dimensions (before VAE scaling)
    num_frames: int = 125
    num_frames_round_down: bool = False  # Whether to round down num_frames if it's not divisible by num_npus
    height: int = 720
    width: int = 1280
    fps: int = 24

    # Denoising parameters
    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    guidance_rescale: float = 0.0

    # TeaCache parameters
    enable_teacache: bool = False

    # Misc
    save_video: bool = True
    return_frames: bool = False

    def __post_init__(self) -> None:
        self.data_type = "video" if self.num_frames > 1 else "image"
        if self.prompt_path and not self.prompt_path.endswith(".txt"):
            raise ValueError("prompt_path must be a txt file")

    def update(self, source_dict: dict[str, Any]) -> None:
        for key, value in source_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.exception("{} has no attribute {}", type(self).__name__, key)

        self.__post_init__()

    @classmethod
    def from_pretrained(cls, model_path: str) -> "SamplingParam":
        raise NotImplementedError


@dataclass
class CacheParams:
    cache_type: str = "none"
