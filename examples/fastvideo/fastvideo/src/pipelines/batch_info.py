import pprint
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Union

from fastvideo.src.configs.sample.teacache import TeaCacheParams, WanTeaCacheParams
from PIL.Image import Image

from mindspore import Generator, Tensor


@dataclass
class ForwardBatch:
    """
    Complete state passed through the pipeline execution.

    This dataclass contains all information needed during the diffusion pipeline
    execution, allowing methods to update specific components without needing
    to manage numerous individual parameters.
    """

    # TODO(will): double check that args are separate from fastvideo_args
    # properly. Also maybe think about providing an abstraction for pipeline
    # specific arguments.
    data_type: str

    generator: Optional[Union[Generator, list[Generator]]] = None

    # Image inputs
    image_path: Optional[str] = None
    image_embeds: list[Tensor] = field(default_factory=list)
    pil_image: Optional[Image] = None
    preprocessed_image: Tensor | None = None

    # Text inputs
    prompt: Optional[Union[str, list[str]]] = None
    negative_prompt: Optional[Union[str, list[str]]] = None
    prompt_path: Optional[str] = None
    output_path: str = "outputs/"
    output_video_name: Optional[str] = None
    # Primary encoder embeddings
    prompt_embeds: list[Tensor] = field(default_factory=list)
    negative_prompt_embeds: Optional[list[Tensor]] = None
    prompt_attention_mask: Optional[list[Tensor]] = None
    negative_attention_mask: Optional[list[Tensor]] = None
    clip_embedding_pos: Optional[list[Tensor]] = None
    clip_embedding_neg: Optional[list[Tensor]] = None

    # Additional text-related parameters
    max_sequence_length: Optional[int] = None
    prompt_template: Optional[dict[str, Any]] = None
    do_classifier_free_guidance: bool = False

    # Batch info
    batch_size: Optional[int] = None
    num_videos_per_prompt: int = 1
    seed: Optional[int] = None
    seeds: Optional[list[int]] = None

    # Tracking if embeddings are already processed
    is_prompt_processed: bool = False

    # Latent tensors
    latents: Tensor | None = None
    raw_latent_shape: Tensor | None = None
    noise_pred: Tensor | None = None
    image_latent: Tensor | None = None

    # Latent dimensions
    height_latents: Optional[int] = None
    width_latents: Optional[int] = None
    num_frames: int = 1  # Default for image models
    num_frames_round_down: bool = False  # Whether to round down num_frames if it's not divisible by num_gpus

    # Original dimensions (before VAE scaling)
    height: Optional[int] = None
    width: Optional[int] = None
    fps: Optional[int] = None

    # Timesteps
    timesteps: Tensor | None = None
    timestep: Tensor | float | int | None = None
    step_index: Optional[int] = None

    # Scheduler parameters
    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    guidance_rescale: float = 0.0
    eta: float = 0.0
    sigmas: Optional[list[float]] = None

    n_tokens: Optional[int] = None

    # Other parameters that may be needed by specific schedulers
    extra_step_kwargs: dict[str, Any] = field(default_factory=dict)

    # Component modules (populated by the pipeline)
    modules: dict[str, Any] = field(default_factory=dict)

    # Final output (after pipeline completion)
    output: Any = None

    # Extra parameters that might be needed by specific pipeline implementations
    extra: dict[str, Any] = field(default_factory=dict)

    # Misc
    save_video: bool = True
    return_frames: bool = False

    # TeaCache parameters
    enable_teacache: bool = False
    teacache_params: Optional[Union[TeaCacheParams, WanTeaCacheParams]] = None

    # STA parameters
    STA_param: Optional[list] = None
    is_cfg_negative: bool = False
    mask_search_final_result_pos: Optional[list[list]] = None
    mask_search_final_result_neg: Optional[list[list]] = None

    # VSA parameters
    VSA_sparsity: float = 0.0

    def __post_init__(self):
        # Set do_classifier_free_guidance based on guidance scale and negative prompt
        if self.guidance_scale > 1.0:
            self.do_classifier_free_guidance = True
        if self.negative_prompt_embeds is None:
            self.negative_prompt_embeds = []

    def __str__(self):
        return pprint.pformat(asdict(self), indent=2, width=120)
