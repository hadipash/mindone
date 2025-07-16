from dataclasses import dataclass, field
from typing import Optional

from fastvideo.src.configs.pipelines import PipelineConfig


# args for fastvideo framework
@dataclass
class FastVideoArgs:
    """
    Configuration arguments for the FastVideo framework.

    Attributes:
        model_path: The path of the model weights. This can be a local folder or a Hugging Face repo ID.
        model_dir: The directory of the model weights.
        cache_strategy: Caching strategy.
        inference_mode: Whether to run in inference mode. Default: True.
        trust_remote_code: Trust remote code for HuggingFace. Default: False.
        revision: Model revision for HuggingFace.
        num_npus: Number of NPUs to use. Default: 1.
        tp_size: Tensor parallelism size. Default: -1.
        sp_size: Sequence parallelism size. Default: -1.
        hsdp_replicate_dim: Data parallelism size.. Default: 1.
        hsdp_shard_dim: Data parallelism shards. Default: -1.
        dist_timeout: Set timeout for mindspore.distributed initialization.
        pipeline_config: Pipeline configuration
        output_type: Output type for the generated video. Default: "pil".
        use_cpu_offload: Not supported.
        use_fsdp_inference: Use FSDP for inference by sharding the model weights.
                            Latency is very low due to prefetch - enable if run out of memory. Default: True.
        text_encoder_offload: Not supported.
        # mask_strategy_file_path: Path to mask strategy file
        # STA_mode: Sliding Tile Attention mode. Default: STA_INFERENCE.
        skip_time_steps: Number of time steps to warmup (full attention). Default: 15.
        # VSA_sparsity: Validation sparsity for VSA. Default: 0.0.
        enable_stage_verification: Enable input/output verification for pipeline stages. Default: True.
    """

    # Model and path configuration (for convenience)
    model_path: str
    model_dir: Optional[str] = None

    # Cache strategy
    cache_strategy: Optional[str] = None

    inference_mode: bool = True  # if False == training mode

    # HuggingFace specific parameters
    trust_remote_code: bool = False
    revision: Optional[str] = None

    # Parallelism
    num_npus: int = 1
    tp_size: int = -1
    sp_size: int = -1
    hsdp_replicate_dim: int = 1
    hsdp_shard_dim: int = -1
    dist_timeout: Optional[int] = None  # timeout for torch.distributed

    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig)

    output_type: str = "pil"

    use_cpu_offload: bool = True  # For DiT
    use_fsdp_inference: bool = True
    text_encoder_offload: bool = True

    # STA (Sliding Tile Attention) parameters
    # mask_strategy_file_path: Optional[str] = None
    # STA_mode: STA_Mode = STA_Mode.STA_INFERENCE
    skip_time_steps: int = 15

    # VSA parameters
    # VSA_sparsity: float = 0.0  # inference/validation sparsity

    # Stage verification
    enable_stage_verification: bool = True

    @property
    def training_mode(self) -> bool:
        return not self.inference_mode

    @classmethod
    def from_kwargs(cls, **kwargs) -> "FastVideoArgs":
        kwargs["pipeline_config"] = PipelineConfig.from_kwargs(kwargs)
        return cls(**kwargs)

    def __post_init__(self):
        if self.pipeline_config is None:
            raise ValueError("pipeline_config is not set in FastVideoArgs")

        """Validate inference arguments for consistency"""
        if not self.inference_mode:
            assert self.hsdp_replicate_dim != -1, "hsdp_replicate_dim must be set for training"
            assert self.hsdp_shard_dim != -1, "hsdp_shard_dim must be set for training"
            assert self.sp_size != -1, "sp_size must be set for training"

        if self.tp_size == -1:
            self.tp_size = 1
        if self.sp_size == -1:
            self.sp_size = self.num_npus
        if self.hsdp_shard_dim == -1:
            self.hsdp_shard_dim = self.num_npus

        if not (self.sp_size <= self.num_npus and self.num_npus % self.sp_size == 0):
            raise ValueError("num_npus must >= and be divisible by sp_size")
        if not (self.hsdp_replicate_dim <= self.num_npus and self.num_npus % self.hsdp_replicate_dim == 0):
            raise ValueError("num_npus must >= and be divisible by hsdp_replicate_dim")
        if not (self.hsdp_shard_dim <= self.num_npus and self.num_npus % self.hsdp_shard_dim == 0):
            raise ValueError("num_npus must >= and be divisible by hsdp_shard_dim")

        if self.num_npus < max(self.tp_size, self.sp_size):
            self.num_npus = max(self.tp_size, self.sp_size)


@dataclass
class TrainingArgs(FastVideoArgs):
    # no training for now
    pass
