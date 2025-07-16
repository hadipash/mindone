import json
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any, Callable, Optional, Union

from fastvideo.src.configs.models import DiTConfig, ModelConfig, VAEConfig
from fastvideo.src.configs.models.encoders import BaseEncoderOutput, EncoderConfig
from fastvideo.src.configs.utils import update_config_from_args
from fastvideo.src.utils import shallow_asdict
from loguru import logger

from mindspore import Tensor


class STA_Mode(str, Enum):
    """STA (Sliding Tile Attention) modes."""

    STA_INFERENCE = "STA_inference"
    STA_SEARCHING = "STA_searching"
    STA_TUNING = "STA_tuning"
    STA_TUNING_CFG = "STA_tuning_cfg"
    NONE = None


def preprocess_text(prompt: str) -> str:
    return prompt


def postprocess_text(output: BaseEncoderOutput) -> Tensor:
    raise NotImplementedError


# config for a single pipeline
@dataclass
class PipelineConfig:
    """
    Base configuration for all pipeline architectures.

    Attributes:
        model_path: Path to the pretrained model.
        pipeline_config_path: Path to the pipeline config file. Default: None.
        embedded_cfg_scale: Embedded CFG scale. Default: 6.0.
        flow_shift: Flow shift parameter for video generation. Default: None.

        dit_config: Configuration for DiT model. Default: DiTConfig().
        dit_precision: Precision for DiT model. Default: "bf16".

        vae_config: Configuration for VAE model. Default: VAEConfig().
        vae_precision: Precision for VAE. Default: "fp32".
        vae_tiling: Enable VAE tiling. Default: True.
        vae_sp: Enable VAE spatial parallelism. Default: True.

        image_encoder_config: Configuration for image encoder. Default: EncoderConfig().
        image_encoder_precision: Precision for image encoder. Default: "fp32".

        text_encoder_configs: Configurations for text encoders. Default: (EncoderConfig(),).
        text_encoder_precisions: Precisions for text encoders. Default: ("fp32",).
        preprocess_text_funcs: Text preprocessing functions. Default: (preprocess_text,).
        postprocess_text_funcs: Text postprocessing functions. Default: (postprocess_text,).

        lora_path: Path to LoRA adapter weights. Default: None.
        lora_nickname: Nickname for LoRA adapter. Default: "default".
        lora_target_names: List of target layer names for LoRA. Default: None.

        mask_strategy_file_path: Path to STA mask strategy file. Default: None.
        STA_mode: STA operation mode. Default: STA_Mode.STA_INFERENCE.
        skip_time_steps: Number of time steps to skip in STA. Default: 15.
    """

    model_path: Optional[str] = None
    pipeline_config_path: Optional[str] = None

    # Video generation parameters
    embedded_cfg_scale: float = 6.0
    flow_shift: Optional[float] = None

    # Model configuration
    dit_config: DiTConfig = field(default_factory=DiTConfig)
    dit_precision: str = "bf16"

    # VAE configuration
    vae_config: VAEConfig = field(default_factory=VAEConfig)
    vae_precision: str = "fp32"
    vae_tiling: bool = True
    vae_sp: bool = True

    # Image encoder configuration
    image_encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    image_encoder_precision: str = "fp32"

    # Text encoder configuration
    DEFAULT_TEXT_ENCODER_PRECISIONS = ("fp32",)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(default_factory=lambda: (EncoderConfig(),))
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp32",))
    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(default_factory=lambda: (preprocess_text,))
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], Tensor], ...] = field(
        default_factory=lambda: (postprocess_text,)
    )

    # LoRA parameters
    lora_path: Optional[str] = None
    lora_nickname: Optional[str] = "default"  # for swapping adapters in the pipeline
    lora_target_names: Optional[list[str]] = None  # can restrict list of layers to adapt, e.g. ["q_proj"]

    # STA (Sliding Tile Attention) parameters
    mask_strategy_file_path: Optional[str] = None
    STA_mode: STA_Mode = STA_Mode.STA_INFERENCE
    skip_time_steps: int = 15

    # Compilation
    # enable_torch_compile: bool = False

    def update_config_from_dict(self, args: dict[str, Any], prefix: str = "") -> None:
        prefix_with_dot = f"{prefix}." if (prefix.strip() != "") else ""
        update_config_from_args(self, args, prefix, pop_args=True)
        update_config_from_args(self.vae_config, args, f"{prefix_with_dot}vae_config", pop_args=True)
        update_config_from_args(self.dit_config, args, f"{prefix_with_dot}dit_config", pop_args=True)

    @classmethod
    def from_pretrained(cls, model_path: str) -> "PipelineConfig":
        raise NotImplementedError

    @classmethod
    def from_kwargs(cls, kwargs: dict[str, Any], config_cli_prefix: str = "") -> "PipelineConfig":
        """
        Load PipelineConfig from kwargs Dictionary.
        kwargs: dictionary of kwargs
        config_cli_prefix: prefix of CLI arguments for this PipelineConfig instance
        """
        from fastvideo.src.configs.pipelines.registry import get_pipeline_config_cls_from_name

        prefix_with_dot = f"{config_cli_prefix}." if (config_cli_prefix.strip() != "") else ""
        model_path: Optional[str] = kwargs.get(prefix_with_dot + "model_path", None) or kwargs.get("model_path")
        pipeline_config_or_path: Union[str, PipelineConfig, dict[str, Any], None] = kwargs.get(
            prefix_with_dot + "pipeline_config", None
        ) or kwargs.get("pipeline_config")
        if model_path is None:
            raise ValueError("model_path is required in kwargs")

        # 1. Get the pipeline config class from the registry
        pipeline_config_cls = get_pipeline_config_cls_from_name(model_path)

        # 2. Instantiate PipelineConfig
        if pipeline_config_cls is None:
            logger.warning("Couldn't find pipeline config for {}. Using the default pipeline config.", model_path)
            pipeline_config = cls()
        else:
            pipeline_config = pipeline_config_cls()

        # 3. Load PipelineConfig from a json file or a PipelineConfig object if provided
        if isinstance(pipeline_config_or_path, str):
            pipeline_config.load_from_json(pipeline_config_or_path)
            kwargs[prefix_with_dot + "pipeline_config_path"] = pipeline_config_or_path
        elif isinstance(pipeline_config_or_path, PipelineConfig):
            pipeline_config = pipeline_config_or_path
        elif isinstance(pipeline_config_or_path, dict):
            pipeline_config.update_pipeline_config(pipeline_config_or_path)

        # 4. Update PipelineConfig from CLI arguments if provided
        kwargs[prefix_with_dot + "model_path"] = model_path
        pipeline_config.update_config_from_dict(kwargs, config_cli_prefix)
        return pipeline_config

    def __post_init__(self):
        if self.vae_sp and not self.vae_tiling:
            raise ValueError("Currently enabling vae_sp requires enabling vae_tiling, please set --vae-tiling to True.")

        if len(self.text_encoder_configs) != len(self.text_encoder_precisions):
            raise ValueError(
                f"Length of text encoder configs ({len(self.text_encoder_configs)}) must be"
                f" equal to length of text encoder precisions ({len(self.text_encoder_precisions)})"
            )

        if len(self.text_encoder_configs) != len(self.preprocess_text_funcs):
            raise ValueError(
                f"Length of text encoder configs ({len(self.text_encoder_configs)}) must be"
                f" equal to length of text preprocessing functions ({len(self.preprocess_text_funcs)})"
            )

        if len(self.preprocess_text_funcs) != len(self.postprocess_text_funcs):
            raise ValueError(
                f"Length of text postprocess functions ({len(self.postprocess_text_funcs)}) must be"
                f" equal to length of text preprocessing functions ({len(self.preprocess_text_funcs)})"
            )

    def dump_to_json(self, file_path: str):
        output_dict = shallow_asdict(self)
        del_keys = []
        for key, value in output_dict.items():
            if isinstance(value, ModelConfig):
                model_dict = asdict(value)
                # Model Arch Config should be hidden away from the users
                model_dict.pop("arch_config")
                output_dict[key] = model_dict
            elif isinstance(value, tuple) and all(isinstance(v, ModelConfig) for v in value):
                model_dicts = []
                for v in value:
                    model_dict = asdict(v)
                    # Model Arch Config should be hidden away from the users
                    model_dict.pop("arch_config")
                    model_dicts.append(model_dict)
                output_dict[key] = model_dicts
            elif isinstance(value, tuple) and all(callable(f) for f in value):
                # Skip dumping functions
                del_keys.append(key)

        for key in del_keys:
            output_dict.pop(key, None)

        with open(file_path, "w") as f:
            json.dump(output_dict, f, indent=2)

    def load_from_json(self, file_path: str):
        with open(file_path) as f:
            input_pipeline_dict = json.load(f)
        self.update_pipeline_config(input_pipeline_dict)

    def update_pipeline_config(self, source_pipeline_dict: dict[str, Any]) -> None:
        for f in fields(self):
            key = f.name
            if key in source_pipeline_dict:
                current_value = getattr(self, key)
                new_value = source_pipeline_dict[key]

                # If it's a nested ModelConfig, update it recursively
                if isinstance(current_value, ModelConfig):
                    current_value.update_model_config(new_value)
                elif isinstance(current_value, tuple) and all(isinstance(v, ModelConfig) for v in current_value):
                    assert len(current_value) == len(
                        new_value
                    ), "Users shouldn't delete or add text encoder config objects in your json"
                    for target_config, source_config in zip(current_value, new_value, strict=True):
                        target_config.update_model_config(source_config)
                else:
                    setattr(self, key, new_value)

        if hasattr(self, "__post_init__"):
            self.__post_init__()
