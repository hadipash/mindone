# SPDX-License-Identifier: Apache-2.0
"""
Diffusion pipelines for fastvideo.v1.

This package contains diffusion pipelines for generating videos and images.
"""

from typing import cast

from fastvideo.src.entrypoints.args import FastVideoArgs
from fastvideo.src.pipelines.batch_info import ForwardBatch
from fastvideo.src.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.src.pipelines.lora_pipeline import LoRAPipeline
from fastvideo.src.pipelines.registry import PipelineRegistry
from fastvideo.src.utils import maybe_download_model, verify_model_config_and_directory
from loguru import logger


class PipelineWithLoRA(LoRAPipeline, ComposedPipelineBase):
    """Type for a pipeline that has both ComposedPipelineBase and LoRAPipeline functionality."""

    pass


def build_pipeline(fastvideo_args: FastVideoArgs) -> PipelineWithLoRA:
    """
    Only works with valid hf diffusers configs. (model_index.json)
    We want to build a pipeline based on the inference args mode_path:
    1. download the model from the hub if it's not already downloaded
    2. verify the model config and directory
    3. based on the config, determine the pipeline class
    """
    # Get pipeline type
    model_path = fastvideo_args.model_path
    model_path = maybe_download_model(model_path)
    # fastvideo_args.downloaded_model_path = model_path
    logger.info("Model path: {}", model_path)
    config = verify_model_config_and_directory(model_path)

    pipeline_architecture = config.get("_class_name")
    if pipeline_architecture is None:
        raise ValueError(
            "Model config does not contain a _class_name attribute. " "Only diffusers format is supported."
        )

    pipeline_cls, pipeline_architecture = PipelineRegistry.resolve_pipeline_cls(pipeline_architecture)

    # instantiate the pipeline
    pipeline = pipeline_cls(model_path, fastvideo_args)
    logger.info("Pipeline instantiated")

    # pipeline is now initialized and ready to use
    return cast(PipelineWithLoRA, pipeline)


__all__ = ["build_pipeline", "ComposedPipelineBase", "PipelineRegistry", "ForwardBatch", "LoRAPipeline"]
