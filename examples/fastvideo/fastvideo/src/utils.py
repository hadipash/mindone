import ctypes
import hashlib
import importlib
import json
import math
import os
import signal
import sys
import tempfile
import traceback
from dataclasses import fields, is_dataclass
from typing import Any, Optional, cast

import filelock
from huggingface_hub import snapshot_download
from loguru import logger

import mindspore as ms

PRECISION_TO_TYPE = {
    "fp32": ms.float32,
    "fp16": ms.float16,
    "bf16": ms.bfloat16,
}


def get_lock(model_name_or_path: str):
    lock_dir = tempfile.gettempdir()
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)
    model_name = model_name_or_path.replace("/", "-")
    hash_name = hashlib.sha256(model_name.encode()).hexdigest()
    # add hash to avoid conflict with old users' lock files
    lock_file_name = hash_name + model_name + ".lock"
    # mode 0o666 is required for the filelock to be shared across users
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name), mode=0o666)
    return lock


def align_to(value: int, alignment: int) -> int:
    """align height, width according to alignment

    Args:
        value (int): height or width
        alignment (int): target alignment factor

    Returns:
        int: the aligned value
    """
    return int(math.ceil(value / alignment) * alignment)


def resolve_obj_by_qualname(qualname: str) -> Any:
    """
    Resolve an object by its fully qualified name.
    """
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


def maybe_download_model(model_name_or_path: str, local_dir: Optional[str] = None, download: bool = True) -> str:
    """
    Check if the model path is a Hugging Face Hub model ID and download it if needed.

    Args:
        model_name_or_path: Local path or Hugging Face Hub model ID
        local_dir: Local directory to save the model
        download: Whether to download the model from Hugging Face Hub

    Returns:
        Local path to the model
    """

    # If the path exists locally, return it
    if os.path.exists(model_name_or_path):
        logger.info("Model already exists locally at {}", model_name_or_path)
        return model_name_or_path

    # Otherwise, assume it's a HF Hub model ID and try to download it
    try:
        logger.info("Downloading model snapshot from HF Hub for {}...", model_name_or_path)
        with get_lock(model_name_or_path):
            local_path = snapshot_download(
                repo_id=model_name_or_path, ignore_patterns=["*.onnx", "*.msgpack"], local_dir=local_dir
            )
        logger.info("Downloaded model to {}", local_path)
        return str(local_path)
    except Exception as e:
        raise ValueError(f"Could not find model at {model_name_or_path} and failed to download from HF Hub: {e}") from e


def verify_model_config_and_directory(model_path: str) -> dict[str, Any]:
    """
    Verify that the model directory contains a valid diffusers configuration.

    Args:
        model_path: Path to the model directory

    Returns:
        The loaded model configuration as a dictionary
    """

    # Check for model_index.json which is required for diffusers models
    config_path = os.path.join(model_path, "model_index.json")
    if not os.path.exists(config_path):
        raise ValueError(
            f"Model directory {model_path} does not contain model_index.json. "
            "Only Hugging Face diffusers format is supported."
        )

    # Check for transformer and vae directories
    transformer_dir = os.path.join(model_path, "transformer")
    vae_dir = os.path.join(model_path, "vae")

    if not os.path.exists(transformer_dir):
        raise ValueError(f"Model directory {model_path} does not contain a transformer/ directory.")

    if not os.path.exists(vae_dir):
        raise ValueError(f"Model directory {model_path} does not contain a vae/ directory.")

    # Load the config
    with open(config_path) as f:
        config = json.load(f)

    # Verify diffusers version exists
    if "_diffusers_version" not in config:
        raise ValueError("model_index.json does not contain _diffusers_version")

    logger.info("Diffusers version: {}", config["_diffusers_version"])
    return cast(dict[str, Any], config)


def maybe_download_model_index(model_name_or_path: str) -> dict[str, Any]:
    """
    Download and extract just the model_index.json for a Hugging Face model.

    Args:
        model_name_or_path: Path or HF Hub model ID

    Returns:
        The parsed model_index.json as a dictionary
    """
    import tempfile

    from huggingface_hub import hf_hub_download

    # If it's a local path, verify it directly
    if os.path.exists(model_name_or_path):
        return verify_model_config_and_directory(model_name_or_path)

    # For remote models, download just the model_index.json
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download just the model_index.json file
            model_index_path = hf_hub_download(
                repo_id=model_name_or_path, filename="model_index.json", local_dir=tmp_dir
            )

            # Load the model_index.json
            with open(model_index_path) as f:
                config: dict[str, Any] = json.load(f)

            # Verify it has the required fields
            if "_class_name" not in config:
                raise ValueError(f"model_index.json for {model_name_or_path} does not contain _class_name field")

            if "_diffusers_version" not in config:
                raise ValueError(f"model_index.json for {model_name_or_path} does not contain _diffusers_version field")

            # Add the pipeline name for downstream use
            config["pipeline_name"] = config["_class_name"]

            logger.info("Downloaded model_index.json for {}, pipeline: {}", model_name_or_path, config["_class_name"])
            return config

    except Exception as e:
        raise ValueError(f"Failed to download or parse model_index.json for {model_name_or_path}: {e}") from e


def shallow_asdict(obj) -> dict[str, Any]:
    if not is_dataclass(obj):
        raise TypeError("Expected dataclass instance")
    return {f.name: getattr(obj, f.name) for f in fields(obj)}


# TODO: validate that this is fine
def kill_itself_when_parent_died() -> None:
    # if sys.platform == "linux":
    # sigkill this process when parent worker manager dies
    PR_SET_PDEATHSIG = 1
    import platform

    if platform.system() == "Linux":
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    # elif platform.system() == "Darwin":
    #     libc = ctypes.CDLL("libc.dylib")
    #     logger.warning("kill_itself_when_parent_died is only supported in linux.")
    else:
        logger.warning("kill_itself_when_parent_died is only supported in linux.")


def get_exception_traceback() -> str:
    etype, value, tb = sys.exc_info()
    err_str = "".join(traceback.format_exception(etype, value, tb))
    return err_str
