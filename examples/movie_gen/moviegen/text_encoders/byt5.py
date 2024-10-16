from typing import Optional

from mindone.transformers import T5ForConditionalGeneration


class ByT5:  # nn.Cell?
    def __init__(self, name: str, model_path: Optional[str] = None):
        self.model = T5ForConditionalGeneration.from_pretrained(name, local_files_only=True, cache_dir=model_path)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
