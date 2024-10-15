import numpy as np

from mindspore import Tensor, float16, int32

from mindone.transformers import T5ForConditionalGeneration


class ByT5:  # nn.Cell?
    def __init__(self, name: str, model_path: str):
        self.model = T5ForConditionalGeneration.from_pretrained(name, local_files_only=True, cache_dir=model_path)
        self.model.to_float(float16)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


if __name__ == "__main__":
    model = ByT5("google/byt5-small", model_path="./models")

    num_special_tokens = 3
    input_ids = Tensor(
        np.array([list("Life is like a box of chocolates.".encode("utf-8"))]) + num_special_tokens, dtype=int32
    )
    labels = Tensor(
        np.array([list("La vie est comme une boîte de chocolat.".encode("utf-8"))]) + num_special_tokens, dtype=int32
    )
    loss = model(input_ids, labels=labels).loss
    print(loss.item())
