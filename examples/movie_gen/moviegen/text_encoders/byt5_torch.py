import torch
from transformers import T5ForConditionalGeneration


class ByT5:  # nn.Cell?
    def __init__(self, name: str, model_path: str):
        self.model = T5ForConditionalGeneration.from_pretrained(name, local_files_only=True)
        self.model.half()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


if __name__ == "__main__":
    model = ByT5("google/byt5-small", model_path="./models")

    num_special_tokens = 3
    input_ids = torch.tensor([list("Life is like a box of chocolates.".encode("utf-8"))]) + num_special_tokens
    labels = torch.tensor([list("La vie est comme une boîte de chocolat.".encode("utf-8"))]) + num_special_tokens
    loss = model(input_ids, labels=labels).loss
    print(loss.item())
