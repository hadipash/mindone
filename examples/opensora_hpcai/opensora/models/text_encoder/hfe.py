from transformers import CLIPTokenizer, T5Tokenizer

from mindspore import Tensor, mint, nn

from mindone.transformers import CLIPTextModel, T5EncoderModel


class HFEmbedder(nn.Cell):
    def __init__(self, from_pretrained: str, max_length: int, **hf_kwargs):
        super().__init__()
        self.is_clip = "openai" in from_pretrained
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(from_pretrained, max_length=max_length)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(from_pretrained, **hf_kwargs)
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
                from_pretrained, max_length=max_length, legacy=True
            )
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(from_pretrained, **hf_kwargs)

    def construct(self, text: list[str], added_tokens: int = 0, seq_align: int = 1) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        seq_len = batch_encoding["input_ids"].shape[1]
        if (added_tokens + seq_len) % seq_align != 0:
            num_pad_tokens = seq_align - (added_tokens + seq_len) % seq_align
            batch_encoding["input_ids"] = mint.nn.functional.pad(
                batch_encoding["input_ids"], (0, num_pad_tokens), value=self.tokenizer.pad_token_id
            )

        outputs = self.hf_module(input_ids=batch_encoding["input_ids"], attention_mask=None, output_hidden_states=False)
        return outputs[self.output_key]
