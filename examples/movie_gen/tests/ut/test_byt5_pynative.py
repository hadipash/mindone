import numpy as np
import pytest
import torch
from moviegen.text_encoders.byt5 import ByT5
from transformers import T5ForConditionalGeneration

import mindspore as ms

ms.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", ascend_config={"precision_mode": "must_keep_origin_dtype"})

fp32_fwd_tolerance = 1e-5

test_samples = [
    (
        np.array([list("Life is like a box of chocolates.".encode("utf-8"))]),
        np.array([list("La vie est comme une boîte de chocolat.".encode("utf-8"))]),
    )
]

byt5_ms = ByT5("google/byt5-small")
byt5_pt = T5ForConditionalGeneration.from_pretrained("google/byt5-small", local_files_only=True)


@pytest.mark.parametrize("x, y", test_samples)
def test_forward_fp32(x, y):
    num_special_tokens = 3
    ms_loss = byt5_ms(
        ms.Tensor(x + num_special_tokens, dtype=ms.int32), labels=ms.Tensor(y + num_special_tokens, dtype=ms.int32)
    ).loss
    ms_loss = ms_loss.asnumpy().astype(np.float32)
    pt_loss = byt5_pt(
        input_ids=torch.tensor(x + num_special_tokens, dtype=torch.int32),
        labels=torch.tensor(y + num_special_tokens, dtype=torch.int32),
    ).loss
    pt_loss = pt_loss.numpy().astype(np.float32)
    assert np.allclose(ms_loss, pt_loss, atol=fp32_fwd_tolerance, rtol=0)
