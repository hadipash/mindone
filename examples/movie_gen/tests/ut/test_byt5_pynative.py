import os
import sys

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import T5EncoderModel as T5EncoderModel_PyTorch

import mindspore as ms

# FIXME: remove in future when mindone is ready for install
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
from mindone.transformers import T5EncoderModel

ms.set_context(mode=ms.PYNATIVE_MODE)

fp32_tolerance = 1e-4

test_samples = [
    "Life is like a box of chocolates.",
    "La vie est comme une boîte de chocolat.",
    "Today is Monday.",
    "Aujourd'hui c'est lundi.",
]

tokenizer = AutoTokenizer.from_pretrained("google/byt5-small", local_files_only=True)
test_samples = tokenizer(test_samples, padding="longest", return_tensors="np")

byt5_ms = T5EncoderModel.from_pretrained("google/byt5-small", local_files_only=True)
byt5_pt = T5EncoderModel_PyTorch.from_pretrained("google/byt5-small", local_files_only=True)


def test_fp32():
    ms_enc = byt5_ms(
        ms.Tensor(test_samples.input_ids, dtype=ms.int32), ms.Tensor(test_samples.attention_mask, dtype=ms.uint8)
    )
    ms_enc = ms_enc[0].asnumpy().astype(np.float32)
    pt_enc = byt5_pt(torch.tensor(test_samples.input_ids), torch.tensor(test_samples.attention_mask), return_dict=False)
    pt_enc = pt_enc[0].detach().numpy().astype(np.float32)
    assert np.allclose(ms_enc, pt_enc, atol=fp32_tolerance, rtol=0)
