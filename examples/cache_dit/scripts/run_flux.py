from time import perf_counter

import cache_dit
from numpy.random import default_rng
from utils import get_args

from mindspore import bfloat16

from mindone.diffusers import FluxPipeline

args = get_args()
print(args)


pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", mindone_dtype=bfloat16)


if args.cache:
    cache_dit.enable_cache(pipe)


start = perf_counter()
image = pipe("A cat holding a sign that says hello world", num_inference_steps=28, generator=default_rng(0)).images[0]

end = perf_counter()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"flux.{cache_dit.strify(stats)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
