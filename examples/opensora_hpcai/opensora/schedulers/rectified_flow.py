from tqdm import tqdm

from mindspore import Tensor, dtype, ops
from mindspore.nn.probability.distribution import LogNormal


class RFLOW:
    def __init__(
        self,
        num_sampling_steps=10,
        num_timesteps=1000,
        cfg_scale=4.0,
        use_discrete_timesteps=False,
        use_timestep_transform=False,
        **kwargs,
    ):
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform

        self.scheduler = RFlowScheduler(
            num_timesteps=num_timesteps,
            num_sampling_steps=num_sampling_steps,
            use_discrete_timesteps=use_discrete_timesteps,
            use_timestep_transform=use_timestep_transform,
            **kwargs,
        )

    def __call__(
        self,
        model,
        shape,
        z: Tensor,
        model_kwargs,
        frames_mask=None,
        clip_denoised=False,
        progress=True,
    ):
        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [Tensor([t] * z.shape[0]) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, model_kwargs, num_timesteps=self.num_timesteps) for t in timesteps]

        if frames_mask is not None:
            noise_added = ops.zeros_like(frames_mask, dtype=dtype.bool_)
            noise_added = noise_added | (frames_mask == 1)

        progress_wrap = tqdm if progress else (lambda x: x)
        for i, t in progress_wrap(enumerate(timesteps)):
            # mask for adding noise
            if frames_mask is not None:
                mask_t = frames_mask * self.num_timesteps
                x0 = z.copy()
                x_noise = self.scheduler.add_noise(x0, ops.randn_like(x0), t)

                model_kwargs["frames_mask"] = mask_t_upper = mask_t >= t.unsqueeze(1)
                mask_add_noise = mask_t_upper & ~noise_added

                z = ops.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                noise_added = mask_t_upper

            pred = model(z, t, **model_kwargs)

            # update z
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            dt = dt / self.num_timesteps
            z = z + pred * dt[:, None, None, None, None]

            if frames_mask is not None:
                z = ops.where(mask_t_upper[:, None, :, None, None], z, x0)

        return z


def timestep_transform(
    t,
    model_kwargs,
    base_resolution=512 * 512,
    base_num_frames=1,
    scale=1.0,
    num_timesteps=1,
):
    # FIXME: avoid calculations on tensors outside `construct`
    t = t / num_timesteps
    resolution = model_kwargs["height"].astype(dtype.float32) * model_kwargs["width"].astype(dtype.float32)
    ratio_space = (resolution / base_resolution).sqrt()
    # NOTE: currently, we do not take fps into account
    # NOTE: temporal_reduction is hardcoded, this should be equal to the temporal reduction factor of the vae
    if model_kwargs["num_frames"][0] == 1:
        num_frames = ops.ones_like(model_kwargs["num_frames"])
    else:
        num_frames = model_kwargs["num_frames"].astype(dtype.float32) // 17 * 5
    ratio_time = (num_frames / base_num_frames).sqrt()

    ratio = ratio_space * ratio_time * scale
    new_t = ratio * t / (1 + (ratio - 1) * t)

    new_t = new_t * num_timesteps
    return new_t


class RFlowScheduler:
    def __init__(
        self,
        num_timesteps=1000,
        num_sampling_steps=10,
        use_discrete_timesteps=False,
        sample_method="uniform",
        loc=0.0,
        scale=1.0,
        use_timestep_transform=False,
        transform_scale=1.0,
    ):
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps
        self.use_discrete_timesteps = use_discrete_timesteps

        # sample method
        assert sample_method in ["uniform", "logit-normal"]
        assert (
            sample_method == "uniform" or not use_discrete_timesteps
        ), "Only uniform sampling is supported for discrete timesteps"
        self.sample_method = sample_method
        if sample_method == "logit-normal":  # TODO: check this
            # seed=None <- use global seed
            self.distribution = LogNormal(Tensor([loc]), Tensor([scale]), seed=None, dtype=dtype.float32)
            self.sample_t = lambda x: self.distribution.sample(x.shape[0])[:, 0]

        # timestep transform
        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale

    def add_noise(
        self,
        original_samples: Tensor,
        noise: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        """
        Compatible with diffusers add_noise()
        """
        timepoints = timesteps.float() / self.num_timesteps
        timepoints = 1 - timepoints  # [1,1/1000]

        # timepoint  (bsz) noise: (bsz, 4, frame, w ,h)
        # expand timepoint to noise shape
        timepoints = timepoints[:, None, None, None, None]
        timepoints = timepoints.tile((1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4]))

        return timepoints * original_samples + (1 - timepoints) * noise
