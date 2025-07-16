def maybe_init_distributed_environment_and_model_parallel(
    tp_size: int, sp_size: int, distributed_init_method: str = "env://"
):
    pass


def cleanup_dist_env_and_memory(shutdown_ray: bool = False):
    pass
