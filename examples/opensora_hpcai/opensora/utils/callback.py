import logging
import os
import time
from typing import List

import numpy as np

from mindspore.train import Callback, RunContext

_logger = logging.getLogger(__name__)


class PerfRecorder(Callback):
    """
    Improved version of `mindone.trainers.recorder.PerfRecorder` that tracks validation metrics as well.
    Used here first for testing.
    """

    def __init__(
        self,
        save_dir: str,
        file_name: str = "result.log",
        metric_names: List[str] = None,
        separator: str = "\t",
        resume: bool = False,
    ):
        self._sep = separator
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            _logger.info(f"{save_dir} does not exist. Created.")

        self._metrics = metric_names

        self._log_file = os.path.join(save_dir, file_name)
        if not resume:
            header = separator.join(["step", "loss", "train_time(s)"] + metric_names)
            with open(self._log_file, "w", encoding="utf-8") as fp:
                fp.write(header + "\n")

    def on_train_step_begin(self, run_context: RunContext):
        self._step_time = time.perf_counter()

    def on_train_step_end(self, run_context: RunContext):
        step_time = time.perf_counter() - self._step_time
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num
        loss = cb_params.net_outputs
        loss = loss[0].asnumpy() if isinstance(loss, tuple) else np.mean(loss.asnumpy())

        with open(self._log_file, "a", encoding="utf-8") as fp:
            fp.write(f"{cur_step:<8}{self._sep}{loss.item():<10.6f}{self._sep}{step_time:<8.3f}\n")

    def on_eval_end(self, run_context: RunContext):
        cb_params = run_context.original_args()
        cur_step = "-"
        step_time = "-"
        loss = "-"

        metrics = cb_params.metrics
        try:
            metrics = self._sep.join([f"{metrics[m]:.4f}" for m in self._metrics])
        except KeyError:
            raise KeyError(f"Metric ({self._metrics}) not found in eval result ({list(metrics.keys())}).")
        with open(self._log_file, "a", encoding="utf-8") as fp:
            fp.write(f"{cur_step:<8}{self._sep}{loss:<10}{self._sep}{step_time:<8}{self._sep}{metrics}\n")
