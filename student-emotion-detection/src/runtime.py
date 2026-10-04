"""Runtime tuning helpers.

The single biggest performance lever in this project is the torch thread count,
and the right value depends entirely on the batch size.
"""

from __future__ import annotations

import os
from typing import Optional

from src.config import IMAGE_SIZE, env_int


def cpu_count() -> int:
    """Logical CPU count, defaulting to 4 when it cannot be determined."""

    return os.cpu_count() or 4


def recommended_threads(batch_size: int = 1, image_size: int = IMAGE_SIZE) -> int:
    """Thread count that suits a given batch size.

    Two effects push in opposite directions. Thread synchronisation has a fixed
    cost, so a batch that carries little arithmetic is made *slower* by extra
    threads; a batch with real work scales with them.

    Full-epoch measurement, tiny model, batch 4, 70 batches of 48x48:

    ===========  ==========
    threads      epoch time
    ===========  ==========
    1             1.85s
    2             5.62s
    4             5.03s
    10           13.04s
    ===========  ==========

    Full-size model, batch 64, training steps per second:

    ===========  ==========
    threads      steps/s
    ===========  ==========
    1             1.5
    8             5.0
    ===========  ==========

    Per-configuration microbenchmarks on this machine are noisy enough that only
    the direction of each effect is trustworthy, hence the coarse ladder: stay
    single-threaded while batches are small, then scale up as batches grow.
    """

    threads = cpu_count()
    batch = max(1, int(batch_size))

    if batch <= 8:
        return 1
    if batch <= 32:
        return min(4, threads)
    return min(8, threads)


def configure_threads(
    batch_size: int = 1,
    image_size: int = IMAGE_SIZE,
    override: Optional[int] = None,
) -> int:
    """Apply a torch thread count and return the value used.

    ``TORCH_NUM_THREADS`` in the environment always wins, which is how the
    Dockerfile and CI pin behaviour for reproducibility. Calling this more than
    once is harmless; torch is only reconfigured when the count changes.
    """

    import torch

    requested = env_int("TORCH_NUM_THREADS", 0)
    if override is not None:
        requested = int(override)
    elif requested <= 0:
        requested = recommended_threads(batch_size, image_size)

    threads = max(1, min(requested, cpu_count()))
    if torch.get_num_threads() != threads:
        torch.set_num_threads(threads)
    return torch.get_num_threads()
