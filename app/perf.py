"""Process tweaks that matter on a hybrid-CPU Windows laptop.

Token-by-token decoding launches thousands of tiny GPU kernels, so it is limited by how fast
one CPU thread can launch them. If Windows schedules that thread on an efficiency core, the
GPU starves, down-clocks, and generation gets ~2x slower. Pinning to performance cores and
raising priority avoids that.
"""
import logging
import sys

import psutil

log = logging.getLogger(__name__)


def performance_cores() -> list[int] | None:
    """Logical CPU ids of the P-cores on a hybrid Intel CPU, or None if it isn't hybrid.

    Hybrid parts have hyper-threaded P-cores and single-threaded E-cores, numbered P first:
    logical = 2P + E and physical = P + E, so P = logical - physical.
    """
    logical, physical = psutil.cpu_count(), psutil.cpu_count(logical=False)
    if not logical or not physical:
        return None
    p_cores = logical - physical
    if p_cores <= 0 or logical in (physical, 2 * physical):  # no HT, or uniform HT
        return None
    return list(range(2 * p_cores))


def boost_process():
    proc = psutil.Process()
    cores = performance_cores()
    try:
        if cores:
            proc.cpu_affinity(cores)
        if sys.platform == "win32":
            proc.nice(psutil.HIGH_PRIORITY_CLASS)
        log.info("Pinned to CPUs %s with high priority", cores or "all")
    except (psutil.AccessDenied, OSError) as e:
        log.warning("Could not boost process: %s", e)
