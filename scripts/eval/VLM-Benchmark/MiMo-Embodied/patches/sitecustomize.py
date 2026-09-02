"""
Workaround for psutil race condition when reading /proc/meminfo
in containerized environments (Docker/Kubernetes) with concurrent processes.

Linux kernel's procfs can return truncated data under cgroup virtualization
when multiple processes read /proc/meminfo simultaneously, causing psutil
to crash with `IndexError: list index out of range`.

This patch adds retry logic and safe parsing as a fallback.

Reference: https://github.com/giampaolo/psutil/issues/2483

Note: This issue does NOT occur on bare-metal machines.
"""

import psutil
import psutil._pslinux as _pslinux

_original_virtual_memory = _pslinux.virtual_memory


def _patched_virtual_memory():
    # Retry a few times — the race is transient
    for attempt in range(5):
        try:
            return _original_virtual_memory()
        except (IndexError, ValueError):
            import time
            time.sleep(0.1 * (attempt + 1))

    # Fallback: manually parse with safety checks
    mems = {}
    with open("/proc/meminfo", "rb") as f:
        for line in f:
            fields = line.split()
            if len(fields) >= 2:
                try:
                    mems[fields[0]] = int(fields[1]) * 1024
                except ValueError:
                    continue

    total = mems.get(b"MemTotal:", 0)
    free = mems.get(b"MemFree:", 0)
    buffers = mems.get(b"Buffers:", 0)
    cached = mems.get(b"Cached:", 0)
    shared = mems.get(b"Shmem:", 0)
    sreclaimable = mems.get(b"SReclaimable:", 0)
    available = mems.get(b"MemAvailable:", 0)
    used = total - free - buffers - cached
    if used < 0:
        used = total - free
    percent = (used / total * 100.0) if total > 0 else 0.0

    from collections import namedtuple
    svmem = namedtuple('svmem', [
        'total', 'available', 'percent', 'used', 'free',
        'active', 'inactive', 'buffers', 'cached', 'shared', 'slab'
    ])

    return svmem(
        total, available, percent, used, free,
        0, 0, buffers, cached, shared, sreclaimable,
    )


_pslinux.virtual_memory = _patched_virtual_memory