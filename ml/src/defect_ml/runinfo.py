"""System state and a safety guard for long runs on the 8 GB Mac (read-only macOS commands).

Swap use and free disk are what went wrong on this machine before (PLAN.md Phase 0F, dev notes), so
long runs record them and a watchdog stops the run if either gets too tight.
"""

from __future__ import annotations

import os
import resource
import shutil
import subprocess
import sys
import threading
from datetime import UTC, datetime

DATA_VOLUME = "/System/Volumes/Data"  # the user-data volume on macOS; absent on Linux (CI)
GUARD_EXIT_CODE = 3


def shell(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return "unavailable"


def swap_used_mb() -> float | None:
    # Output looks like: "total = 3072.00M  used = 1919.88M  free = 1152.12M  (encrypted)"
    text = shell(["sysctl", "-n", "vm.swapusage"])
    try:
        return float(text.split("used =")[1].split()[0].rstrip("M"))
    except (IndexError, ValueError):
        return None


def free_disk_gib() -> float:
    """Free space on the data volume (macOS), or on the volume holding the working folder."""
    path = DATA_VOLUME if os.path.isdir(DATA_VOLUME) else os.getcwd()
    return shutil.disk_usage(path).free / 2**30


def rss_peak_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    unit = 2**20 if sys.platform == "darwin" else 2**10  # bytes on macOS, kibibytes on Linux
    return round(peak / unit, 1)


def system_state() -> dict[str, object]:
    memory = [x for x in shell(["memory_pressure"]).splitlines() if "free percentage" in x]
    return {
        "time_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "swap_used_mb": swap_used_mb(),
        "free_disk_gib": round(free_disk_gib(), 2),
        "load_average": [round(x, 2) for x in os.getloadavg()],
        "memory_free_percentage": memory[0].strip() if memory else "unavailable",
        "power": shell(["pmset", "-g", "batt"]).splitlines()[:2],
        "thermal": shell(["pmset", "-g", "therm"]).splitlines()[:3],
    }


def conditions_ok(max_swap_used_mb: float, min_free_disk_gib: float) -> str | None:
    """None if it is fine to start, otherwise a plain reason not to."""
    swap, free = swap_used_mb(), free_disk_gib()
    if swap is not None and swap >= max_swap_used_mb:
        return f"swap used is {swap:.0f} MB (limit {max_swap_used_mb:.0f} MB)"
    if free <= min_free_disk_gib:
        return f"free disk is {free:.1f} GiB (limit {min_free_disk_gib:.1f} GiB)"
    return None


class Watchdog(threading.Thread):
    """Stops the process if swap or disk pressure reaches the limits; records the extremes."""

    def __init__(self, max_swap_used_mb: float, min_free_disk_gib: float) -> None:
        super().__init__(daemon=True)
        self.max_swap_used_mb = max_swap_used_mb
        self.min_free_disk_gib = min_free_disk_gib
        self.max_swap_seen = swap_used_mb() or 0.0
        self.min_free_seen = free_disk_gib()
        self._stopped = threading.Event()

    def stop(self) -> None:
        self._stopped.set()

    def run(self) -> None:
        while not self._stopped.is_set():
            swap, free = swap_used_mb(), free_disk_gib()
            if swap is not None:
                self.max_swap_seen = max(self.max_swap_seen, swap)
            self.min_free_seen = min(self.min_free_seen, free)
            if (swap or 0) >= self.max_swap_used_mb or free <= self.min_free_disk_gib:
                sys.stderr.write(
                    f"GUARD: swap used {swap} MB, free disk {free:.1f} GiB; stopping the run\n"
                )
                sys.stderr.flush()
                os._exit(GUARD_EXIT_CODE)
            self._stopped.wait(2.0)
