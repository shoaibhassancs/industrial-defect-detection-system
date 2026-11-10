"""Tests for defect_ml.runinfo: it must also work where the macOS-only parts do not exist (CI runs
on Linux, where /System/Volumes/Data, sysctl, pmset and memory_pressure are all missing)."""

from __future__ import annotations

import sys

import pytest
from defect_ml import runinfo


def test_free_disk_falls_back_to_the_working_folder_when_the_mac_volume_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runinfo, "DATA_VOLUME", "/nonexistent/volume")
    assert runinfo.free_disk_gib() > 0
    assert runinfo.conditions_ok(max_swap_used_mb=1e9, min_free_disk_gib=0) is None


def test_missing_mac_commands_give_unavailable_or_none_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_such_command(cmd: list[str], **_kwargs: object) -> object:
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(runinfo.subprocess, "run", no_such_command)
    monkeypatch.setattr(runinfo, "DATA_VOLUME", "/nonexistent/volume")
    assert runinfo.shell(["sysctl"]) == "unavailable"
    assert runinfo.swap_used_mb() is None
    state = runinfo.system_state()
    assert state["swap_used_mb"] is None and state["free_disk_gib"] > 0
    assert state["memory_free_percentage"] == "unavailable"
    # an unknown swap value does not stop a run; the disk limit still does
    assert runinfo.conditions_ok(max_swap_used_mb=1, min_free_disk_gib=0) is None
    assert "free disk" in runinfo.conditions_ok(max_swap_used_mb=1, min_free_disk_gib=1e9)


def test_peak_memory_uses_the_unit_of_the_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    class Usage:
        ru_maxrss = 512 * 2**20  # 512 MB if bytes (macOS), 512 GiB if kibibytes (Linux)

    monkeypatch.setattr(runinfo.resource, "getrusage", lambda _who: Usage())
    monkeypatch.setattr(sys, "platform", "darwin")
    assert runinfo.rss_peak_mb() == 512.0
    monkeypatch.setattr(sys, "platform", "linux")
    assert runinfo.rss_peak_mb() == 512 * 1024.0
