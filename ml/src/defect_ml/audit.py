"""Record which files the running process opens, as evidence rather than a promise.

A Python audit hook sees every ``open``, ``os.listdir`` and ``os.scandir`` event, whoever makes
it. ``OpenAudit`` collects them while it is active, so a run can show that it opened exactly the
files it should have inside the dataset folder. (Audit hooks cannot be removed once installed; this
one does nothing unless an ``OpenAudit`` is active.)
"""

from __future__ import annotations

import sys
from pathlib import Path

_ACTIVE: list[OpenAudit] = []
_EVENTS_OF_INTEREST = ("open", "os.listdir", "os.scandir")


def _hook(event: str, args: tuple) -> None:
    if not _ACTIVE or event not in _EVENTS_OF_INTEREST:
        return
    target = args[0] if args else None
    if isinstance(target, bytes):
        target = target.decode(errors="replace")
    if isinstance(target, str | Path):
        for audit in _ACTIVE:
            audit.events.append((event, str(target)))


sys.addaudithook(_hook)


class OpenAudit:
    """Context manager that collects ``(event, path)`` pairs: file opens and directory listings."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def __enter__(self) -> OpenAudit:
        _ACTIVE.append(self)
        return self

    def __exit__(self, *exc_info: object) -> None:
        _ACTIVE.remove(self)

    def inside(self, directory: str | Path) -> tuple[list[Path], list[str]]:
        """Return (files opened, directories listed) whose path lies inside ``directory``."""
        root = str(Path(directory).resolve())
        opened: list[Path] = []
        listed: list[str] = []
        for event, path in self.events:
            if root not in path:
                continue
            if event == "open":
                opened.append(Path(path))
            else:
                listed.append(path)
        return opened, listed
