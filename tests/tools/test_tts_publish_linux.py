from __future__ import annotations

import errno
import os
from pathlib import Path
import sys

import pytest

from tools import tts_publish


pytestmark = pytest.mark.linux_only


def test_linux_absent_uses_renameat2_noreplace(tmp_path: Path):
    if not sys.platform.startswith("linux"):
        pytest.skip("Linux runtime required")
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = os.open("source", os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=parent_fd)
        os.write(fd, b"linux")
        os.close(fd)
        tts_publish._rename_absent_linux(parent_fd, "source", parent_fd, "final")
        assert (tmp_path / "final").read_bytes() == b"linux"
    finally:
        os.close(parent_fd)


def test_linux_enosys_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    if not sys.platform.startswith("linux"):
        pytest.skip("Linux runtime required")
    monkeypatch.setattr(
        tts_publish,
        "_linux_renameat2",
        lambda *_args: (_ for _ in ()).throw(OSError(errno.ENOSYS, "unavailable")),
    )
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = os.open("source", os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=parent_fd)
        os.close(fd)
        with pytest.raises(OSError) as exc_info:
            tts_publish._rename_absent_linux(parent_fd, "source", parent_fd, "final")
        assert exc_info.value.errno == errno.ENOSYS
        assert not (tmp_path / "final").exists()
    finally:
        os.close(parent_fd)


def test_linux_existing_uses_replace(tmp_path: Path):
    if not sys.platform.startswith("linux"):
        pytest.skip("Linux runtime required")
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        source = os.open("source", os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=parent_fd)
        os.write(source, b"new")
        os.close(source)
        (tmp_path / "final").write_bytes(b"old")
        tts_publish._replace_existing(parent_fd, "source", parent_fd, "final")
        assert (tmp_path / "final").read_bytes() == b"new"
    finally:
        os.close(parent_fd)
