from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "build_release.py"
SPEC = importlib.util.spec_from_file_location("vmlib_release_build", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def _raw_sdist(path: Path, *, reverse: bool, identity: int) -> None:
    entries = [
        ("vmlib_course-1.1.0", None),
        ("vmlib_course-1.1.0/README.md", b"hello\n"),
        ("vmlib_course-1.1.0/vmlib.py", b"VALUE = 1\n"),
    ]
    if reverse:
        entries.reverse()
    with tarfile.open(path, mode="w:gz") as archive:
        for name, payload in entries:
            info = tarfile.TarInfo(name)
            info.uid = identity
            info.gid = identity
            info.uname = f"user-{identity}"
            info.gname = f"group-{identity}"
            info.mtime = identity
            if payload is None:
                info.type = tarfile.DIRTYPE
                info.mode = 0o700
                archive.addfile(info)
            else:
                info.mode = 0o600
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))


def test_normalized_sdist_is_owner_free_and_reproducible(tmp_path: Path):
    first_raw = tmp_path / "first.tar.gz"
    second_raw = tmp_path / "second.tar.gz"
    first = tmp_path / "first-normalized.tar.gz"
    second = tmp_path / "second-normalized.tar.gz"
    _raw_sdist(first_raw, reverse=False, identity=501)
    _raw_sdist(second_raw, reverse=True, identity=1001)

    release.normalize_sdist(first_raw, first)
    release.normalize_sdist(second_raw, second)
    release.verify_normalized_sdist(first)
    release.verify_normalized_sdist(second)

    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(
        second.read_bytes()
    ).digest()
    assert int.from_bytes(first.read_bytes()[4:8], "little") == release.SOURCE_DATE_EPOCH
    with tarfile.open(first, mode="r:gz") as archive:
        assert all(
            (item.uid, item.gid, item.uname, item.gname) == (0, 0, "root", "root")
            for item in archive.getmembers()
        )


def test_normalizer_rejects_path_traversal(tmp_path: Path):
    source = tmp_path / "unsafe.tar.gz"
    with tarfile.open(source, mode="w:gz") as archive:
        info = tarfile.TarInfo("vmlib_course-1.1.0/../secret")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))

    with pytest.raises(ValueError, match="unsafe sdist member path"):
        release.normalize_sdist(source, tmp_path / "result.tar.gz")


def test_normalizer_rejects_links(tmp_path: Path):
    source = tmp_path / "link.tar.gz"
    with gzip.open(source, mode="wb") as stream:
        with tarfile.open(fileobj=stream, mode="w") as archive:
            root = tarfile.TarInfo("vmlib_course-1.1.0")
            root.type = tarfile.DIRTYPE
            archive.addfile(root)
            link = tarfile.TarInfo("vmlib_course-1.1.0/link")
            link.type = tarfile.SYMTYPE
            link.linkname = "/etc/passwd"
            archive.addfile(link)

    with pytest.raises(ValueError, match="links and special files"):
        release.normalize_sdist(source, tmp_path / "result.tar.gz")
