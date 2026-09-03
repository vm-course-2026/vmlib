#!/usr/bin/env python3
"""Build and verify reproducible ``vmlib-course`` release artifacts.

The PEP 517 backend is allowed to create its usual sdist first.  This script
then rewrites the archive without extracting it, rejects unsafe members and
normalizes every tar header.  Two independent builds must have identical
SHA-256 digests before the generated ``dist/`` directory is replaced.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = PROJECT_ROOT / "dist"
SOURCE_DATE_EPOCH = 1_788_393_600  # 2026-09-03 00:00:00 UTC
MAX_MEMBER_SIZE = 100 * 1024 * 1024
MAX_ARCHIVE_SIZE = 200 * 1024 * 1024
MAX_MEMBERS = 10_000


def _safe_member_name(name: str) -> PurePosixPath:
    """Return a safe relative POSIX path or reject the archive member."""
    if not name or "\\" in name:
        raise ValueError(f"unsafe sdist member name: {name!r}")
    canonical_name = name.rstrip("/")
    raw_parts = canonical_name.split("/")
    path = PurePosixPath(canonical_name)
    if (
        not canonical_name
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in raw_parts)
    ):
        raise ValueError(f"unsafe sdist member path: {name!r}")
    return path


def _regular_mode(member: tarfile.TarInfo) -> int:
    """Keep only the portable executable bit; discard host-specific modes."""
    return 0o755 if member.mode & 0o111 else 0o644


def normalize_sdist(source: Path, destination: Path) -> None:
    """Write a deterministic, safe gzip-compressed PAX tar archive."""
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must be different files")

    members: list[tuple[str, bool, int, bytes]] = []
    names: set[str] = set()
    total_size = 0
    roots: set[str] = set()
    with tarfile.open(source, mode="r:gz") as archive:
        archive_members = archive.getmembers()
        if not archive_members or len(archive_members) > MAX_MEMBERS:
            raise ValueError("sdist has an invalid number of members")
        for member in archive_members:
            safe_path = _safe_member_name(member.name)
            name = safe_path.as_posix().rstrip("/")
            if name in names:
                raise ValueError(f"duplicate sdist member: {name!r}")
            names.add(name)
            roots.add(safe_path.parts[0])

            if member.isdir():
                payload = b""
                mode = 0o755
            elif member.isfile():
                if member.size < 0 or member.size > MAX_MEMBER_SIZE:
                    raise ValueError(f"invalid sdist member size: {member.name!r}")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"cannot read sdist member: {member.name!r}")
                payload = extracted.read(MAX_MEMBER_SIZE + 1)
                if len(payload) != member.size or len(payload) > MAX_MEMBER_SIZE:
                    raise ValueError(f"truncated or oversized member: {member.name!r}")
                total_size += len(payload)
                if total_size > MAX_ARCHIVE_SIZE:
                    raise ValueError("uncompressed sdist is too large")
                mode = _regular_mode(member)
            else:
                raise ValueError(
                    f"links and special files are forbidden in sdist: {member.name!r}"
                )
            members.append((name, member.isdir(), mode, payload))

    if len(roots) != 1:
        raise ValueError(f"sdist must contain exactly one top-level directory: {roots}")
    root = next(iter(roots))
    if not any(name == root and is_directory for name, is_directory, _, _ in members):
        raise ValueError("sdist top-level directory has no explicit directory member")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("wb") as raw_stream:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=raw_stream,
                mtime=SOURCE_DATE_EPOCH,
            ) as gzip_stream:
                with tarfile.open(
                    fileobj=gzip_stream,
                    mode="w",
                    format=tarfile.USTAR_FORMAT,
                ) as output:
                    for name, is_directory, mode, payload in sorted(
                        members, key=lambda item: item[0]
                    ):
                        info = tarfile.TarInfo(f"{name}/" if is_directory else name)
                        info.type = tarfile.DIRTYPE if is_directory else tarfile.REGTYPE
                        info.size = 0 if is_directory else len(payload)
                        info.mode = mode
                        info.mtime = SOURCE_DATE_EPOCH
                        info.uid = 0
                        info.gid = 0
                        info.uname = "root"
                        info.gname = "root"
                        info.pax_headers = {}
                        output.addfile(info, None if is_directory else io.BytesIO(payload))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def verify_normalized_sdist(path: Path) -> None:
    """Check all identity, timestamp, ordering and member-type invariants."""
    with path.open("rb") as stream:
        header = stream.read(10)
    if (
        len(header) != 10
        or header[:3] != b"\x1f\x8b\x08"
        or header[3] != 0
        or int.from_bytes(header[4:8], "little") != SOURCE_DATE_EPOCH
        or header[8:] != b"\x02\xff"
    ):
        raise ValueError("sdist has a non-canonical gzip header")
    names: list[str] = []
    roots: set[str] = set()
    root_directories: set[str] = set()
    total_size = 0
    with tarfile.open(path, mode="r:gz") as archive:
        archive_members = archive.getmembers()
        if not archive_members or len(archive_members) > MAX_MEMBERS:
            raise ValueError("sdist has an invalid number of members")
        for member in archive_members:
            safe_path = _safe_member_name(member.name)
            canonical_name = safe_path.as_posix().rstrip("/")
            names.append(canonical_name)
            roots.add(safe_path.parts[0])
            if member.isdir() and canonical_name == safe_path.parts[0]:
                root_directories.add(canonical_name)
            if not (member.isdir() or member.isfile()):
                raise ValueError(f"special sdist member: {member.name!r}")
            if member.size < 0 or member.size > MAX_MEMBER_SIZE:
                raise ValueError(f"invalid sdist member size: {member.name!r}")
            total_size += member.size
            if total_size > MAX_ARCHIVE_SIZE:
                raise ValueError("uncompressed sdist is too large")
            if (member.uid, member.gid, member.uname, member.gname) != (
                0,
                0,
                "root",
                "root",
            ):
                raise ValueError(f"non-normalized owner in sdist: {member.name!r}")
            if member.mtime != SOURCE_DATE_EPOCH:
                raise ValueError(f"non-normalized timestamp in sdist: {member.name!r}")
            if member.pax_headers:
                raise ValueError(f"PAX metadata is forbidden in sdist: {member.name!r}")
            expected_mode = 0o755 if member.isdir() or member.mode & 0o111 else 0o644
            if member.mode != expected_mode:
                raise ValueError(f"non-normalized mode in sdist: {member.name!r}")
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError("sdist member order is not canonical")
    if len(roots) != 1 or roots != root_directories:
        raise ValueError("sdist must have one explicit top-level directory")


def _artifacts(directory: Path) -> tuple[Path, Path]:
    wheels = sorted(directory.glob("*.whl"))
    sdists = sorted(directory.glob("*.tar.gz"))
    allowed = {*wheels, *sdists}
    unexpected = sorted(
        path.name for path in directory.iterdir() if path not in allowed
    )
    if len(wheels) != 1 or len(sdists) != 1 or unexpected:
        raise RuntimeError(
            "expected exactly one wheel and one sdist; "
            f"wheels={len(wheels)}, sdists={len(sdists)}, unexpected={unexpected}"
        )
    return wheels[0], sdists[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_once(output: Path) -> tuple[Path, Path]:
    environment = os.environ.copy()
    environment.update(
        {
            "LC_ALL": "C",
            "LANG": "C",
            "TZ": "UTC",
            "PYTHONHASHSEED": "0",
            "SOURCE_DATE_EPOCH": str(SOURCE_DATE_EPOCH),
        }
    )
    subprocess.run(
        [
            "uv",
            "build",
            "--clear",
            "--no-create-gitignore",
            "--out-dir",
            str(output),
            ".",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )
    wheel, sdist = _artifacts(output)
    normalized = output / f".{sdist.name}.normalized"
    normalize_sdist(sdist, normalized)
    os.replace(normalized, sdist)
    verify_normalized_sdist(sdist)
    return wheel, sdist


def _replace_dist(wheel: Path, sdist: Path) -> None:
    if DIST_DIR.is_symlink() or (DIST_DIR.exists() and not DIST_DIR.is_dir()):
        raise RuntimeError(f"refusing to replace unsafe dist path: {DIST_DIR}")
    if DIST_DIR.exists():
        unexpected = sorted(
            path.name
            for path in DIST_DIR.iterdir()
            if path.is_symlink()
            or not path.is_file()
            or not (path.name.endswith(".whl") or path.name.endswith(".tar.gz"))
        )
        if unexpected:
            raise RuntimeError(f"refusing to replace dist with unknown entries: {unexpected}")

    staging = Path(tempfile.mkdtemp(prefix=".dist-ready-", dir=PROJECT_ROOT))
    backup = PROJECT_ROOT / f".dist-backup-{os.getpid()}"
    try:
        shutil.copyfile(wheel, staging / wheel.name)
        shutil.copyfile(sdist, staging / sdist.name)
        if DIST_DIR.exists():
            if backup.exists():
                raise RuntimeError(f"refusing to overwrite backup path: {backup}")
            DIST_DIR.rename(backup)
        staging.rename(DIST_DIR)
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if not DIST_DIR.exists() and backup.exists():
            backup.rename(DIST_DIR)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def build_release() -> dict[str, str]:
    """Build twice, require byte-for-byte equality and publish to ``dist``."""
    with tempfile.TemporaryDirectory(prefix="vmlib-release-a-") as first_dir:
        with tempfile.TemporaryDirectory(prefix="vmlib-release-b-") as second_dir:
            first = _build_once(Path(first_dir))
            second = _build_once(Path(second_dir))
            comparisons = [
                (left.name, _sha256(left), right.name, _sha256(right))
                for left, right in zip(first, second, strict=True)
            ]
            if any(
                left_name != right_name or left_hash != right_hash
                for left_name, left_hash, right_name, right_hash in comparisons
            ):
                raise RuntimeError(f"release artifacts are not reproducible: {comparisons}")
            _replace_dist(*second)

    wheel, sdist = _artifacts(DIST_DIR)
    verify_normalized_sdist(sdist)
    return {wheel.name: _sha256(wheel), sdist.name: _sha256(sdist)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify the existing dist/sdist without rebuilding anything",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check_only:
        wheel, sdist = _artifacts(DIST_DIR)
        verify_normalized_sdist(sdist)
        hashes = {wheel.name: _sha256(wheel), sdist.name: _sha256(sdist)}
    else:
        hashes = build_release()
    for name, digest in sorted(hashes.items()):
        print(f"{digest}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
