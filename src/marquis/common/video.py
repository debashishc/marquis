"""Video ID normalization, local path resolution, and shard materialization.

MicroVENT ships videos as a WebDataset: ``videos/shard_NNNNNN.tar`` shards plus a
``videos/catalog.csv`` that maps each ``chunk_id`` to its ``shard_index``. The
extraction / generation CLIs, however, consume loose ``{video_id}.mp4`` files
under a ``video_root`` (``data.video_root`` / ``$MAGMAR_VIDEO_ROOT``).

:class:`TarVideoReader` and :func:`materialize_videos` bridge that gap. They read
each video straight out of its shard with ``tarfile.extractfile`` -- the archive
is never fully unpacked, only the requested member is streamed to disk -- so the
same preprocessing scales from the quicktest's handful of clips to the full
corpus. Videos are grouped by shard and written shard-by-shard, so at most one
tar handle is open at a time and each shard is read once in catalog order.
"""

from __future__ import annotations

import csv
import fnmatch
import shutil
import subprocess
import tarfile
import time
from collections.abc import Iterable
from pathlib import Path

from marquis.common.contracts import (
    DEFAULT_VIDEO_SHARDS_ROOT,
    normalize_video_id,
    resolve_video_path,
)

__all__ = [
    "DEFAULT_VIDEO_SHARDS_ROOT",
    "normalize_video_id",
    "resolve_video_path",
    "TarVideoReader",
    "materialize_videos",
    "materialize_audio",
]


class TarVideoReader:
    """Stream per-chunk ``.mp4`` bytes out of the videos WebDataset shards.

    Mirrors the RoutIR ``video`` view resolver: ``catalog.csv`` maps
    ``chunk_id`` -> ``shard_index``, ``shard_{shard:06d}.tar`` names the shard,
    and the in-shard member is ``{chunk_id}.mp4``. The catalog is read once and
    cached; shards are opened on demand by the caller (see
    :func:`materialize_videos`) so a large run never holds many handles open.
    """

    def __init__(self, dataset_root: Path | str, subdir: str = "videos", ext: str = ".mp4"):
        self.videos_dir = Path(dataset_root) / subdir
        self.ext = ext
        self.catalog = self.videos_dir / "catalog.csv"
        self.tar_template = str(self.videos_dir / "shard_{shard:06d}.tar")
        self._shard_of: dict[str, int] | None = None

    def _shard_index(self) -> dict[str, int]:
        if self._shard_of is None:
            with open(self.catalog, newline="", encoding="utf-8") as f:
                self._shard_of = {
                    row["chunk_id"]: int(row["shard_index"]) for row in csv.DictReader(f)
                }
        return self._shard_of

    def shard_for(self, chunk_id: str) -> int | None:
        return self._shard_index().get(chunk_id)

    def group_by_shard(
        self, chunk_ids: Iterable[str]
    ) -> tuple[dict[int, list[str]], list[str]]:
        """Bucket ``chunk_ids`` by their shard; return (groups, not-in-catalog)."""
        groups: dict[int, list[str]] = {}
        missing: list[str] = []
        for cid in chunk_ids:
            shard = self.shard_for(cid)
            if shard is None:
                missing.append(cid)
            else:
                groups.setdefault(shard, []).append(cid)
        return groups, missing

    def open_shard(self, shard: int) -> tarfile.TarFile:
        return tarfile.open(self.tar_template.format(shard=shard), "r")

    def _member(self, tar: tarfile.TarFile, chunk_id: str) -> tarfile.TarInfo:
        member = f"{chunk_id}{self.ext}"
        try:
            return tar.getmember(member)
        except KeyError:
            # Fall back to a glob if the member carries a path prefix in the tar.
            names = [n for n in tar.getnames() if fnmatch.fnmatch(n, f"*{chunk_id}{self.ext}")]
            if not names:
                raise KeyError(f"{member} not found in {tar.name}") from None
            return tar.getmember(names[0])

    def extract_to(self, tar: tarfile.TarFile, chunk_id: str, dest: Path) -> int:
        """Stream one member's bytes to ``dest`` (no full unpack).

        Writes to a ``.part`` sibling and renames on success so an interrupted
        run never leaves a truncated file that a later run would mistake for a
        completed download. Returns the number of bytes written.
        """
        info = self._member(tar, chunk_id)
        tmp = dest.with_name(dest.name + ".part")
        try:
            with tar.extractfile(info) as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)
            tmp.replace(dest)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return info.size


def materialize_videos(
    video_ids: Iterable[str],
    video_root: Path | str,
    *,
    dataset_root: Path | str = DEFAULT_VIDEO_SHARDS_ROOT,
    subdir: str = "videos",
    ext: str = ".mp4",
    force: bool = False,
    verbose: bool = True,
) -> dict:
    """Materialize ``{chunk_id}{ext}`` files into ``video_root`` from tar shards.

    Reads each chunk's bytes straight out of its WebDataset shard
    (``<dataset_root>/<subdir>/shard_*.tar``) and streams them to disk. Chunks are
    grouped by shard so every shard is opened once and read in catalog order --
    this bounds open file handles to one and keeps reads sequential, which is what
    makes it safe to point at the full corpus. ``subdir``/``ext`` select the media
    view (``videos``/``.mp4`` for video, ``audio``/``.m4a`` for audio).

    Returns a summary dict ``{"written", "skipped", "missing", "total"}`` where
    ``missing`` lists ids absent from the catalog or their shard.
    """
    video_ids = list(dict.fromkeys(video_ids))  # de-dup, preserve order
    video_root = Path(video_root)
    video_root.mkdir(parents=True, exist_ok=True)

    reader = TarVideoReader(dataset_root, subdir, ext)
    groups, missing = reader.group_by_shard(video_ids)

    written = skipped = 0
    t0 = time.time()
    for shard in sorted(groups):
        pending: list[tuple[str, Path]] = []
        for cid in groups[shard]:
            dest = video_root / f"{cid}{ext}"
            if dest.exists() and not force:
                skipped += 1
                if verbose:
                    print(f"  [have] {cid}: {dest.stat().st_size / 1e6:.1f} MB")
            else:
                pending.append((cid, dest))
        if not pending:
            continue
        with reader.open_shard(shard) as tar:
            for cid, dest in pending:
                try:
                    size = reader.extract_to(tar, cid, dest)
                except KeyError as exc:
                    missing.append(cid)
                    if verbose:
                        print(f"  [skip] {cid}: {exc}")
                    continue
                written += 1
                if verbose:
                    print(f"  [ok]   {cid}: {size / 1e6:.1f} MB (shard {shard})")

    if verbose:
        print(
            f"[ok] {written} new / {skipped} present / {len(missing)} missing "
            f"of {len(video_ids)} in {time.time() - t0:.1f}s -> {video_root}"
        )
    return {
        "written": written,
        "skipped": skipped,
        "missing": missing,
        "total": len(video_ids),
    }


def write_silent_audio(
    dest: Path | str, *, duration_sec: float = 1.0, sample_rate: int = 16000
) -> bool:
    """Write a valid *silent* audio file at ``dest`` via ffmpeg.

    Used as a stand-in for chunks whose source clip has no audio track: a real,
    decodable container (so Whisper reads it and returns an empty transcript)
    rather than a 0-byte file (which ffmpeg/Whisper would choke on). Writes to a
    ``.part`` sibling and renames on success. Returns True on success, False if
    ffmpeg is unavailable or fails.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    cmd = [
        "ffmpeg", "-nostdin", "-y",
        "-f", "lavfi", "-i", f"anullsrc=channel_layout=mono:sample_rate={sample_rate}",
        "-t", str(duration_sec), "-c:a", "aac",
        "-f", "ipod", str(tmp),  # force m4a muxer: the .part tmp name has no usable extension
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tmp.replace(dest)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        tmp.unlink(missing_ok=True)
        return False


def _read_audio_catalog(dataset_root: Path | str) -> dict[str, dict]:
    """Map ``chunk_id`` -> catalog row from ``<dataset_root>/audio/catalog.csv``.

    Returns an empty dict if the catalog is absent. Used to tell expected-silent
    chunks (``has_audio == False``) apart from genuinely missing audio.
    """
    catalog = Path(dataset_root) / "audio" / "catalog.csv"
    try:
        with open(catalog, newline="", encoding="utf-8") as f:
            return {row["chunk_id"]: row for row in csv.DictReader(f)}
    except FileNotFoundError:
        return {}


def materialize_audio(
    video_ids: Iterable[str],
    audio_root: Path | str,
    *,
    dataset_root: Path | str = DEFAULT_VIDEO_SHARDS_ROOT,
    ext: str = ".m4a",
    force: bool = False,
    verbose: bool = True,
    silent_for_missing: bool = True,
) -> dict:
    """Materialize loose ``{chunk_id}{ext}`` audio files out of the audio shards.

    Thin wrapper over :func:`materialize_videos` pointed at the ``audio`` view of
    the same MicroVENT WebDataset; QA's Whisper transcribes the resulting files.

    Some clips have no audio track (``has_audio == False`` in the audio catalog),
    so no ``.m4a`` is packed in the shard. When ``silent_for_missing`` is set,
    those expected-silent chunks get a valid silent placeholder written instead
    of being reported missing, so the run doesn't fail on them. Chunks that are
    *genuinely* missing (catalog says they should have audio, or aren't in the
    catalog at all) stay in ``missing``.
    """
    audio_root = Path(audio_root)
    summary = materialize_videos(
        video_ids,
        audio_root,
        dataset_root=dataset_root,
        subdir="audio",
        ext=ext,
        force=force,
        verbose=verbose,
    )
    if not (silent_for_missing and summary["missing"]):
        return summary

    catalog = _read_audio_catalog(dataset_root)
    silent: list[str] = []
    still_missing: list[str] = []
    for cid in summary["missing"]:
        row = catalog.get(cid)
        has_audio = str(row.get("has_audio", "")).strip().lower() if row else ""
        expected_silent = row is not None and has_audio == "false"
        if not expected_silent:
            still_missing.append(cid)
            continue
        try:
            duration = float(row.get("duration_sec") or 0) or 1.0
        except ValueError:
            duration = 1.0
        dest = audio_root / f"{cid}{ext}"
        if write_silent_audio(dest, duration_sec=min(duration, 1.0)):
            silent.append(cid)
        else:
            still_missing.append(cid)

    if verbose and silent:
        print(f"[ok] {len(silent)} silent placeholders written (no audio track) -> {audio_root}")
    summary["missing"] = still_missing
    summary["silent"] = silent
    return summary
