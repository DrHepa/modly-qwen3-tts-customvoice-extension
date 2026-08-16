"""Protocol-safe capture and privacy-minimal workspace diagnostics."""

from __future__ import annotations

import io
import os
from pathlib import Path
import re
import stat
import uuid

from .constants import DIAGNOSTICS_DIRECTORY_NAME, OUTPUT_RELATIVE_DIRECTORY


MAX_CAPTURE_CHARACTERS = 32 * 1024
MAX_DIAGNOSTIC_BYTES = 4 * 1024
WINDOWS_REPARSE_ATTRIBUTE = 0x400
SAFE_FIELD = re.compile(r"^[A-Za-z0-9_ .,:;()'-]{1,240}$")


class BoundedTextCapture(io.TextIOBase):
    """Discard third-party text while bounding the amount counted in memory."""

    def __init__(self, limit: int = MAX_CAPTURE_CHARACTERS) -> None:
        super().__init__()
        self._limit = limit
        self._count = 0
        self._truncated = False

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        length = len(str(value))
        accepted = min(max(self._limit - self._count, 0), length)
        self._count += accepted
        self._truncated = self._truncated or accepted < length
        return length

    @property
    def captured_characters(self) -> int:
        return self._count

    @property
    def truncated(self) -> bool:
        return self._truncated

    def getvalue(self) -> str:
        """Compatibility helper that deliberately never returns captured content."""

        return ""


def _is_alias(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & WINDOWS_REPARSE_ATTRIBUTE
    )


def ensure_workspace_subdirectory(workspace_dir: Path, parts: tuple[str, ...]) -> tuple[Path, Path]:
    workspace_root = workspace_dir.resolve(strict=True)
    if not workspace_root.is_dir():
        raise OSError("workspace root is not a directory")
    current = workspace_dir
    for part in parts:
        if part in {"", ".", ".."} or Path(part).name != part:
            raise OSError("unsafe workspace directory segment")
        current = current / part
        if current.exists() or current.is_symlink():
            if _is_alias(current) or not current.is_dir():
                raise OSError("workspace output path contains an alias or non-directory")
        else:
            current.mkdir()
        resolved = current.resolve(strict=True)
        if workspace_root not in resolved.parents:
            raise OSError("workspace output directory escapes the workspace")
    return workspace_root, current.resolve(strict=True)


def _safe_field(value: str, fallback: str) -> str:
    return value if SAFE_FIELD.fullmatch(value) else fallback


def write_diagnostic(
    workspace_dir: Path | None,
    run_id: str,
    *,
    code: str,
    stage: str,
    action: str,
) -> str | None:
    """Write only allowlisted public fields and return a workspace-relative path."""

    if (
        workspace_dir is None
        or not workspace_dir.is_absolute()
        or not workspace_dir.is_dir()
        or not re.fullmatch(r"[a-f0-9]{32}", run_id)
    ):
        return None
    try:
        workspace_root, directory = ensure_workspace_subdirectory(
            workspace_dir,
            (*OUTPUT_RELATIVE_DIRECTORY, DIAGNOSTICS_DIRECTORY_NAME),
        )
        safe_code = _safe_field(code, "GENERATION_UNEXPECTED")
        safe_stage = _safe_field(stage, "generation")
        safe_action = _safe_field(action, "Run Repair and try again.")
        encoded = (
            "Qwen3-TTS CustomVoice diagnostic\n"
            f"run_id: {run_id}\n"
            f"code: {safe_code}\n"
            f"stage: {safe_stage}\n"
            f"action: {safe_action}\n"
        ).encode("ascii", errors="strict")
        if len(encoded) > MAX_DIAGNOSTIC_BYTES:
            return None
        destination = directory / f"{run_id}.log"
        temporary = directory / f".{run_id}.{uuid.uuid4().hex}.tmp"
        if destination.exists() or destination.is_symlink():
            return None
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if directory.resolve(strict=True) != destination.parent.resolve(strict=True):
            temporary.unlink(missing_ok=True)
            return None
        os.replace(temporary, destination)
        final = destination.resolve(strict=True)
        if workspace_root not in final.parents or not final.is_file():
            destination.unlink(missing_ok=True)
            return None
        return final.relative_to(workspace_root).as_posix()
    except (OSError, ValueError, UnicodeError):
        return None
