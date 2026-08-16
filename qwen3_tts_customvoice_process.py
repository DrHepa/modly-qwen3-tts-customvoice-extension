"""Minimal stdlib bootstrap for the Modly Qwen3-TTS CustomVoice PROCESS."""

from __future__ import annotations

import json
import os
import sys
from typing import TextIO


# These protocol names are declared here so static Modly validation can audit
# the actual entry boundary without importing the internal runtime.
MODLY_PROCESS_CONTRACT = (
    "stdin",
    "json",
    "progress",
    "log",
    "done",
    "error",
    "result",
    "filePath",
    "workspaceDir",
    "nodeId",
)
BOOTSTRAP_ERROR = (
    "[PROCESS_BOOTSTRAP_FAILED] Qwen3-TTS CustomVoice initialization failed. "
    "Run Repair and try again."
)


def _emit_bootstrap_error(channel: TextIO) -> None:
    payload = {"type": "error", "message": BOOTSTRAP_ERROR}
    channel.write(json.dumps(payload, ensure_ascii=True) + "\n")
    channel.flush()


def _main() -> int:
    protocol_fd: int | None = None
    channel: TextIO | None = None
    try:
        # Duplicate the host stdout first. Native code imported later cannot
        # discover this descriptor through ordinary writes to fd 1 or fd 2.
        protocol_fd = os.dup(1)
        null_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(null_fd, 1)
            os.dup2(null_fd, 2)
        finally:
            os.close(null_fd)
        channel = os.fdopen(
            protocol_fd,
            "w",
            encoding="utf-8",
            errors="strict",
            buffering=1,
            newline="\n",
        )
        protocol_fd = None

        # Import only after both public OS descriptors are isolated. This also
        # isolates native writes performed during third-party module imports.
        from qwen3_tts_modly.process_runtime import run_protocol

        return run_protocol(sys.stdin, channel)
    except BaseException:
        if channel is not None:
            try:
                _emit_bootstrap_error(channel)
            except BaseException:
                pass
        elif protocol_fd is not None:
            try:
                encoded = (
                    json.dumps({"type": "error", "message": BOOTSTRAP_ERROR}, ensure_ascii=True)
                    + "\n"
                ).encode("utf-8")
                os.write(protocol_fd, encoded)
            except BaseException:
                pass
        return 1
    finally:
        if channel is not None:
            try:
                channel.close()
            except BaseException:
                pass
        elif protocol_fd is not None:
            try:
                os.close(protocol_fd)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(_main())
