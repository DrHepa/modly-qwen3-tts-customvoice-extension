"""Install and verify the isolated Qwen3-TTS CustomVoice PROCESS runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
from typing import Any, Sequence


ROOT = Path(os.path.normpath(os.path.abspath(__file__))).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen3_tts_modly.assets import ensure_snapshot
from qwen3_tts_modly.constants import EXTENSION_ID, NODE_ID, STATE_FILENAME
from qwen3_tts_modly.paths import (
    SETUP_MODELS_PAYLOAD_KEYS,
    bind_extension,
    native_directory_path,
    owned_model_directory,
    require_storage_disjoint,
    resolve_models_root,
)
from qwen3_tts_modly.setup_support import (
    SUPPORTED_PYTHON_MINORS,
    UTILITY_TIMEOUT_SECONDS,
    VENV_BACKUP_NAME,
    VENV_STAGING_NAME,
    create_or_reuse_venv,
    install_dependencies,
    select_platform_flavor,
    validate_venv_interpreter,
    validate_running_platform,
    verify_runtime,
)
from qwen3_tts_modly.state import invalidate_setup_state, write_setup_state


class SetupFailure(RuntimeError):
    """A stable setup invocation failure."""

    def __init__(self, code: str, public_message: str) -> None:
        self.code = code
        self.public_message = public_message
        super().__init__(f"{code}: {public_message}")


def log(message: str) -> None:
    print(f"[Qwen3-TTS setup] {message}", flush=True)


def _raise_setup_interrupted(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


def _install_termination_handlers() -> list[tuple[int, Any]]:
    """Map catchable host termination signals to the public interruption path."""

    previous_handlers: list[tuple[int, Any]] = []
    for name in ("SIGTERM", "SIGBREAK"):
        signum = getattr(signal, name, None)
        if not isinstance(signum, int):
            continue
        try:
            previous = signal.signal(signum, _raise_setup_interrupted)
        except (OSError, RuntimeError, ValueError):
            continue
        previous_handlers.append((signum, previous))
    return previous_handlers


def _restore_termination_handlers(previous_handlers: list[tuple[int, Any]]) -> None:
    for signum, previous in reversed(previous_handlers):
        try:
            signal.signal(signum, previous)
        except (OSError, RuntimeError, ValueError):
            pass


def parse_setup_args(argv: Sequence[str]) -> dict[str, Any]:
    """Accept current one-JSON setup and a bounded legacy positional shape."""

    if len(argv) == 2:
        try:
            payload = json.loads(argv[1])
        except json.JSONDecodeError as exc:
            raise SetupFailure("SETUP_PAYLOAD_INVALID", "setup JSON is invalid") from exc
        if not isinstance(payload, dict):
            raise SetupFailure("SETUP_PAYLOAD_INVALID", "setup JSON must contain an object")
        return payload
    if len(argv) >= 4:
        if len(argv) not in {4, 5}:
            raise SetupFailure(
                "SETUP_PAYLOAD_INVALID",
                "legacy setup requires <python_exe> <ext_dir> <gpu_sm> [cuda_version]",
            )
        try:
            gpu_sm = int(argv[3])
            cuda_version = int(argv[4]) if len(argv) == 5 else 0
        except ValueError as exc:
            raise SetupFailure(
                "SETUP_PAYLOAD_INVALID", "legacy gpu_sm and cuda_version must be integers"
            ) from exc
        payload: dict[str, Any] = {
            "python_exe": argv[1],
            "ext_dir": argv[2],
            "gpu_sm": gpu_sm,
            "cuda_version": cuda_version,
            "accelerator": "cuda" if gpu_sm else "cpu",
            "platform": sys.platform,
            "arch": platform.machine(),
            "_legacy_setup": True,
        }
        return payload
    raise SetupFailure(
        "SETUP_PAYLOAD_INVALID",
        "expected one Modly JSON argument or legacy arguments <python_exe> <ext_dir> <gpu_sm> [cuda_version]",
    )


def verify_python_runtime(python: Path, extension_dir: Path) -> tuple[int, int]:
    expected_venv = extension_dir / "venv"
    try:
        identity = validate_venv_interpreter(python, expected_venv, sys.platform)
        subprocess.run(
            [str(python), "-m", "pip", "--version"],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=UTILITY_TIMEOUT_SECONDS,
        )
        version = tuple(int(part) for part in identity.python_minor.split(".", 1))
    except SetupFailure:
        raise
    except (OSError, subprocess.SubprocessError, TypeError, ValueError) as exc:
        raise SetupFailure(
            "SETUP_VENV_INVALID", "runtime Python could not be verified; run Repair again"
        ) from exc
    if version not in SUPPORTED_PYTHON_MINORS:
        raise SetupFailure(
            "SETUP_PYTHON_UNSUPPORTED", "this release requires Modly Python 3.11 or 3.12"
        )
    return version


def validate_setup_storage_disjoint(
    models_root: Path,
    extension_dir: Path,
    platform_name: str,
) -> Path:
    """Validate every setup-owned storage boundary before filesystem mutation."""

    owned_model_root = models_root / EXTENSION_ID / NODE_ID
    venv = extension_dir / "venv"
    mutable_extension_paths = (
        extension_dir,
        venv,
        extension_dir / VENV_STAGING_NAME,
        extension_dir / VENV_BACKUP_NAME,
        venv / STATE_FILENAME,
        extension_dir / ".modly-local",
    )
    require_storage_disjoint(
        (models_root, owned_model_root),
        mutable_extension_paths,
        platform_name,
        code="SETUP_STORAGE_OVERLAP",
        public_message=(
            "configured model storage overlaps extension-managed mutable storage; "
            "choose a separate models directory"
        ),
    )
    return owned_model_root


def run_setup(context: dict[str, Any]) -> Path:
    log("Starting Install/Repair")
    binding = bind_extension(context, ROOT)
    flavor = select_platform_flavor(context)
    validate_running_platform(flavor)
    models_root = resolve_models_root(
        context,
        binding.extension_dir,
        ROOT,
        flavor.system,
        payload_keys=SETUP_MODELS_PAYLOAD_KEYS,
        require_existing=False,
    )
    validate_setup_storage_disjoint(
        models_root,
        binding.extension_dir,
        flavor.system,
    )
    models_root = native_directory_path(
        str(models_root),
        "resolved models directory",
        flavor.system,
        must_exist=False,
        create=True,
    )
    model_dir = owned_model_directory(models_root, create=True)
    invalidate_setup_state(binding.extension_dir)

    log(f"Selected {flavor.system}/{flavor.arch} {flavor.torch_variant} dependency plan")
    python = create_or_reuse_venv(
        binding.bootstrap_python,
        binding.extension_dir,
        flavor.system,
        log,
    )
    verify_python_runtime(python, binding.extension_dir)
    # install_dependencies ends with a checked ``python -m pip check``.
    install_dependencies(python, ROOT, flavor, log)
    health_record = verify_runtime(python, flavor, log)

    log("Provisioning the pinned public model snapshot")
    installed_model_dir = ensure_snapshot(model_dir, log=log)
    write_setup_state(binding.extension_dir, flavor, health_record)
    log("Install/Repair completed with verified local assets")
    return installed_model_dir


def _public_failure(exc: BaseException) -> tuple[str, str]:
    code = getattr(exc, "code", None)
    message = getattr(exc, "public_message", None)
    if isinstance(code, str) and isinstance(message, str):
        return code, message
    if isinstance(exc, KeyboardInterrupt):
        return "SETUP_INTERRUPTED", "setup was interrupted; run Repair again"
    return "SETUP_UNEXPECTED", "setup failed unexpectedly; run Repair and consult Modly setup logs"


def cli(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv if argv is None else argv)
    previous_handlers = _install_termination_handlers()
    try:
        context = parse_setup_args(arguments)
        run_setup(context)
        return 0
    except BaseException as exc:
        code, message = _public_failure(exc)
        print(
            f"[Qwen3-TTS setup] ERROR [{code}]: {message}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    finally:
        _restore_termination_handlers(previous_handlers)


if __name__ == "__main__":
    raise SystemExit(cli())
