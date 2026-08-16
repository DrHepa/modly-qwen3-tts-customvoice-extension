"""Portable, pinned dependency installation and health verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import threading
from typing import Callable, Mapping

from .constants import (
    ALL_DISTRIBUTIONS,
    BUILD_DISTRIBUTIONS,
    RUNTIME_DISTRIBUTIONS,
    TORCH_VARIANTS,
)
from .paths import normalize_architecture, normalize_platform_name


LogFunction = Callable[[str], None]
NATIVE_BINARY_DISTRIBUTIONS = (
    "numpy",
    "scipy",
    "scikit-learn",
    "numba",
    "llvmlite",
    "onnxruntime",
    "soundfile",
    "soxr",
)
SUPPORTED_PYTHON_MINORS = ((3, 11), (3, 12))
WINDOWS_REPARSE_ATTRIBUTE = 0x400
VENV_STAGING_NAME = "venv.__modly_staging"
VENV_BACKUP_NAME = "venv.__modly_backup"
COMMAND_OUTPUT_LIMIT = 64 * 1024
COMMAND_TIMEOUT_SECONDS = 60 * 60
COMMAND_READ_SIZE = 8192
SAFE_COMMAND_STAGES = frozenset(
    {
        "Preparing isolated Python environment",
        "Pinning Python build tools",
        "Installing pinned PyTorch cpu runtime",
        "Installing pinned PyTorch cu128 runtime",
        "Installing pinned PyTorch cu130 runtime",
        "Installing pinned native wheels",
        "Installing the pinned Python SoX wrapper",
        "Installing the pinned Qwen3-TTS runtime",
        "Verifying dependency graph",
    }
)

CUSPARSELT_DISTRIBUTION = "nvidia-cusparselt-cu13"
CUSPARSELT_VERSION = "0.8.0"
CUSPARSELT_DIST_INFO = "nvidia_cusparselt_cu13-0.8.0.dist-info"
CUSPARSELT_PIP_CHECK_DIAGNOSTIC = (
    "nvidia-cusparselt-cu13 0.8.0 is not supported on this platform"
)
CUSPARSELT_SBSA_WHEEL_TAG = "Tag: py3-none-manylinux2014_sbsa"
CUSPARSELT_AUDIT_SUCCESS = b"CUSPARSELT_SBSA_VERIFIED\n"

HEALTH_SCHEMA = "modly.qwen3-tts.health.v1"
HEALTH_OUTPUT_LIMIT = 16 * 1024
HEALTH_TIMEOUT_SECONDS = 10 * 60
HEALTH_CATEGORIES = frozenset(
    {"cuda_oom", "import", "cuda_unavailable", "native_probe", "invalid_result"}
)
HEALTH_SUBSTAGES = frozenset(
    {
        "bootstrap",
        "platform_identity",
        "imports_core",
        "imports_numeric",
        "imports_audio",
        "imports_native",
        "dependency_versions",
        "python_version",
        "cuda_availability",
        "cuda_device",
        "float32_alloc",
        "float32_matmul",
        "float32_conv",
        "bf16_probe",
        "sdpa",
        "cuda_sync",
        "torchaudio",
        "onnx_cpu",
        "soundfile",
        "librosa",
        "result",
    }
)


class SetupSupportError(RuntimeError):
    """A stable-code dependency or platform setup failure."""

    def __init__(self, code: str, public_message: str) -> None:
        self.code = code
        self.public_message = public_message
        super().__init__(f"{code}: {public_message}")


@dataclass(frozen=True)
class PlatformFlavor:
    """The public platform and pinned PyTorch wheel flavor selected by setup."""

    system: str
    arch: str
    accelerator: str
    torch_variant: str
    expected_cuda: str | None

    def state_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class InterpreterIdentity:
    """The ABI-relevant identity used to decide whether a venv is reusable."""

    python_minor: str
    system: str
    arch: str
    implementation: str
    cache_tag: str
    soabi: str
    prefix: str

    def abi_signature(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.python_minor,
            self.system,
            self.arch,
            self.implementation,
            self.cache_tag,
            self.soabi,
        )


@dataclass(frozen=True)
class CommandOutcome:
    """Bounded private command output used only for safe classification."""

    returncode: int
    output: bytes
    truncated: bool = False
    timed_out: bool = False


def venv_python(venv: Path, platform_name: str | None = None) -> Path:
    platform_value = normalize_platform_name(platform_name or sys.platform)
    return venv / ("Scripts/python.exe" if platform_value == "win32" else "bin/python")


def _normalize_cuda_hint(value: object) -> str:
    if isinstance(value, bool):
        return ""
    raw = str(value or "").strip().casefold().replace("cuda", "").replace("cu", "")
    compact = raw.replace(".", "")
    if compact == "128":
        return "12.8"
    if compact == "130":
        return "13.0"
    return ""


def select_platform_flavor(context: Mapping[str, object]) -> PlatformFlavor:
    """Select a supported public wheel set solely from explicit setup metadata."""

    system = normalize_platform_name(context.get("platform"))
    arch = normalize_architecture(context.get("arch"))
    accelerator_raw = str(context.get("accelerator") or "").strip().casefold()
    accelerator = "cuda" if accelerator_raw in {"cuda", "nvidia"} else accelerator_raw

    if system not in {"linux", "win32"}:
        raise SetupSupportError(
            "SETUP_PLATFORM_UNSUPPORTED",
            "this release supports Windows x64 and Linux x64, with Linux ARM64 as a wheel-availability candidate",
        )
    if arch not in {"x64", "arm64"} or (system == "win32" and arch != "x64"):
        raise SetupSupportError(
            "SETUP_ARCH_UNSUPPORTED",
            "the requested operating-system and architecture combination has no supported wheel plan",
        )
    if accelerator == "cpu":
        variant = "cpu"
    elif accelerator == "cuda":
        cuda_hint = _normalize_cuda_hint(context.get("cuda_version"))
        if cuda_hint == "12.8":
            variant = "cu128"
        elif cuda_hint == "13.0":
            variant = "cu130"
        else:
            raise SetupSupportError(
                "SETUP_CUDA_UNSUPPORTED",
                "NVIDIA setup requires an explicit CUDA 12.8 or CUDA 13.0 metadata hint",
            )
    else:
        raise SetupSupportError(
            "SETUP_ACCELERATOR_UNSUPPORTED",
            "accelerator must explicitly be cpu or cuda",
        )
    torch_record = TORCH_VARIANTS[variant]
    return PlatformFlavor(
        system=system,
        arch=arch,
        accelerator=accelerator,
        torch_variant=variant,
        expected_cuda=torch_record["cuda"],
    )


def validate_running_platform(flavor: PlatformFlavor) -> None:
    """Reject metadata that would install wheels for a different running host."""

    if flavor.system != normalize_platform_name(sys.platform):
        raise SetupSupportError(
            "SETUP_PLATFORM_MISMATCH", "setup platform metadata does not match the running system"
        )
    if flavor.arch != normalize_architecture(platform.machine()):
        raise SetupSupportError(
            "SETUP_ARCH_MISMATCH", "setup architecture metadata does not match the running system"
        )


def expected_distributions(flavor: PlatformFlavor) -> dict[str, str]:
    torch_record = TORCH_VARIANTS[flavor.torch_variant]
    return {
        **ALL_DISTRIBUTIONS,
        "torch": str(torch_record["torch"]),
        "torchaudio": str(torch_record["torchaudio"]),
    }


def _safe_stage(stage: object) -> str:
    return stage if isinstance(stage, str) and stage in SAFE_COMMAND_STAGES else "Setup command"


def _kill_process(process: subprocess.Popen[bytes]) -> None:
    try:
        process.kill()
    except OSError:
        pass


def _bounded_command(
    command: list[str],
    *,
    output_limit: int = COMMAND_OUTPUT_LIMIT,
    timeout: int = COMMAND_TIMEOUT_SECONDS,
) -> CommandOutcome:
    """Capture a child process privately without allowing unbounded output."""

    if output_limit < 1 or timeout < 1:
        raise ValueError("command limits must be positive")
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if process.stdout is None:
        _kill_process(process)
        raise OSError("command output pipe is unavailable")

    output = bytearray()
    truncated = threading.Event()
    read_failed = threading.Event()

    def read_output() -> None:
        try:
            while True:
                chunk = process.stdout.read(COMMAND_READ_SIZE)
                if not chunk:
                    return
                remaining = output_limit - len(output)
                if remaining > 0:
                    output.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    truncated.set()
                    _kill_process(process)
                    return
        except (OSError, ValueError):
            read_failed.set()
            _kill_process(process)

    reader = threading.Thread(target=read_output, name="modly-setup-output", daemon=True)
    reader.start()
    timed_out = False
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process(process)
        try:
            returncode = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            returncode = -1

    reader.join(timeout=10)
    if reader.is_alive():
        truncated.set()
        _kill_process(process)
        try:
            process.stdout.close()
        except (OSError, ValueError):
            pass
        reader.join(timeout=1)
    if read_failed.is_set():
        truncated.set()
    normalized_returncode = returncode if isinstance(returncode, int) else -1
    return CommandOutcome(
        returncode=normalized_returncode,
        output=bytes(output),
        truncated=truncated.is_set(),
        timed_out=timed_out,
    )


def _classify_command_failure(output: bytes) -> tuple[str, str]:
    """Classify known pip failures without returning any captured text."""

    decoded = output.decode("utf-8", errors="replace").casefold()
    if any(fragment in decoded for fragment in ("no space left on device", "errno 28", "disk quota exceeded")):
        return (
            "SETUP_STORAGE_FULL",
            "dependency setup ran out of storage; free space and run Repair",
        )
    if any(
        fragment in decoded
        for fragment in (
            "resolutionimpossible",
            "conflicting dependencies",
            "dependency conflict",
            "cannot install",
        )
    ):
        return (
            "SETUP_DEPENDENCY_CONFLICT",
            "dependency constraints are incompatible; update the extension and run Repair",
        )
    if any(
        fragment in decoded
        for fragment in (
            "could not fetch url",
            "connection reset",
            "connectionerror",
            "max retries exceeded",
            "network is unreachable",
            "name or service not known",
            "temporary failure in name resolution",
            "readtimeout",
        )
    ):
        return (
            "SETUP_NETWORK_FAILED",
            "dependency download failed; verify network access and run Repair",
        )
    if any(
        fragment in decoded
        for fragment in (
            "no matching distribution found",
            "could not find a version that satisfies the requirement",
            "is not a supported wheel on this platform",
            "is not supported on this platform",
        )
    ):
        return (
            "SETUP_WHEEL_UNAVAILABLE",
            "a required binary wheel is unavailable for this platform; run Repair after verifying platform support",
        )
    return "SETUP_COMMAND_FAILED", "setup command did not complete; run Repair"


def _command_failure(stage: object, outcome: CommandOutcome) -> SetupSupportError:
    safe_stage = _safe_stage(stage)
    code, action = _classify_command_failure(outcome.output)
    return SetupSupportError(
        code,
        f"{safe_stage} failed with exit code {outcome.returncode}; {action}",
    )


def run_checked(command: list[str], stage: str, log: LogFunction) -> None:
    """Run with bounded private output and emit only classified public diagnostics."""

    safe_stage = _safe_stage(stage)
    log(safe_stage)
    try:
        outcome = _bounded_command(command)
    except OSError as exc:
        raise SetupSupportError(
            "SETUP_COMMAND_START_FAILED", f"{safe_stage} could not start; run Repair again"
        ) from exc
    if outcome.returncode != 0 or outcome.truncated or outcome.timed_out:
        raise _command_failure(safe_stage, outcome)


def interpreter_identity(python: Path) -> InterpreterIdentity:
    """Probe a Python in isolated mode without inherited Python/user-site configuration."""

    script = (
        "import json,os,platform,sys,sysconfig;"
        "print(json.dumps({'pythonMinor':f'{sys.version_info.major}.{sys.version_info.minor}',"
        "'system':sys.platform,'arch':platform.machine(),"
        "'implementation':sys.implementation.name,"
        "'cacheTag':sys.implementation.cache_tag or '',"
        "'soabi':sysconfig.get_config_var('SOABI') or '',"
        "'prefix':os.path.normcase(os.path.realpath(sys.prefix))},sort_keys=True))"
    )
    try:
        completed = subprocess.run(
            [str(python), "-I", "-c", script],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=20,
        )
        if len(completed.stdout) > 4096:
            raise ValueError("identity output is too large")
        payload = json.loads(completed.stdout)
        minor = str(payload["pythonMinor"])
        system = normalize_platform_name(payload["system"])
        arch = normalize_architecture(payload["arch"])
        implementation = str(payload["implementation"])
        cache_tag = str(payload["cacheTag"])
        soabi = str(payload["soabi"])
        raw_prefix = str(payload["prefix"])
        if not os.path.isabs(raw_prefix):
            raise ValueError("identity prefix is not absolute")
        prefix = os.path.normcase(os.path.realpath(raw_prefix))
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SetupSupportError(
            "SETUP_INTERPRETER_PROBE_FAILED", "a Python interpreter identity could not be verified"
        ) from exc
    if not all((implementation, cache_tag, soabi)) or not os.path.isabs(prefix):
        raise SetupSupportError(
            "SETUP_INTERPRETER_PROBE_FAILED", "a Python interpreter ABI identity is incomplete"
        )
    return InterpreterIdentity(
        minor,
        system,
        arch,
        implementation,
        cache_tag,
        soabi,
        prefix,
    )


def _path_alias(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & WINDOWS_REPARSE_ATTRIBUTE
    )


def _inspect_regular_directory(path: Path, *, generated: bool) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise SetupSupportError("SETUP_VENV_UNSAFE", "a venv path could not be inspected") from exc
    if _path_alias(info) or not stat.S_ISDIR(info.st_mode):
        label = "generated venv recovery path" if generated else "extension venv"
        raise SetupSupportError("SETUP_VENV_UNSAFE", f"the {label} must be a regular directory")
    return True


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.normpath(os.path.abspath(os.fspath(path))))


def validate_venv_interpreter(
    python: Path,
    venv: Path,
    platform_name: str,
) -> InterpreterIdentity:
    """Validate the expected lexical entry and prove that it starts this venv."""

    if not _inspect_regular_directory(venv, generated=False):
        raise SetupSupportError("SETUP_VENV_MISSING", "the extension venv is unavailable")
    expected = venv_python(venv, platform_name)
    if _lexical_absolute(python) != _lexical_absolute(expected):
        raise SetupSupportError(
            "SETUP_VENV_INVALID", "runtime Python is not at the expected venv location"
        )
    try:
        info = expected.lstat()
    except OSError as exc:
        raise SetupSupportError(
            "SETUP_VENV_MISSING", "the expected extension venv Python is unavailable"
        ) from exc
    system = normalize_platform_name(platform_name)
    alias = _path_alias(info)
    if system == "win32":
        valid_entry = not alias and stat.S_ISREG(info.st_mode)
    else:
        valid_entry = stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
        if valid_entry and stat.S_ISLNK(info.st_mode):
            try:
                valid_entry = stat.S_ISREG(expected.stat().st_mode)
            except OSError:
                valid_entry = False
    if not valid_entry:
        raise SetupSupportError(
            "SETUP_VENV_INVALID", "the expected extension venv Python entry is unsafe"
        )
    identity = interpreter_identity(expected)
    try:
        expected_prefix = os.path.normcase(os.path.realpath(str(venv.resolve(strict=True))))
    except OSError as exc:
        raise SetupSupportError(
            "SETUP_VENV_INVALID", "the extension venv root could not be verified"
        ) from exc
    if identity.prefix != expected_prefix:
        raise SetupSupportError(
            "SETUP_VENV_INVALID", "runtime Python does not identify the validated extension venv"
        )
    return identity


def _identity_matches_venv(
    actual: InterpreterIdentity | None,
    requested: InterpreterIdentity,
) -> bool:
    return actual is not None and actual.abi_signature() == requested.abi_signature()


def _validate_owned_parent(path: Path, extension_dir: Path, extension_root: Path) -> None:
    try:
        path.relative_to(extension_dir)
        parent = path.parent.resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise SetupSupportError(
            "SETUP_VENV_CONTAINMENT", "a generated venv recovery path is unsafe"
        ) from exc
    if parent != extension_root and extension_root not in parent.parents:
        raise SetupSupportError(
            "SETUP_VENV_CONTAINMENT", "a generated venv recovery path escapes the extension"
        )


def _remove_owned_entry(path: Path, extension_dir: Path, extension_root: Path) -> None:
    """Remove a fixed extension-owned recovery entry without following aliases."""

    _validate_owned_parent(path, extension_dir, extension_root)
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise SetupSupportError("SETUP_VENV_CLEANUP_FAILED", "venv recovery cleanup failed") from exc
    if _path_alias(info):
        try:
            if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                path.rmdir()
            else:
                path.unlink()
        except IsADirectoryError:
            try:
                path.rmdir()
            except OSError as exc:
                raise SetupSupportError(
                    "SETUP_VENV_CLEANUP_FAILED", "venv recovery cleanup failed"
                ) from exc
        except OSError as exc:
            raise SetupSupportError(
                "SETUP_VENV_CLEANUP_FAILED", "venv recovery cleanup failed"
            ) from exc
        return
    if stat.S_ISDIR(info.st_mode):
        try:
            entries = list(os.scandir(path))
        except OSError as exc:
            raise SetupSupportError(
                "SETUP_VENV_CLEANUP_FAILED", "venv recovery cleanup failed"
            ) from exc
        for entry in entries:
            _remove_owned_entry(Path(entry.path), extension_dir, extension_root)
        try:
            path.rmdir()
        except OSError as exc:
            raise SetupSupportError(
                "SETUP_VENV_CLEANUP_FAILED", "venv recovery cleanup failed"
            ) from exc
        return
    try:
        path.unlink()
    except OSError as exc:
        raise SetupSupportError("SETUP_VENV_CLEANUP_FAILED", "venv recovery cleanup failed") from exc


def _probe_existing_venv(venv: Path, platform_name: str) -> InterpreterIdentity | None:
    if not _inspect_regular_directory(venv, generated=False):
        return None
    python = venv_python(venv, platform_name)
    try:
        return validate_venv_interpreter(python, venv, platform_name)
    except SetupSupportError:
        return None


def _build_venv(
    bootstrap_python: Path,
    destination: Path,
    platform_name: str,
    expected: InterpreterIdentity,
    log: LogFunction,
) -> Path:
    run_checked(
        [str(bootstrap_python), "-m", "venv", str(destination)],
        "Preparing isolated Python environment",
        log,
    )
    try:
        python = venv_python(destination, platform_name)
        actual = validate_venv_interpreter(python, destination, platform_name)
    except SetupSupportError as exc:
        raise SetupSupportError(
            "SETUP_VENV_INVALID", "the prepared extension venv could not be verified"
        ) from exc
    if not _identity_matches_venv(actual, expected):
        raise SetupSupportError(
            "SETUP_VENV_ABI_MISMATCH", "the prepared extension venv has the wrong Python ABI"
        )
    return python


def _rename(source: Path, destination: Path) -> None:
    try:
        os.replace(source, destination)
    except OSError as exc:
        raise SetupSupportError(
            "SETUP_VENV_SWAP_FAILED", "the replacement venv could not be activated; run Repair again"
        ) from exc


def _activated_venv_python(
    venv: Path,
    platform_name: str,
    requested: InterpreterIdentity,
) -> Path:
    python = venv_python(venv, platform_name)
    try:
        actual = validate_venv_interpreter(python, venv, platform_name)
    except SetupSupportError as exc:
        raise SetupSupportError(
            "SETUP_VENV_INVALID", "the activated extension venv could not be verified"
        ) from exc
    if not _identity_matches_venv(actual, requested):
        raise SetupSupportError(
            "SETUP_VENV_ABI_MISMATCH", "the activated extension venv has the wrong Python ABI"
        )
    return python


def create_or_reuse_venv(
    bootstrap_python: Path,
    extension_dir: Path,
    platform_name: str,
    log: LogFunction,
) -> Path:
    venv = extension_dir / "venv"
    staging = extension_dir / VENV_STAGING_NAME
    backup = extension_dir / VENV_BACKUP_NAME
    try:
        extension_root = extension_dir.resolve(strict=True)
    except OSError as exc:
        raise SetupSupportError("SETUP_VENV_UNSAFE", "the extension directory is unavailable") from exc

    if venv.exists() or venv.is_symlink():
        _inspect_regular_directory(venv, generated=False)

    requested = interpreter_identity(bootstrap_python)
    if requested.python_minor not in {"3.11", "3.12"}:
        raise SetupSupportError(
            "SETUP_PYTHON_UNSUPPORTED", "this release requires Modly Python 3.11 or 3.12"
        )
    if requested.system != normalize_platform_name(platform_name):
        raise SetupSupportError(
            "SETUP_PLATFORM_MISMATCH", "the bootstrap Python does not match setup platform metadata"
        )
    if requested.arch not in {"x64", "arm64"} or (
        requested.system == "win32" and requested.arch != "x64"
    ):
        raise SetupSupportError(
            "SETUP_ARCH_UNSUPPORTED", "the bootstrap Python architecture is not supported"
        )

    # Generated aliases/files are safe to unlink because their fixed names are
    # extension-owned. The live venv itself is never removed until a verified
    # staged replacement exists.
    for generated_path in (staging, backup):
        if generated_path.exists() or generated_path.is_symlink():
            try:
                info = generated_path.lstat()
            except OSError as exc:
                raise SetupSupportError(
                    "SETUP_VENV_UNSAFE", "a venv recovery path could not be inspected"
                ) from exc
            if _path_alias(info) or not stat.S_ISDIR(info.st_mode):
                _remove_owned_entry(generated_path, extension_dir, extension_root)

    current = _probe_existing_venv(venv, platform_name)
    staged = _probe_existing_venv(staging, platform_name)

    # Recover a completed first half of the two-rename swap.
    if current is None and not venv.exists() and backup.exists():
        if _identity_matches_venv(staged, requested):
            try:
                _rename(staging, venv)
                activated = _activated_venv_python(venv, platform_name, requested)
            except SetupSupportError:
                if venv.exists():
                    _remove_owned_entry(venv, extension_dir, extension_root)
                _rename(backup, venv)
                raise
            _remove_owned_entry(backup, extension_dir, extension_root)
            return activated
        _rename(backup, venv)
        current = _probe_existing_venv(venv, platform_name)

    # A live verified replacement wins over an interrupted leftover backup.
    if backup.exists() and _identity_matches_venv(current, requested):
        _remove_owned_entry(backup, extension_dir, extension_root)
    elif backup.exists() and venv.exists():
        # The activation was not verifiably complete. Restore the previous venv.
        _remove_owned_entry(venv, extension_dir, extension_root)
        _rename(backup, venv)
        current = _probe_existing_venv(venv, platform_name)

    reuse_staged = _identity_matches_venv(staged, requested) and not _identity_matches_venv(
        current, requested
    )
    if staging.exists() and not reuse_staged:
        _remove_owned_entry(staging, extension_dir, extension_root)

    if _identity_matches_venv(current, requested):
        # Same-ABI Repair may refresh the venv in place.
        return _build_venv(
            bootstrap_python, venv, platform_name, requested, log
        )

    if not reuse_staged:
        _build_venv(bootstrap_python, staging, platform_name, requested, log)
    moved_current = False
    activated = False
    try:
        if venv.exists():
            _rename(venv, backup)
            moved_current = True
        _rename(staging, venv)
        activated = True
        python = _activated_venv_python(venv, platform_name, requested)
    except SetupSupportError:
        if activated and venv.exists():
            _remove_owned_entry(venv, extension_dir, extension_root)
        if moved_current and not venv.exists() and backup.exists():
            _rename(backup, venv)
        raise
    if backup.exists():
        _remove_owned_entry(backup, extension_dir, extension_root)
    return python


def dependency_commands(
    venv_python_path: Path,
    root: Path,
    flavor: PlatformFlavor,
) -> list[tuple[str, list[str]]]:
    python = str(venv_python_path)
    constraints = str(root / "constraints.txt")
    requirements = str(root / "requirements.txt")
    build_specs = [f"{name}=={version}" for name, version in BUILD_DISTRIBUTIONS.items()]
    native_specs = [
        f"{name}=={RUNTIME_DISTRIBUTIONS[name]}" for name in NATIVE_BINARY_DISTRIBUTIONS
    ]
    torch_record = TORCH_VARIANTS[flavor.torch_variant]
    torch_specs = [
        f"torch=={torch_record['torch']}",
        f"torchaudio=={torch_record['torchaudio']}",
    ]
    return [
        (
            "Pinning Python build tools",
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--only-binary=:all:",
                *build_specs,
            ],
        ),
        (
            f"Installing pinned PyTorch {flavor.torch_variant} runtime",
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--only-binary=:all:",
                "--index-url",
                str(torch_record["index"]),
                *torch_specs,
            ],
        ),
        (
            "Installing pinned native wheels",
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--only-binary=:all:",
                "--constraint",
                constraints,
                *native_specs,
            ],
        ),
        (
            "Installing the pinned Python SoX wrapper",
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-build-isolation",
                "--no-binary=sox",
                "--no-deps",
                "--constraint",
                constraints,
                f"sox=={RUNTIME_DISTRIBUTIONS['sox']}",
            ],
        ),
        (
            "Installing the pinned Qwen3-TTS runtime",
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--only-binary=:all:",
                "--constraint",
                constraints,
                "--requirement",
                requirements,
                *torch_specs,
            ],
        ),
    ]


def installed_versions(python: Path, names: tuple[str, ...]) -> dict[str, str]:
    script = (
        "import importlib.metadata as m,json;"
        f"names={list(names)!r};"
        "out={};"
        "\nfor n in names:\n"
        " try: out[n]=m.version(n)\n"
        " except m.PackageNotFoundError: out[n]=''\n"
        "print(json.dumps(out,sort_keys=True))"
    )
    try:
        completed = subprocess.run(
            [str(python), "-c", script],
            check=True,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
        )
        parsed = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise SetupSupportError(
            "SETUP_VERSION_INSPECTION_FAILED", "installed versions could not be inspected"
        ) from exc
    if not isinstance(parsed, dict):
        raise SetupSupportError(
            "SETUP_VERSION_INSPECTION_FAILED", "installed version output is invalid"
        )
    return {str(key): str(value) for key, value in parsed.items()}


CUSPARSELT_AUDIT_SCRIPT = r'''
import importlib.metadata as metadata
import os
from pathlib import Path, PurePath
import stat
import sys

WINDOWS_REPARSE_ATTRIBUTE = 0x400

def unsafe_alias(info):
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & WINDOWS_REPARSE_ATTRIBUTE
    )

try:
    distribution = metadata.distribution(__DISTRIBUTION__)
    if (
        str(distribution.metadata.get("Name", "")).casefold() != __DISTRIBUTION__
        or distribution.version != __VERSION__
    ):
        raise RuntimeError("version")
    records = tuple(distribution.files or ())
    if not records:
        raise RuntimeError("records")
    base = Path(distribution.locate_file(""))
    base_info = base.lstat()
    if unsafe_alias(base_info) or not stat.S_ISDIR(base_info.st_mode):
        raise RuntimeError("base")
    base_canonical = base.resolve(strict=True)
    wheels = []
    for record in records:
        relative = PurePath(str(record))
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise RuntimeError("relative")
        current = base
        for index, part in enumerate(relative.parts):
            current = current / part
            info = current.lstat()
            if unsafe_alias(info):
                raise RuntimeError("alias")
            if index + 1 < len(relative.parts):
                if not stat.S_ISDIR(info.st_mode):
                    raise RuntimeError("parent")
            elif not stat.S_ISREG(info.st_mode):
                raise RuntimeError("file")
        resolved = current.resolve(strict=True)
        if base_canonical not in resolved.parents:
            raise RuntimeError("containment")
        if relative.parts == (__DIST_INFO__, "WHEEL"):
            wheels.append(current)
    if len(wheels) != 1:
        raise RuntimeError("wheel-count")
    wheel_info = wheels[0].lstat()
    if wheel_info.st_size <= 0 or wheel_info.st_size > 65536:
        raise RuntimeError("wheel-size")
    wheel_text = wheels[0].read_text(encoding="utf-8")
    tags = [line for line in wheel_text.splitlines() if line.startswith("Tag: ")]
    if tags != [__WHEEL_TAG__]:
        raise RuntimeError("wheel-tag")
except BaseException:
    raise SystemExit(1)

sys.stdout.buffer.write(__SUCCESS__)
'''


def _cusparselt_sbsa_distribution_is_safe(python: Path) -> bool:
    script = (
        CUSPARSELT_AUDIT_SCRIPT.replace("__DISTRIBUTION__", repr(CUSPARSELT_DISTRIBUTION))
        .replace("__VERSION__", repr(CUSPARSELT_VERSION))
        .replace("__DIST_INFO__", repr(CUSPARSELT_DIST_INFO))
        .replace("__WHEEL_TAG__", repr(CUSPARSELT_SBSA_WHEEL_TAG))
        .replace("__SUCCESS__", repr(CUSPARSELT_AUDIT_SUCCESS))
    )
    try:
        outcome = _bounded_command(
            [str(python), "-I", "-c", script],
            output_limit=4096,
            timeout=30,
        )
    except (OSError, ValueError):
        return False
    return (
        outcome.returncode == 0
        and not outcome.truncated
        and not outcome.timed_out
        and outcome.output == CUSPARSELT_AUDIT_SUCCESS
    )


def _pip_check_exception_allowed(
    python: Path,
    flavor: PlatformFlavor,
    outcome: CommandOutcome,
) -> bool:
    if not (
        flavor.system == "linux"
        and flavor.arch == "arm64"
        and flavor.accelerator == "cuda"
        and flavor.torch_variant == "cu130"
        and flavor.expected_cuda == "13.0"
        and outcome.returncode == 1
        and not outcome.truncated
        and not outcome.timed_out
    ):
        return False
    try:
        output = outcome.output.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False
    if output not in {
        CUSPARSELT_PIP_CHECK_DIAGNOSTIC,
        CUSPARSELT_PIP_CHECK_DIAGNOSTIC + "\n",
    }:
        return False
    return _cusparselt_sbsa_distribution_is_safe(python)


def _pip_check_outcome(python: Path) -> CommandOutcome:
    return _bounded_command([str(python), "-m", "pip", "check"])


def pip_check_passes(python: Path, flavor: PlatformFlavor, log: LogFunction) -> bool:
    log("Checking dependency graph")
    try:
        outcome = _pip_check_outcome(python)
    except OSError:
        log("Dependency graph check could not start; Repair will reinstall")
        return False
    if outcome.returncode == 0 and not outcome.truncated and not outcome.timed_out:
        return True
    if _pip_check_exception_allowed(python, flavor, outcome):
        log("Accepted verified Linux ARM64 cu130 wheel-tag compatibility exception")
        return True
    log("Dependency graph is inconsistent; Repair will reinstall")
    return False


def verify_pip_check(python: Path, flavor: PlatformFlavor, log: LogFunction) -> None:
    stage = "Verifying dependency graph"
    log(stage)
    try:
        outcome = _pip_check_outcome(python)
    except OSError as exc:
        raise SetupSupportError(
            "SETUP_COMMAND_START_FAILED", f"{stage} could not start; run Repair again"
        ) from exc
    if outcome.returncode == 0 and not outcome.truncated and not outcome.timed_out:
        return
    if _pip_check_exception_allowed(python, flavor, outcome):
        log("Accepted verified Linux ARM64 cu130 wheel-tag compatibility exception")
        return
    raise _command_failure(stage, outcome)


def install_dependencies(
    python: Path,
    root: Path,
    flavor: PlatformFlavor,
    log: LogFunction,
) -> None:
    for filename in ("constraints.txt", "requirements.txt"):
        if not (root / filename).is_file():
            raise SetupSupportError("SETUP_REQUIREMENTS_MISSING", f"{filename} is missing")
    expected = expected_distributions(flavor)
    try:
        versions = installed_versions(python, tuple(expected))
    except SetupSupportError:
        versions = {}
    if versions == expected and pip_check_passes(python, flavor, log):
        log("Selected direct dependency pins are healthy; skipped installation")
        return
    for stage, command in dependency_commands(python, root, flavor):
        run_checked(command, stage, log)
    verify_pip_check(python, flavor, log)


HEALTH_PLATFORM_HELPERS = r'''
def _normalize_health_system(value):
    normalized = str(value or "").strip().casefold()
    if normalized.startswith("linux"):
        return "linux"
    if normalized in {"win32", "windows"}:
        return "win32"
    return normalized

def _normalize_health_arch(value):
    normalized = str(value or "").strip().casefold()
    if normalized in {"x64", "amd64", "x86_64"}:
        return "x64"
    if normalized in {"arm64", "aarch64"}:
        return "arm64"
    return normalized
'''


HEALTH_FAILURE_HELPERS = r'''
class _HealthSignal(RuntimeError):
    def __init__(self, category):
        self.category = category
        super().__init__(category)

def _health_category(error, substage):
    if isinstance(error, _HealthSignal):
        return error.category
    error_type = type(error)
    type_name = str(getattr(error_type, "__name__", "")).casefold()
    type_module = str(getattr(error_type, "__module__", "")).casefold()
    try:
        private_message = str(error).casefold()
    except BaseException:
        private_message = ""
    cuda_oom = (
        (type_name == "outofmemoryerror" and type_module.startswith("torch"))
        or "cudaerrormemoryallocation" in private_message
        or "cuda error: out of memory" in private_message
        or "cuda out of memory" in private_message
        or "cuda error: memory allocation" in private_message
    )
    if cuda_oom and substage in {
        "cuda_device",
        "float32_alloc",
        "float32_matmul",
        "float32_conv",
        "bf16_probe",
        "sdpa",
        "cuda_sync",
        "torchaudio",
    }:
        return "cuda_oom"
    if substage in {"imports_core", "imports_numeric", "imports_audio", "imports_native"}:
        return "import"
    if substage in {"cuda_availability", "cuda_device"}:
        return "cuda_unavailable"
    return "native_probe"
'''


HEALTH_SCRIPT = r'''
import importlib.metadata as metadata
import io
import json
import os
import platform
import shutil
import sys

''' + HEALTH_PLATFORM_HELPERS + HEALTH_FAILURE_HELPERS + r'''

HEALTH_SCHEMA = __HEALTH_SCHEMA__
ALLOWED_CATEGORIES = __HEALTH_CATEGORIES__
ALLOWED_SUBSTAGES = __HEALTH_SUBSTAGES__
expected = __EXPECTED__
flavor = __FLAVOR__

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

protocol_fd = os.dup(1)
discard_fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(discard_fd, 1)
os.dup2(discard_fd, 2)
os.close(discard_fd)
protocol = os.fdopen(protocol_fd, "w", encoding="utf-8", buffering=1)
terminal = False

def emit(payload):
    global terminal
    if terminal:
        return
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > 4096:
        encoded = json.dumps(
            {
                "schema": HEALTH_SCHEMA,
                "category": "invalid_result",
                "substage": "result",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    protocol.write(encoded + "\n")
    protocol.flush()
    terminal = True

substage = "bootstrap"
exit_code = 1
try:
    substage = "platform_identity"
    runtime_system = _normalize_health_system(sys.platform)
    runtime_arch = _normalize_health_arch(platform.machine())
    if runtime_system != flavor["system"] or runtime_arch != flavor["arch"]:
        raise _HealthSignal("invalid_result")

    substage = "imports_core"
    import accelerate
    import einops
    import gradio
    import qwen_tts
    import transformers

    substage = "imports_numeric"
    import llvmlite
    import numba
    import numpy as np
    import scipy
    import sklearn

    substage = "imports_audio"
    import librosa
    import soundfile as sf
    import sox
    import soxr

    substage = "imports_native"
    import onnxruntime as ort
    import torch
    import torch.nn.functional as F
    import torchaudio

    substage = "dependency_versions"
    try:
        actual = {name: metadata.version(name) for name in expected}
    except BaseException:
        raise _HealthSignal("invalid_result")
    if actual != expected:
        raise _HealthSignal("invalid_result")

    substage = "python_version"
    if sys.version_info[:2] not in ((3, 11), (3, 12)):
        raise _HealthSignal("invalid_result")

    device = torch.device("cpu")
    dtype = torch.float32
    bf16 = False
    if flavor["accelerator"] == "cuda":
        substage = "cuda_availability"
        if not torch.cuda.is_available() or torch.version.cuda != flavor["expected_cuda"]:
            raise _HealthSignal("cuda_unavailable")
        substage = "cuda_device"
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    elif flavor["accelerator"] != "cpu":
        raise _HealthSignal("invalid_result")

    substage = "float32_alloc"
    a = torch.randn((16, 16), device=device, dtype=torch.float32)
    b = torch.randn((16, 16), device=device, dtype=torch.float32)
    conv_input = torch.randn((1, 2, 8, 8), device=device, dtype=torch.float32)
    conv_weight = torch.randn((4, 2, 3, 3), device=device, dtype=torch.float32)

    substage = "float32_matmul"
    matmul = a @ b
    if not bool(torch.isfinite(matmul).all().item()):
        raise _HealthSignal("native_probe")

    substage = "float32_conv"
    conv = F.conv2d(conv_input, conv_weight)
    if not bool(torch.isfinite(conv).all().item()):
        raise _HealthSignal("native_probe")

    substage = "sdpa"
    q = torch.randn((1, 2, 8, 16), device=device, dtype=torch.float32)
    sdpa = F.scaled_dot_product_attention(q, q, q)
    if not bool(torch.isfinite(sdpa).all().item()):
        raise _HealthSignal("native_probe")

    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        substage = "bf16_probe"
        try:
            bf16_a = torch.randn((8, 8), device=device, dtype=torch.bfloat16)
            bf16_b = torch.randn((8, 8), device=device, dtype=torch.bfloat16)
            bf16_result = bf16_a @ bf16_b
            bf16_conv = F.conv2d(
                torch.randn((1, 2, 8, 8), device=device, dtype=torch.bfloat16),
                torch.randn((4, 2, 3, 3), device=device, dtype=torch.bfloat16),
            )
            bf16_q = torch.randn((1, 1, 4, 8), device=device, dtype=torch.bfloat16)
            bf16_sdpa = F.scaled_dot_product_attention(bf16_q, bf16_q, bf16_q)
            bf16 = all(
                bool(torch.isfinite(value.float()).all().item())
                for value in (bf16_result, bf16_conv, bf16_sdpa)
            )
        except BaseException as bf16_error:
            if _health_category(bf16_error, substage) == "cuda_oom":
                raise
            bf16 = False
        if bf16:
            dtype = torch.bfloat16

    if device.type == "cuda":
        substage = "cuda_sync"
        torch.cuda.synchronize(device)

    substage = "torchaudio"
    tone = torch.linspace(-1.0, 1.0, 240, dtype=torch.float32)
    resampled = torchaudio.functional.resample(tone, 24000, 16000)
    if resampled.numel() == 0 or not bool(torch.isfinite(resampled).all().item()):
        raise _HealthSignal("native_probe")

    substage = "onnx_cpu"
    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise _HealthSignal("native_probe")

    substage = "soundfile"
    samples = np.array([-1.0, -0.25, 0.0, 0.25, 1.0], dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, samples, 24000, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    decoded, rate = sf.read(buffer, dtype="float32", always_2d=False)
    if rate != 24000 or decoded.ndim != 1 or decoded.size != samples.size:
        raise _HealthSignal("native_probe")

    substage = "librosa"
    librosa_out = librosa.resample(samples, orig_sr=24000, target_sr=16000)
    if librosa_out.size == 0 or not bool(np.isfinite(librosa_out).all()):
        raise _HealthSignal("native_probe")

    substage = "result"
    facts = {
        "system": runtime_system,
        "arch": runtime_arch,
        "pythonMinor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "accelerator": flavor["accelerator"],
        "torchVariant": flavor["torch_variant"],
        "cudaBf16": bf16,
        "runtimeDtype": "bfloat16" if dtype == torch.bfloat16 else "float32",
        "attention": "sdpa",
        "externalSox": shutil.which("sox") is not None,
    }
    emit({"schema": HEALTH_SCHEMA, "facts": facts})
    exit_code = 0
except BaseException as error:
    category = _health_category(error, substage)
    if category not in ALLOWED_CATEGORIES:
        category = "native_probe"
    if substage not in ALLOWED_SUBSTAGES:
        substage = "bootstrap"
    emit(
        {
            "schema": HEALTH_SCHEMA,
            "category": category,
            "substage": substage,
        }
    )
finally:
    try:
        protocol.close()
    except BaseException:
        pass

raise SystemExit(exit_code)
'''


def _render_health_script(flavor: PlatformFlavor) -> str:
    return (
        HEALTH_SCRIPT.replace("__HEALTH_SCHEMA__", repr(HEALTH_SCHEMA))
        .replace("__HEALTH_CATEGORIES__", repr(set(HEALTH_CATEGORIES)))
        .replace("__HEALTH_SUBSTAGES__", repr(set(HEALTH_SUBSTAGES)))
        .replace("__EXPECTED__", repr(expected_distributions(flavor)))
        .replace("__FLAVOR__", repr(flavor.state_payload()))
    )


def _generic_health_failure() -> SetupSupportError:
    return SetupSupportError(
        "SETUP_HEALTH_FAILED",
        "runtime health output could not be verified; run Repair again",
    )


def _health_category_failure(category: str, substage: str) -> SetupSupportError:
    messages = {
        "cuda_oom": (
            "SETUP_HEALTH_CUDA_OOM",
            "CUDA health allocation ran out of GPU memory",
            "release GPU memory and run Repair again",
        ),
        "import": (
            "SETUP_HEALTH_IMPORT_FAILED",
            "a pinned runtime import failed",
            "run Repair to restore the extension environment",
        ),
        "cuda_unavailable": (
            "SETUP_HEALTH_CUDA_UNAVAILABLE",
            "the selected CUDA runtime is unavailable",
            "verify the accelerator runtime and run Repair again",
        ),
        "native_probe": (
            "SETUP_HEALTH_NATIVE_PROBE_FAILED",
            "a native tensor or audio probe failed",
            "run Repair and verify platform support",
        ),
        "invalid_result": (
            "SETUP_HEALTH_INVALID_RESULT",
            "runtime health identity or results were inconsistent",
            "run Repair again",
        ),
    }
    if category not in messages or substage not in HEALTH_SUBSTAGES:
        return _generic_health_failure()
    code, reason, action = messages[category]
    return SetupSupportError(
        code,
        f"{reason} at health substage {substage}; {action}",
    )


def _strict_health_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate health envelope key")
        result[key] = value
    return result


def _reject_health_constant(_value: str) -> object:
    raise ValueError("non-finite health envelope value")


def _parse_health_envelope(outcome: CommandOutcome) -> tuple[str, dict[str, object]]:
    if outcome.truncated or outcome.timed_out or len(outcome.output) > HEALTH_OUTPUT_LIMIT:
        raise _generic_health_failure()
    try:
        decoded = outcome.output.decode("utf-8", errors="strict")
        if not decoded.endswith("\n") or decoded.count("\n") != 1:
            raise ValueError("health envelope line count")
        envelope = json.loads(
            decoded[:-1],
            object_pairs_hook=_strict_health_object,
            parse_constant=_reject_health_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise _generic_health_failure() from exc
    if not isinstance(envelope, dict) or envelope.get("schema") != HEALTH_SCHEMA:
        raise _generic_health_failure()
    if set(envelope) == {"schema", "category", "substage"}:
        if (
            outcome.returncode != 1
            or envelope.get("category") not in HEALTH_CATEGORIES
            or envelope.get("substage") not in HEALTH_SUBSTAGES
        ):
            raise _generic_health_failure()
        return "error", {
            "category": str(envelope["category"]),
            "substage": str(envelope["substage"]),
        }
    if set(envelope) == {"schema", "facts"}:
        if outcome.returncode != 0:
            raise _generic_health_failure()
        facts = envelope.get("facts")
        if not isinstance(facts, dict):
            raise _generic_health_failure()
        return "ok", facts
    raise _generic_health_failure()


def verify_runtime(
    python: Path,
    flavor: PlatformFlavor,
    log: LogFunction,
) -> dict[str, object]:
    log("Running portable tensor, audio, and runtime health checks")
    script = _render_health_script(flavor)
    try:
        outcome = _bounded_command(
            [str(python), "-I", "-c", script],
            output_limit=HEALTH_OUTPUT_LIMIT,
            timeout=HEALTH_TIMEOUT_SECONDS,
        )
    except OSError as exc:
        raise SetupSupportError("SETUP_HEALTH_START_FAILED", "health check could not start") from exc
    status, result = _parse_health_envelope(outcome)
    if status == "error":
        raise _health_category_failure(
            str(result["category"]),
            str(result["substage"]),
        )
    expected_result = {
        "system": flavor.system,
        "arch": flavor.arch,
        "accelerator": flavor.accelerator,
        "torchVariant": flavor.torch_variant,
        "attention": "sdpa",
    }
    if (
        any(result.get(key) != value for key, value in expected_result.items())
        or result.get("pythonMinor") not in {"3.11", "3.12"}
        or not isinstance(result.get("cudaBf16"), bool)
        or result.get("runtimeDtype")
        != ("bfloat16" if result.get("cudaBf16") else "float32")
        or not isinstance(result.get("externalSox"), bool)
        or set(result)
        != {
            "system",
            "arch",
            "pythonMinor",
            "accelerator",
            "torchVariant",
            "cudaBf16",
            "runtimeDtype",
            "attention",
            "externalSox",
        }
    ):
        raise _health_category_failure("invalid_result", "result")
    if not result.get("externalSox"):
        log("Warning: external SoX is absent; this scalar CustomVoice path does not invoke it")
    log("Runtime health checks passed")
    return result
