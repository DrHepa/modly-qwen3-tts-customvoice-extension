from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import setup
from qwen3_tts_modly import setup_support
from qwen3_tts_modly.assets import AssetError
from qwen3_tts_modly.constants import STATE_FILENAME
from qwen3_tts_modly.paths import PathContractError
from qwen3_tts_modly.state import StateError


NATIVE_SYSTEM = setup_support.normalize_platform_name(sys.platform)


def test_parse_setup_args_accepts_current_json_and_parses_actual_legacy_slots() -> None:
    payload = {
        "python_exe": "/runtime/python",
        "ext_dir": "/runtime/extension",
        "models_dir": "/runtime/models",
        "gpu_sm": 90,
        "cuda_version": 128,
        "accelerator": "cuda",
        "platform": "linux",
        "arch": "x64",
    }
    assert setup.parse_setup_args(["setup.py", json.dumps(payload)]) == payload
    for arguments, expected_cuda in (
        (["setup.py", "/runtime/python", "/runtime/extension", "90"], 0),
        (["setup.py", "/runtime/python", "/runtime/extension", "90", "128"], 128),
    ):
        legacy = setup.parse_setup_args(arguments)
        assert "models_dir" not in legacy
        assert legacy["gpu_sm"] == 90
        assert legacy["cuda_version"] == expected_cuda
        assert legacy["accelerator"] == "cuda"
        assert legacy["_legacy_setup"] is True


@pytest.mark.parametrize(
    "argv",
    [
        ["setup.py"],
        ["setup.py", "[]"],
        ["setup.py", "not-json"],
        ["setup.py", "/python", "/extension", "90", "/invented-models", "128"],
    ],
)
def test_parse_setup_args_rejects_invalid_invocations(argv: list[str]) -> None:
    with pytest.raises(setup.SetupFailure, match="SETUP_PAYLOAD_INVALID"):
        setup.parse_setup_args(argv)


@pytest.mark.parametrize("system", ["linux", "win32"])
@pytest.mark.parametrize(
    ("accelerator", "cuda_version", "variant"),
    [("cpu", 0, "cpu"), ("cuda", 128, "cu128"), ("nvidia", "13.0", "cu130")],
)
def test_public_x64_platform_matrix_selects_exact_torch_flavor(
    system: str,
    accelerator: str,
    cuda_version: object,
    variant: str,
) -> None:
    flavor = setup_support.select_platform_flavor(
        {
            "platform": system,
            "arch": "amd64",
            "accelerator": accelerator,
            "cuda_version": cuda_version,
            "gpu_sm": 999,
        }
    )
    assert flavor.system == system
    assert flavor.arch == "x64"
    assert flavor.torch_variant == variant


@pytest.mark.parametrize("variant_hint,variant", [(0, "cpu"), (128, "cu128"), (130, "cu130")])
def test_linux_arm64_is_a_non_equality_gated_candidate(variant_hint: int, variant: str) -> None:
    accelerator = "cpu" if variant_hint == 0 else "cuda"
    flavor = setup_support.select_platform_flavor(
        {
            "platform": "linux",
            "arch": "aarch64",
            "accelerator": accelerator,
            "cuda_version": variant_hint,
            "gpu_sm": 1,
        }
    )
    assert flavor.arch == "arm64"
    assert flavor.torch_variant == variant


@pytest.mark.parametrize(
    ("context", "code"),
    [
        ({"platform": "darwin", "arch": "x64", "accelerator": "cpu"}, "SETUP_PLATFORM_UNSUPPORTED"),
        ({"platform": "win32", "arch": "arm64", "accelerator": "cpu"}, "SETUP_ARCH_UNSUPPORTED"),
        ({"platform": "linux", "arch": "x64", "accelerator": "mps"}, "SETUP_ACCELERATOR_UNSUPPORTED"),
        ({"platform": "linux", "arch": "x64", "accelerator": "cuda", "cuda_version": 129}, "SETUP_CUDA_UNSUPPORTED"),
        ({"platform": "linux", "arch": "x64", "accelerator": ""}, "SETUP_ACCELERATOR_UNSUPPORTED"),
    ],
)
def test_unsupported_platform_metadata_fails_with_stable_code(context: dict, code: str) -> None:
    with pytest.raises(setup_support.SetupSupportError, match=code):
        setup_support.select_platform_flavor(context)


def test_running_platform_validation_rejects_metadata_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flavor = setup_support.PlatformFlavor("linux", "x64", "cpu", "cpu", None)
    monkeypatch.setattr(setup_support.sys, "platform", "win32")
    with pytest.raises(setup_support.SetupSupportError, match="SETUP_PLATFORM_MISMATCH"):
        setup_support.validate_running_platform(flavor)
    monkeypatch.setattr(setup_support.sys, "platform", "linux")
    monkeypatch.setattr(setup_support.platform, "machine", lambda: "aarch64")
    with pytest.raises(setup_support.SetupSupportError, match="SETUP_ARCH_MISMATCH"):
        setup_support.validate_running_platform(flavor)


def test_venv_python_is_platform_safe() -> None:
    root = Path("extension") / "venv"
    assert setup_support.venv_python(root, "linux") == root / "bin" / "python"
    assert setup_support.venv_python(root, "win32") == root / "Scripts" / "python.exe"


def test_interpreter_identity_probe_is_isolated_bounded_and_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command[1:3] == ["-I", "-c"]
        assert kwargs["stderr"] is subprocess.DEVNULL
        assert kwargs["timeout"] == 20
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "pythonMinor": "3.12",
                    "system": "linux",
                    "arch": "x86_64",
                    "implementation": "cpython",
                    "cacheTag": "cpython-312",
                    "soabi": "cpython-312-x86_64-linux-gnu",
                    "prefix": str(tmp_path),
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(setup_support.subprocess, "run", fake_run)
    assert setup_support.interpreter_identity(Path("python")) == (
        setup_support.InterpreterIdentity(
            "3.12",
            "linux",
            "x64",
            "cpython",
            "cpython-312",
            "cpython-312-x86_64-linux-gnu",
            os.path.normcase(os.path.realpath(tmp_path)),
        )
    )


def _write_fake_python(venv: Path, system: str, marker: str) -> Path:
    python = setup_support.venv_python(venv, system)
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text(marker, encoding="utf-8")
    return python


def _fake_venv_tools(
    monkeypatch: pytest.MonkeyPatch,
    *,
    requested_minor: str,
    system: str,
    arch: str = "x64",
) -> list[Path]:
    built: list[Path] = []

    def identity(python: Path) -> setup_support.InterpreterIdentity:
        marker = python.read_text(encoding="utf-8")
        parts = marker.split("|")
        minor, marker_system, marker_arch = parts[:3]
        implementation = parts[3] if len(parts) > 3 else "cpython"
        cache_tag = parts[4] if len(parts) > 4 else f"cpython-{minor.replace('.', '')}"
        soabi = parts[5] if len(parts) > 5 else f"cp{minor.replace('.', '')}-{marker_system}-{marker_arch}"
        if python.parent.name.casefold() in {"bin", "scripts"}:
            prefix_path = python.parent.parent
        else:
            prefix_path = python.parent
        prefix = os.path.normcase(os.path.realpath(prefix_path))
        return setup_support.InterpreterIdentity(
            minor,
            marker_system,
            marker_arch,
            implementation,
            cache_tag,
            soabi,
            prefix,
        )

    def run(command: list[str], _stage: str, _log: object) -> None:
        destination = Path(command[-1])
        built.append(destination)
        _write_fake_python(
            destination,
            system,
            f"{requested_minor}|{system}|{arch}",
        )

    monkeypatch.setattr(setup_support, "interpreter_identity", identity)
    monkeypatch.setattr(setup_support, "run_checked", run)
    return built


@pytest.mark.parametrize(
    ("existing_minor", "requested_minor"),
    [("3.11", "3.12"), ("3.12", "3.11")],
)
def test_incompatible_python_minor_uses_verified_staged_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_minor: str,
    requested_minor: str,
) -> None:
    extension = tmp_path / "extension"
    extension.mkdir()
    bootstrap = tmp_path / "bootstrap-python"
    bootstrap.write_text(f"{requested_minor}|linux|x64", encoding="utf-8")
    _write_fake_python(extension / "venv", "linux", f"{existing_minor}|linux|x64")
    built = _fake_venv_tools(
        monkeypatch, requested_minor=requested_minor, system="linux"
    )

    result = setup_support.create_or_reuse_venv(
        bootstrap, extension, "linux", lambda _message: None
    )

    assert result == extension / "venv" / "bin" / "python"
    assert result.read_text(encoding="utf-8") == f"{requested_minor}|linux|x64"
    assert built == [extension / setup_support.VENV_STAGING_NAME]
    assert not (extension / setup_support.VENV_STAGING_NAME).exists()
    assert not (extension / setup_support.VENV_BACKUP_NAME).exists()


def test_compatible_venv_repairs_in_place(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    extension = tmp_path / "extension"
    extension.mkdir()
    bootstrap = tmp_path / "bootstrap-python"
    bootstrap.write_text("3.12|linux|x64", encoding="utf-8")
    _write_fake_python(extension / "venv", "linux", "3.12|linux|x64")
    built = _fake_venv_tools(monkeypatch, requested_minor="3.12", system="linux")

    setup_support.create_or_reuse_venv(
        bootstrap, extension, "linux", lambda _message: None
    )

    assert built == [extension / "venv"]
    assert not (extension / setup_support.VENV_STAGING_NAME).exists()


@pytest.mark.parametrize(
    "existing_marker",
    [
        "3.12|linux|x64|pypy|cpython-312|cp312-linux-x64",
        "3.12|linux|x64|cpython|tampered-cache|cp312-linux-x64",
        "3.12|linux|x64|cpython|cpython-312|tampered-soabi",
    ],
)
def test_implementation_cache_tag_or_soabi_mismatch_forces_staged_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_marker: str,
) -> None:
    extension = tmp_path / "extension"
    extension.mkdir()
    bootstrap = tmp_path / "bootstrap-python"
    bootstrap.write_text("3.12|linux|x64", encoding="utf-8")
    _write_fake_python(extension / "venv", "linux", existing_marker)
    built = _fake_venv_tools(monkeypatch, requested_minor="3.12", system="linux")

    setup_support.create_or_reuse_venv(
        bootstrap, extension, "linux", lambda _message: None
    )

    assert built == [extension / setup_support.VENV_STAGING_NAME]
    assert (extension / "venv" / "bin" / "python").read_text(encoding="utf-8") == (
        "3.12|linux|x64"
    )


def test_venv_interpreter_rejects_non_venv_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    venv = tmp_path / "venv"
    python = _write_fake_python(venv, "linux", "fixture")
    monkeypatch.setattr(
        setup_support,
        "interpreter_identity",
        lambda _python: setup_support.InterpreterIdentity(
            "3.12",
            "linux",
            "x64",
            "cpython",
            "cpython-312",
            "cp312-linux-x64",
            os.path.normcase(os.path.realpath(tmp_path / "outside-prefix")),
        ),
    )
    with pytest.raises(setup_support.SetupSupportError, match="SETUP_VENV_INVALID"):
        setup_support.validate_venv_interpreter(python, venv, "linux")


def test_venv_interpreter_requires_the_expected_lexical_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    venv = tmp_path / "venv"
    _write_fake_python(venv, "linux", "fixture")
    outside = tmp_path / "outside-python"
    outside.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(
        setup_support,
        "interpreter_identity",
        lambda _python: pytest.fail("unexpected lexical entry must not be executed"),
    )
    with pytest.raises(setup_support.SetupSupportError, match="SETUP_VENV_INVALID"):
        setup_support.validate_venv_interpreter(outside, venv, "linux")


def test_mocked_windows_venv_validates_scripts_python_and_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    venv = tmp_path / "venv"
    python = _write_fake_python(venv, "win32", "fixture")
    expected = setup_support.InterpreterIdentity(
        "3.12",
        "win32",
        "x64",
        "cpython",
        "cpython-312",
        "cp312-win_amd64",
        os.path.normcase(os.path.realpath(venv)),
    )
    monkeypatch.setattr(setup_support, "interpreter_identity", lambda _python: expected)
    assert setup_support.validate_venv_interpreter(python, venv, "win32") == expected


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX venv symlink layout")
def test_real_posix_venv_python_symlink_has_valid_prefix(tmp_path: Path) -> None:
    if tuple(sys.version_info[:2]) not in setup_support.SUPPORTED_PYTHON_MINORS:
        pytest.skip("test interpreter is outside the supported extension minors")
    extension = tmp_path / "extension"
    extension.mkdir()
    venv = extension / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--symlinks", str(venv)],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    python = setup_support.venv_python(venv, "linux")
    assert python.is_symlink()
    identity = setup_support.validate_venv_interpreter(python, venv, "linux")
    assert identity.prefix == os.path.normcase(os.path.realpath(venv))
    assert setup.verify_python_runtime(python, extension) == tuple(sys.version_info[:2])


def test_windows_x64_uses_scripts_python_and_staged_abi_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = tmp_path / "extension"
    extension.mkdir()
    bootstrap = tmp_path / "bootstrap-python.exe"
    bootstrap.write_text("3.12|win32|x64", encoding="utf-8")
    _write_fake_python(extension / "venv", "win32", "3.11|win32|x64")
    built = _fake_venv_tools(monkeypatch, requested_minor="3.12", system="win32")

    result = setup_support.create_or_reuse_venv(
        bootstrap, extension, "win32", lambda _message: None
    )

    assert result == extension / "venv" / "Scripts" / "python.exe"
    assert result.read_text(encoding="utf-8") == "3.12|win32|x64"
    assert built == [extension / setup_support.VENV_STAGING_NAME]


@pytest.mark.parametrize(
    ("existing_system", "existing_arch"),
    [("linux", "arm64"), ("win32", "x64")],
)
def test_platform_or_architecture_mismatch_never_repairs_venv_in_place(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_system: str,
    existing_arch: str,
) -> None:
    extension = tmp_path / "extension"
    extension.mkdir()
    bootstrap = tmp_path / "bootstrap-python"
    bootstrap.write_text("3.12|linux|x64", encoding="utf-8")
    _write_fake_python(
        extension / "venv",
        "linux",
        f"3.12|{existing_system}|{existing_arch}",
    )
    built = _fake_venv_tools(monkeypatch, requested_minor="3.12", system="linux")

    setup_support.create_or_reuse_venv(
        bootstrap, extension, "linux", lambda _message: None
    )

    assert built == [extension / setup_support.VENV_STAGING_NAME]
    assert (extension / "venv" / "bin" / "python").read_text(encoding="utf-8") == (
        "3.12|linux|x64"
    )


def test_failed_staged_activation_rolls_back_previous_venv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = tmp_path / "extension"
    extension.mkdir()
    bootstrap = tmp_path / "bootstrap-python"
    bootstrap.write_text("3.12|linux|x64", encoding="utf-8")
    original = _write_fake_python(extension / "venv", "linux", "3.11|linux|x64")
    _fake_venv_tools(monkeypatch, requested_minor="3.12", system="linux")
    real_replace = setup_support.os.replace
    failed = False

    def replace(source: object, destination: object) -> None:
        nonlocal failed
        if (
            Path(source).name == setup_support.VENV_STAGING_NAME
            and Path(destination).name == "venv"
            and not failed
        ):
            failed = True
            raise OSError("simulated activation failure")
        real_replace(source, destination)

    monkeypatch.setattr(setup_support.os, "replace", replace)
    with pytest.raises(setup_support.SetupSupportError, match="SETUP_VENV_SWAP_FAILED"):
        setup_support.create_or_reuse_venv(
            bootstrap, extension, "linux", lambda _message: None
        )

    assert original.read_text(encoding="utf-8") == "3.11|linux|x64"
    assert (extension / setup_support.VENV_STAGING_NAME).is_dir()
    assert not (extension / setup_support.VENV_BACKUP_NAME).exists()


def test_interrupted_swap_promotes_verified_stage_and_removes_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = tmp_path / "extension"
    extension.mkdir()
    bootstrap = tmp_path / "bootstrap-python"
    bootstrap.write_text("3.12|linux|x64", encoding="utf-8")
    _write_fake_python(
        extension / setup_support.VENV_STAGING_NAME, "linux", "3.12|linux|x64"
    )
    _write_fake_python(
        extension / setup_support.VENV_BACKUP_NAME, "linux", "3.11|linux|x64"
    )
    built = _fake_venv_tools(monkeypatch, requested_minor="3.12", system="linux")

    result = setup_support.create_or_reuse_venv(
        bootstrap, extension, "linux", lambda _message: None
    )

    assert result.read_text(encoding="utf-8") == "3.12|linux|x64"
    assert built == []
    assert not (extension / setup_support.VENV_STAGING_NAME).exists()
    assert not (extension / setup_support.VENV_BACKUP_NAME).exists()


def test_interrupted_completed_swap_discards_backup_then_repairs_in_place(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = tmp_path / "extension"
    extension.mkdir()
    bootstrap = tmp_path / "bootstrap-python"
    bootstrap.write_text("3.12|linux|x64", encoding="utf-8")
    _write_fake_python(extension / "venv", "linux", "3.12|linux|x64")
    _write_fake_python(
        extension / setup_support.VENV_BACKUP_NAME, "linux", "3.11|linux|x64"
    )
    built = _fake_venv_tools(monkeypatch, requested_minor="3.12", system="linux")

    setup_support.create_or_reuse_venv(
        bootstrap, extension, "linux", lambda _message: None
    )

    assert built == [extension / "venv"]
    assert not (extension / setup_support.VENV_BACKUP_NAME).exists()


@pytest.mark.parametrize("variant", ["cpu", "cu128", "cu130"])
def test_dependency_commands_enforce_order_indices_and_pins(tmp_path: Path, variant: str) -> None:
    record = setup_support.TORCH_VARIANTS[variant]
    flavor = setup_support.PlatformFlavor(
        system="linux",
        arch="x64",
        accelerator="cpu" if variant == "cpu" else "cuda",
        torch_variant=variant,
        expected_cuda=record["cuda"],
    )
    commands = setup_support.dependency_commands(tmp_path / "venv-python", tmp_path, flavor)
    assert [stage for stage, _command in commands] == [
        "Pinning Python build tools",
        f"Installing pinned PyTorch {variant} runtime",
        "Installing pinned native wheels",
        "Installing the pinned Python SoX wrapper",
        "Installing the pinned Qwen3-TTS runtime",
    ]
    torch_command = commands[1][1]
    assert torch_command[torch_command.index("--index-url") + 1] == record["index"]
    assert f"torch=={record['torch']}" in torch_command
    assert f"torchaudio=={record['torchaudio']}" in torch_command
    assert "--only-binary=:all:" in commands[2][1]
    assert {"--no-build-isolation", "--no-binary=sox", "--no-deps"}.issubset(commands[3][1])
    for index, (_stage, command) in enumerate(commands):
        if index == 3:
            assert "--only-binary=:all:" not in command
        else:
            assert "--only-binary=:all:" in command
    qwen_command = commands[4][1]
    assert "--constraint" in qwen_command and "--requirement" in qwen_command
    assert "--upgrade" not in qwen_command
    assert f"torch=={record['torch']}" in qwen_command


def test_installed_version_probe_is_short_bounded_and_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command[:2] == ["venv-python", "-c"]
        assert kwargs == {
            "check": True,
            "stdin": subprocess.DEVNULL,
            "capture_output": True,
            "text": True,
            "timeout": setup_support.UTILITY_TIMEOUT_SECONDS,
        }
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"torch": "2.11.0", "torchaudio": "2.11.0"}),
            stderr="",
        )

    monkeypatch.setattr(setup_support.subprocess, "run", fake_run)
    assert setup_support.installed_versions(
        Path("venv-python"), ("torch", "torchaudio")
    ) == {"torch": "2.11.0", "torchaudio": "2.11.0"}


def test_installed_version_probe_timeout_has_stable_private_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_command = ["private-python", "-c", "private-script"]

    def timeout(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(private_command, kwargs["timeout"])

    monkeypatch.setattr(setup_support.subprocess, "run", timeout)
    with pytest.raises(setup_support.SetupSupportError) as failure:
        setup_support.installed_versions(Path("private-python"), ("torch",))

    assert failure.value.code == "SETUP_VERSION_INSPECTION_FAILED"
    exposed = failure.value.public_message + str(failure.value)
    assert "private-python" not in exposed
    assert "private-script" not in exposed


@pytest.mark.parametrize(
    ("fragment", "expected_code"),
    [
        (b"ERROR: ResolutionImpossible: conflicting dependencies", "SETUP_DEPENDENCY_CONFLICT"),
        (b"ERROR: No matching distribution found for native-wheel", "SETUP_WHEEL_UNAVAILABLE"),
        (b"WARNING: Retrying after ConnectionError: Max retries exceeded", "SETUP_NETWORK_FAILED"),
        (b"OSError: [Errno 28] No space left on device", "SETUP_STORAGE_FULL"),
        (b"unrecognized private command failure", "SETUP_COMMAND_FAILED"),
    ],
)
def test_run_checked_classifies_bounded_output_without_leaking_it(
    monkeypatch: pytest.MonkeyPatch,
    fragment: bytes,
    expected_code: str,
) -> None:
    private_path = b"/" + b"home" + b"/private-user/extension"
    secrets = (
        b"\n\x1b[31mhttps://user:hf_secret_value@example.invalid/simple\x1b[0m\n"
        + private_path
        + b"\n"
        + b"X" * setup_support.COMMAND_OUTPUT_LIMIT
    )
    monkeypatch.setattr(
        setup_support,
        "_bounded_command",
        lambda _command, **_limits: setup_support.CommandOutcome(
            returncode=19,
            output=fragment + secrets,
            truncated=True,
        ),
    )
    messages: list[str] = []
    with pytest.raises(setup_support.SetupSupportError) as failure:
        setup_support.run_checked(
            ["private-python", "private-command"],
            "Installing pinned native wheels",
            messages.append,
        )

    assert failure.value.code == expected_code
    exposed = "\n".join(messages + [failure.value.public_message, str(failure.value)])
    assert "Installing pinned native wheels" in exposed
    assert "exit code 19" in exposed
    for forbidden in (
        "https://",
        "hf_secret_value",
        private_path.decode(),
        "private-python",
        "private-command",
        "\x1b",
        "ResolutionImpossible",
        "No matching distribution",
    ):
        assert forbidden not in exposed


def test_run_checked_replaces_untrusted_stage_and_handles_invalid_utf8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_secret = "unsafe stage https://example.invalid token-value"
    monkeypatch.setattr(
        setup_support,
        "_bounded_command",
        lambda _command, **_limits: setup_support.CommandOutcome(
            7, b"\xff\xfe\x00private"
        ),
    )
    messages: list[str] = []
    with pytest.raises(setup_support.SetupSupportError) as failure:
        setup_support.run_checked(["secret-command"], stage_secret, messages.append)
    assert failure.value.code == "SETUP_COMMAND_FAILED"
    assert messages == ["Setup command"]
    exposed = failure.value.public_message + str(failure.value)
    assert "Setup command failed with exit code 7" in exposed
    assert stage_secret not in exposed
    assert "private" not in exposed
    assert "secret-command" not in exposed


def test_run_checked_timeout_uses_stable_actionable_private_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_output = b"ReadTimeout https://user:secret@example.invalid/private/path"

    def fake_bounded(command: list[str], **limits: object) -> setup_support.CommandOutcome:
        assert command == ["private-python", "private-command"]
        assert limits == {"timeout": setup_support.COMMAND_TIMEOUT_SECONDS}
        return setup_support.CommandOutcome(
            returncode=-9,
            output=private_output,
            timed_out=True,
        )

    monkeypatch.setattr(setup_support, "_bounded_command", fake_bounded)
    messages: list[str] = []
    with pytest.raises(setup_support.SetupSupportError) as failure:
        setup_support.run_checked(
            ["private-python", "private-command"],
            "Preparing isolated Python environment",
            messages.append,
        )

    assert failure.value.code == "SETUP_COMMAND_TIMEOUT"
    assert messages == ["Preparing isolated Python environment"]
    exposed = failure.value.public_message + str(failure.value)
    assert "exceeded its bounded setup time" in exposed
    assert "run Repair" in exposed
    for forbidden in (
        "ReadTimeout",
        "https://",
        "secret",
        "private/path",
        "private-python",
        "private-command",
    ):
        assert forbidden not in exposed


def test_dependency_installs_have_no_total_wall_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "constraints.txt").write_text("pins", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("pins", encoding="utf-8")
    flavor = setup_support.PlatformFlavor("linux", "x64", "cuda", "cu130", "13.0")
    calls: list[tuple[str, float | None]] = []

    monkeypatch.setattr(setup_support, "installed_versions", lambda *_args: {})
    monkeypatch.setattr(
        setup_support,
        "run_checked",
        lambda _command, stage, _log, *, timeout: calls.append((stage, timeout)),
    )
    monkeypatch.setattr(setup_support, "verify_pip_check", lambda *_args: None)

    setup_support.install_dependencies(
        Path("venv-python"), tmp_path, flavor, lambda _message: None
    )

    assert [stage for stage, _timeout in calls] == [
        "Pinning Python build tools",
        "Installing pinned PyTorch cu130 runtime",
        "Installing pinned native wheels",
        "Installing the pinned Python SoX wrapper",
        "Installing the pinned Qwen3-TTS runtime",
    ]
    assert {timeout for _stage, timeout in calls} == {None}


def test_unbounded_dependency_command_cannot_emit_wall_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float | None] = []

    def fake_bounded(
        _command: list[str], *, timeout: float | None
    ) -> setup_support.CommandOutcome:
        calls.append(timeout)
        return setup_support.CommandOutcome(returncode=0, output=b"")

    monkeypatch.setattr(setup_support, "_bounded_command", fake_bounded)
    messages: list[str] = []
    setup_support.run_checked(
        ["private-python", "-m", "pip", "install", "private-package"],
        "Installing pinned native wheels",
        messages.append,
        timeout=None,
    )

    assert calls == [None]
    assert messages == ["Installing pinned native wheels"]
    assert "SETUP_COMMAND_TIMEOUT" not in "\n".join(messages)


def test_bounded_command_kills_output_over_limit() -> None:
    script = (
        "import os;"
        f"os.write(1,b'private-prefix'+b'X'*{setup_support.COMMAND_OUTPUT_LIMIT * 4})"
    )
    outcome = setup_support._bounded_command(
        [sys.executable, "-c", script],
        output_limit=1024,
        timeout=10,
    )
    assert outcome.truncated is True
    assert len(outcome.output) == 1024


def _arm64_cu130_flavor() -> setup_support.PlatformFlavor:
    return setup_support.PlatformFlavor("linux", "arm64", "cuda", "cu130", "13.0")


def _cusparselt_pip_outcome(output: bytes | None = None) -> setup_support.CommandOutcome:
    return setup_support.CommandOutcome(
        1,
        output
        if output is not None
        else (setup_support.CUSPARSELT_PIP_CHECK_DIAGNOSTIC + "\n").encode(),
    )


def test_pip_check_accepts_only_verified_linux_arm64_cu130_sbsa_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        setup_support,
        "_cusparselt_sbsa_distribution_is_safe",
        lambda _python: True,
    )
    assert setup_support._pip_check_exception_allowed(
        Path("venv-python"),
        _arm64_cu130_flavor(),
        _cusparselt_pip_outcome(),
    )


@pytest.mark.parametrize(
    "flavor",
    [
        setup_support.PlatformFlavor("linux", "x64", "cuda", "cu130", "13.0"),
        setup_support.PlatformFlavor("linux", "arm64", "cuda", "cu128", "12.8"),
        setup_support.PlatformFlavor("win32", "x64", "cuda", "cu130", "13.0"),
        setup_support.PlatformFlavor("linux", "arm64", "cpu", "cu130", "13.0"),
    ],
)
def test_pip_check_sbsa_exception_rejects_every_other_platform_flavor(
    monkeypatch: pytest.MonkeyPatch,
    flavor: setup_support.PlatformFlavor,
) -> None:
    monkeypatch.setattr(
        setup_support,
        "_cusparselt_sbsa_distribution_is_safe",
        lambda _python: pytest.fail("metadata audit must not run for an ineligible flavor"),
    )
    assert not setup_support._pip_check_exception_allowed(
        Path("venv-python"), flavor, _cusparselt_pip_outcome()
    )


@pytest.mark.parametrize(
    "outcome",
    [
        setup_support.CommandOutcome(
            2,
            (setup_support.CUSPARSELT_PIP_CHECK_DIAGNOSTIC + "\n").encode(),
        ),
        setup_support.CommandOutcome(
            1,
            b"nvidia-cusparselt-cu13 0.8.1 is not supported on this platform\n",
        ),
        setup_support.CommandOutcome(
            1,
            (setup_support.CUSPARSELT_PIP_CHECK_DIAGNOSTIC + "\nextra issue\n").encode(),
        ),
        setup_support.CommandOutcome(
            1,
            ("\x1b[31m" + setup_support.CUSPARSELT_PIP_CHECK_DIAGNOSTIC + "\x1b[0m\n").encode(),
        ),
        setup_support.CommandOutcome(
            1,
            (setup_support.CUSPARSELT_PIP_CHECK_DIAGNOSTIC + "\n").encode(),
            truncated=True,
        ),
        setup_support.CommandOutcome(
            1,
            (setup_support.CUSPARSELT_PIP_CHECK_DIAGNOSTIC + "\n").encode(),
            timed_out=True,
        ),
    ],
)
def test_pip_check_sbsa_exception_rejects_noncanonical_or_extra_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    outcome: setup_support.CommandOutcome,
) -> None:
    monkeypatch.setattr(
        setup_support,
        "_cusparselt_sbsa_distribution_is_safe",
        lambda _python: pytest.fail("metadata audit must not run for invalid pip output"),
    )
    assert not setup_support._pip_check_exception_allowed(
        Path("venv-python"), _arm64_cu130_flavor(), outcome
    )


def test_pip_check_sbsa_exception_rejects_unsafe_or_wrong_tag_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        setup_support,
        "_cusparselt_sbsa_distribution_is_safe",
        lambda _python: False,
    )
    assert not setup_support._pip_check_exception_allowed(
        Path("venv-python"), _arm64_cu130_flavor(), _cusparselt_pip_outcome()
    )
    source = setup_support.CUSPARSELT_AUDIT_SCRIPT
    assert "unsafe_alias" in source
    assert "stat.S_ISREG" in source
    assert "base_canonical not in resolved.parents" in source
    assert "relative.parts == (__DIST_INFO__, \"WHEEL\")" in source
    assert "tags != [__WHEEL_TAG__]" in source
    assert setup_support.CUSPARSELT_DIST_INFO == "nvidia_cusparselt_cu13-0.8.0.dist-info"
    assert setup_support.CUSPARSELT_SBSA_WHEEL_TAG == "Tag: py3-none-manylinux2014_sbsa"


def test_cusparselt_metadata_audit_builds_isolated_exact_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_bounded(
        command: list[str],
        **limits: object,
    ) -> setup_support.CommandOutcome:
        assert command[:3] == ["venv-python", "-I", "-c"]
        compile(command[3], "<cusparselt-audit>", "exec")
        assert "__DISTRIBUTION__" not in command[3]
        assert repr(setup_support.CUSPARSELT_DISTRIBUTION) in command[3]
        assert repr(setup_support.CUSPARSELT_VERSION) in command[3]
        assert repr(setup_support.CUSPARSELT_DIST_INFO) in command[3]
        assert repr(setup_support.CUSPARSELT_SBSA_WHEEL_TAG) in command[3]
        assert limits == {"output_limit": 4096, "timeout": 30}
        return setup_support.CommandOutcome(
            0,
            setup_support.CUSPARSELT_AUDIT_SUCCESS,
        )

    monkeypatch.setattr(setup_support, "_bounded_command", fake_bounded)
    assert setup_support._cusparselt_sbsa_distribution_is_safe(Path("venv-python"))

    monkeypatch.setattr(
        setup_support,
        "_bounded_command",
        lambda *_args, **_kwargs: setup_support.CommandOutcome(0, b"unexpected"),
    )
    assert not setup_support._cusparselt_sbsa_distribution_is_safe(Path("venv-python"))


def test_final_pip_check_accepts_only_the_audited_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        setup_support,
        "_pip_check_outcome",
        lambda _python: _cusparselt_pip_outcome(),
    )
    monkeypatch.setattr(
        setup_support,
        "_cusparselt_sbsa_distribution_is_safe",
        lambda _python: True,
    )
    messages: list[str] = []
    setup_support.verify_pip_check(
        Path("venv-python"), _arm64_cu130_flavor(), messages.append
    )
    assert messages == [
        "Verifying dependency graph",
        "Accepted verified Linux ARM64 cu130 wheel-tag compatibility exception",
    ]

    monkeypatch.setattr(
        setup_support,
        "_pip_check_outcome",
        lambda _python: _cusparselt_pip_outcome(
            (setup_support.CUSPARSELT_PIP_CHECK_DIAGNOSTIC + "\nother conflict\n").encode()
        ),
    )
    with pytest.raises(setup_support.SetupSupportError, match="SETUP_WHEEL_UNAVAILABLE"):
        setup_support.verify_pip_check(
            Path("venv-python"), _arm64_cu130_flavor(), lambda _message: None
        )


def test_healthy_exact_environment_skips_installation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "constraints.txt").write_text("pins", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("pins", encoding="utf-8")
    flavor = setup_support.PlatformFlavor("linux", "x64", "cpu", "cpu", None)
    expected = setup_support.expected_distributions(flavor)
    monkeypatch.setattr(
        setup_support, "installed_versions", lambda _python, _names: dict(expected)
    )
    monkeypatch.setattr(
        setup_support,
        "pip_check_passes",
        lambda _python, _flavor, _log: True,
    )
    monkeypatch.setattr(
        setup_support,
        "run_checked",
        lambda *_args, **_kwargs: pytest.fail("healthy environment must skip installation"),
    )
    setup_support.install_dependencies(Path("venv-python"), tmp_path, flavor, lambda _message: None)


def test_health_script_covers_python_cpu_cuda_bf16_and_audio_checks() -> None:
    flavor = setup_support.PlatformFlavor("linux", "x64", "cpu", "cpu", None)
    source = setup_support._render_health_script(flavor)
    compile(source, "<health-script>", "exec")
    for needle in (
        "(3, 11)",
        "(3, 12)",
        "torch.float32",
        "torch.cuda.is_available",
        "torch.cuda.is_bf16_supported",
        "scaled_dot_product_attention",
        "torch.cuda.synchronize",
        "torchaudio.functional.resample",
        "CPUExecutionProvider",
        'subtype="PCM_16"',
        "librosa.resample",
        'shutil.which("sox")',
        "runtime_arch = _normalize_health_arch(platform.machine())",
        'runtime_system != flavor["system"]',
        "os.dup2(discard_fd, 1)",
        "os.dup2(discard_fd, 2)",
    ):
        assert needle in source
    for substage in setup_support.HEALTH_SUBSTAGES:
        assert repr(substage) in source
    assert setup_support.HEALTH_CATEGORIES == {
        "cuda_oom",
        "import",
        "cuda_unavailable",
        "native_probe",
        "invalid_result",
    }
    assert "__EXPECTED__" not in source
    assert "__FLAVOR__" not in source
    assert "get_device_capability" not in source


@pytest.mark.parametrize(
    ("raw_system", "raw_arch", "expected_system", "expected_arch"),
    [
        ("linux", "x86_64", "linux", "x64"),
        ("win32", "AMD64", "win32", "x64"),
        ("linux", "aarch64", "linux", "arm64"),
    ],
)
def test_isolated_health_platform_mapping_matches_public_normalization(
    raw_system: str,
    raw_arch: str,
    expected_system: str,
    expected_arch: str,
) -> None:
    namespace: dict[str, object] = {}
    exec(setup_support.HEALTH_PLATFORM_HELPERS, namespace)
    normalize_system = namespace["_normalize_health_system"]
    normalize_arch = namespace["_normalize_health_arch"]
    assert normalize_system(raw_system) == expected_system
    assert normalize_arch(raw_arch) == expected_arch
    assert normalize_system(raw_system) == setup_support.normalize_platform_name(raw_system)
    assert normalize_arch(raw_arch) == setup_support.normalize_architecture(raw_arch)
    assert setup_support.HEALTH_PLATFORM_HELPERS in setup_support.HEALTH_SCRIPT


def _health_success_outcome(payload: dict[str, object]) -> setup_support.CommandOutcome:
    envelope = {
        "schema": setup_support.HEALTH_SCHEMA,
        "facts": payload,
    }
    return setup_support.CommandOutcome(
        0,
        (json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )


def _health_error_outcome(
    category: str,
    substage: str,
    *,
    extra: dict[str, object] | None = None,
) -> setup_support.CommandOutcome:
    envelope: dict[str, object] = {
        "schema": setup_support.HEALTH_SCHEMA,
        "category": category,
        "substage": substage,
    }
    if extra:
        envelope.update(extra)
    return setup_support.CommandOutcome(
        1,
        (json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )


def test_health_orchestration_returns_only_public_record_and_warns_for_sox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flavor = setup_support.PlatformFlavor("linux", "x64", "cpu", "cpu", None)
    payload = {
        "system": "linux",
        "arch": "x64",
        "pythonMinor": "3.12",
        "accelerator": "cpu",
        "torchVariant": "cpu",
        "cudaBf16": False,
        "runtimeDtype": "float32",
        "attention": "sdpa",
        "externalSox": False,
    }

    def fake_bounded(
        command: list[str],
        **limits: object,
    ) -> setup_support.CommandOutcome:
        assert command[:3] == ["venv-python", "-I", "-c"]
        compile(command[3], "<health-script>", "exec")
        assert limits == {
            "output_limit": setup_support.HEALTH_OUTPUT_LIMIT,
            "timeout": setup_support.HEALTH_TIMEOUT_SECONDS,
        }
        return _health_success_outcome(payload)

    messages: list[str] = []
    monkeypatch.setattr(setup_support, "_bounded_command", fake_bounded)
    assert setup_support.verify_runtime(Path("venv-python"), flavor, messages.append) == payload
    assert any("external SoX" in message for message in messages)
    assert messages[-1] == "Runtime health checks passed"


@pytest.mark.parametrize(
    ("category", "substage", "expected_code", "action_fragment"),
    [
        ("cuda_oom", "float32_alloc", "SETUP_HEALTH_CUDA_OOM", "release GPU memory"),
        ("import", "imports_core", "SETUP_HEALTH_IMPORT_FAILED", "run Repair"),
        (
            "cuda_unavailable",
            "cuda_availability",
            "SETUP_HEALTH_CUDA_UNAVAILABLE",
            "accelerator runtime",
        ),
        (
            "native_probe",
            "soundfile",
            "SETUP_HEALTH_NATIVE_PROBE_FAILED",
            "platform support",
        ),
        (
            "invalid_result",
            "dependency_versions",
            "SETUP_HEALTH_INVALID_RESULT",
            "run Repair",
        ),
    ],
)
def test_health_failure_envelopes_map_to_stable_safe_codes(
    monkeypatch: pytest.MonkeyPatch,
    category: str,
    substage: str,
    expected_code: str,
    action_fragment: str,
) -> None:
    flavor = setup_support.PlatformFlavor("linux", "arm64", "cuda", "cu130", "13.0")
    monkeypatch.setattr(
        setup_support,
        "_bounded_command",
        lambda *_args, **_kwargs: _health_error_outcome(category, substage),
    )
    with pytest.raises(setup_support.SetupSupportError) as failure:
        setup_support.verify_runtime(Path("venv-python"), flavor, lambda _message: None)
    assert failure.value.code == expected_code
    assert f"health substage {substage}" in failure.value.public_message
    assert action_fragment in failure.value.public_message


def test_health_oom_classifier_handles_torch_and_cuda_allocation_forms() -> None:
    namespace: dict[str, object] = {}
    exec(setup_support.HEALTH_FAILURE_HELPERS, namespace)
    classify = namespace["_health_category"]
    torch_oom_type = type(
        "OutOfMemoryError",
        (RuntimeError,),
        {"__module__": "torch.cuda"},
    )
    assert classify(torch_oom_type("private detail"), "float32_alloc") == "cuda_oom"
    assert (
        classify(RuntimeError("CUDA failure: cudaErrorMemoryAllocation"), "float32_alloc")
        == "cuda_oom"
    )
    assert classify(RuntimeError("ordinary failure"), "float32_alloc") == "native_probe"


@pytest.mark.parametrize(
    "outcome",
    [
        _health_error_outcome(
            "cuda_oom",
            "float32_alloc",
            extra={"raw": "https://user:secret@example.invalid/private"},
        ),
        _health_error_outcome("unknown_category", "float32_alloc"),
        _health_error_outcome("cuda_oom", "unknown_substage"),
        setup_support.CommandOutcome(
            1,
            b'{"schema":"wrong","status":"error"}\n',
        ),
        setup_support.CommandOutcome(
            1,
            (
                '{"schema":"%s","schema":"%s",'
                '"category":"import","substage":"imports_core"}\n'
                % (setup_support.HEALTH_SCHEMA, setup_support.HEALTH_SCHEMA)
            ).encode(),
        ),
        setup_support.CommandOutcome(
            1,
            b"third-party raw banner\n"
            + _health_error_outcome("import", "imports_core").output,
        ),
        setup_support.CommandOutcome(
            1,
            b"X" * setup_support.HEALTH_OUTPUT_LIMIT,
            truncated=True,
        ),
    ],
)
def test_health_rejects_malicious_malformed_or_oversized_output_without_leakage(
    monkeypatch: pytest.MonkeyPatch,
    outcome: setup_support.CommandOutcome,
) -> None:
    secret = "https://user:secret@example.invalid/private"
    flavor = setup_support.PlatformFlavor("linux", "x64", "cpu", "cpu", None)
    monkeypatch.setattr(
        setup_support,
        "_bounded_command",
        lambda *_args, **_kwargs: outcome,
    )
    with pytest.raises(setup_support.SetupSupportError) as failure:
        setup_support.verify_runtime(Path("venv-python"), flavor, lambda _message: None)
    assert failure.value.code == "SETUP_HEALTH_FAILED"
    exposed = failure.value.public_message + str(failure.value)
    assert secret not in exposed
    assert "third-party raw banner" not in exposed
    assert "unknown_category" not in exposed
    assert "unknown_substage" not in exposed


def test_health_script_contains_native_fd_writes_and_emits_one_sanitized_envelope() -> None:
    flavor = setup_support.PlatformFlavor(
        setup_support.normalize_platform_name(sys.platform),
        setup_support.normalize_architecture(setup_support.platform.machine()),
        "cpu",
        "cpu",
        None,
    )
    script = setup_support._render_health_script(flavor)
    target = 'try:\n    substage = "platform_identity"\n'
    injected = (
        'try:\n    substage = "imports_core"\n'
        '    os.write(1, b"private-health-stdout")\n'
        '    os.write(2, b"private-health-stderr")\n'
        '    raise ImportError("private-health-exception")\n'
    )
    assert target in script
    script = script.replace(target, injected, 1)

    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=False,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=20,
    )

    assert completed.returncode == 1
    assert completed.stderr == b""
    assert b"private-health" not in completed.stdout
    [line] = completed.stdout.splitlines()
    envelope = json.loads(line)
    assert envelope == {
        "schema": setup_support.HEALTH_SCHEMA,
        "category": "import",
        "substage": "imports_core",
    }


@pytest.mark.parametrize("version", [[3, 11], [3, 12]])
def test_venv_verification_accepts_python_311_and_312(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: list[int],
) -> None:
    extension = tmp_path / "extension"
    python = extension / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"fixture")
    identity = setup_support.InterpreterIdentity(
        f"{version[0]}.{version[1]}",
        "linux",
        "x64",
        "cpython",
        f"cpython-{version[0]}{version[1]}",
        f"cp{version[0]}{version[1]}-linux-x64",
        os.path.normcase(os.path.realpath(extension / "venv")),
    )
    monkeypatch.setattr(setup, "validate_venv_interpreter", lambda *_args: identity)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command[1:] == ["-m", "pip", "--version"]
        assert kwargs == {
            "check": True,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "timeout": setup_support.UTILITY_TIMEOUT_SECONDS,
        }
        return subprocess.CompletedProcess(command, 0, stdout="pip fixture", stderr="")

    monkeypatch.setattr(setup.subprocess, "run", fake_run)
    assert setup.verify_python_runtime(python, extension) == tuple(version)


def test_venv_creation_rejects_alias_before_running_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = tmp_path / "extension"
    outside = tmp_path / "outside"
    extension.mkdir()
    outside.mkdir()
    try:
        (extension / "venv").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory aliases unavailable: {exc}")
    monkeypatch.setattr(
        setup_support,
        "run_checked",
        lambda *_args, **_kwargs: pytest.fail("unsafe venv must be rejected before execution"),
    )
    with pytest.raises(setup_support.SetupSupportError, match="SETUP_VENV_UNSAFE"):
        setup_support.create_or_reuse_venv(
            tmp_path / "python", extension, "linux", lambda _message: None
        )


def test_setup_aborts_before_venv_repair_when_state_parent_is_aliased(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = tmp_path / "extension"
    outside_venv = tmp_path / "outside-venv"
    models = tmp_path / "models"
    extension.mkdir()
    outside_venv.mkdir()
    models.mkdir()
    external_state = outside_venv / STATE_FILENAME
    external_state.write_text("external", encoding="utf-8")
    try:
        (extension / "venv").symlink_to(outside_venv, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory aliases unavailable: {exc}")
    binding = SimpleNamespace(extension_dir=extension, bootstrap_python=tmp_path / "python")
    flavor = setup_support.PlatformFlavor(NATIVE_SYSTEM, "x64", "cpu", "cpu", None)
    monkeypatch.setattr(setup, "bind_extension", lambda *_args: binding)
    monkeypatch.setattr(setup, "select_platform_flavor", lambda _context: flavor)
    monkeypatch.setattr(setup, "validate_running_platform", lambda _flavor: None)
    monkeypatch.setattr(setup, "resolve_models_root", lambda *_args, **_kwargs: models)
    monkeypatch.setattr(
        setup,
        "owned_model_directory",
        lambda *_args, **_kwargs: models / "owned-model",
    )
    monkeypatch.setattr(
        setup,
        "create_or_reuse_venv",
        lambda *_args: pytest.fail("unsafe state parent must abort before venv repair"),
    )

    with pytest.raises(StateError, match="STATE_VENV_UNSAFE"):
        setup.run_setup({"models_dir": str(models)})

    assert external_state.read_text(encoding="utf-8") == "external"


@pytest.mark.parametrize(
    "relationship",
    ["extension", "extension-child", "venv", "venv-child"],
)
def test_setup_rejects_models_root_in_extension_or_venv_before_any_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relationship: str,
) -> None:
    extension = tmp_path / "extension"
    venv = extension / "venv"
    venv.mkdir(parents=True)
    sentinel = venv / "sentinel.txt"
    sentinel.write_text("untouched", encoding="utf-8")
    models = {
        "extension": extension,
        "extension-child": extension / "weights",
        "venv": venv,
        "venv-child": venv / "nested" / "models",
    }[relationship]
    binding = SimpleNamespace(extension_dir=extension, bootstrap_python=tmp_path / "python")
    flavor = setup_support.PlatformFlavor(NATIVE_SYSTEM, "x64", "cpu", "cpu", None)
    monkeypatch.setattr(setup, "bind_extension", lambda *_args: binding)
    monkeypatch.setattr(setup, "select_platform_flavor", lambda _context: flavor)
    monkeypatch.setattr(setup, "validate_running_platform", lambda _flavor: None)
    monkeypatch.setattr(
        setup,
        "resolve_models_root",
        lambda *_args, **_kwargs: models,
    )
    monkeypatch.setattr(
        setup,
        "native_directory_path",
        lambda *_args, **_kwargs: pytest.fail("overlap must fail before models root creation"),
    )
    monkeypatch.setattr(
        setup,
        "invalidate_setup_state",
        lambda *_args: pytest.fail("overlap must fail before state invalidation"),
    )

    with pytest.raises(PathContractError, match="SETUP_STORAGE_OVERLAP"):
        setup.run_setup({"models_dir": str(models)})

    assert sentinel.read_text(encoding="utf-8") == "untouched"
    if relationship in {"extension-child", "venv-child"}:
        assert not models.exists()


def test_setup_storage_disjoint_accepts_linux_windows_and_extension_root_alias(
    tmp_path: Path,
) -> None:
    extension = tmp_path / "physical-extension"
    extension_alias = tmp_path / "extension-link"
    models = tmp_path / "models"
    (extension / "venv").mkdir(parents=True)
    models.mkdir()
    try:
        extension_alias.symlink_to(extension, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"extension aliases unavailable: {exc}")

    assert setup.validate_setup_storage_disjoint(models, extension_alias, NATIVE_SYSTEM) == (
        models / setup.EXTENSION_ID / setup.NODE_ID
    )
    assert setup.validate_setup_storage_disjoint(
        Path(r"C:\Modly\Models"),
        Path(r"C:\Modly\Extensions\Qwen"),
        "win32",
    ) == Path(r"C:\Modly\Models") / setup.EXTENSION_ID / setup.NODE_ID


def test_setup_storage_overlap_follows_models_root_alias_before_mutation(
    tmp_path: Path,
) -> None:
    extension = tmp_path / "extension"
    venv = extension / "venv"
    models_alias = tmp_path / "models-link"
    venv.mkdir(parents=True)
    sentinel = venv / "sentinel.txt"
    sentinel.write_text("untouched", encoding="utf-8")
    try:
        models_alias.symlink_to(venv, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory aliases unavailable: {exc}")

    with pytest.raises(PathContractError, match="SETUP_STORAGE_OVERLAP"):
        setup.validate_setup_storage_disjoint(models_alias, extension, NATIVE_SYSTEM)

    assert sentinel.read_text(encoding="utf-8") == "untouched"


@pytest.mark.parametrize("context", [{}, {"_legacy_setup": True}])
def test_setup_uses_shared_models_resolver_for_vanilla_and_legacy_and_writes_state_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    context: dict[str, object],
) -> None:
    calls: list[str] = []
    extension = tmp_path / "extension"
    models = tmp_path / "models"
    model_dir = models / "owner" / "node"
    binding = SimpleNamespace(extension_dir=extension, bootstrap_python=tmp_path / "bootstrap")
    flavor = setup_support.PlatformFlavor(NATIVE_SYSTEM, "x64", "cpu", "cpu", None)
    runtime_python = setup_support.venv_python(extension / "venv", NATIVE_SYSTEM)
    monkeypatch.setattr(setup, "bind_extension", lambda *_args: calls.append("bind") or binding)
    monkeypatch.setattr(setup, "select_platform_flavor", lambda _context: calls.append("flavor") or flavor)
    monkeypatch.setattr(setup, "validate_running_platform", lambda _flavor: calls.append("running"))

    def resolve(
        supplied: dict,
        extension_dir: Path,
        code_root: Path,
        system: str,
        **kwargs: object,
    ) -> Path:
        assert supplied == context
        assert extension_dir == extension
        assert code_root == setup.ROOT
        assert system == NATIVE_SYSTEM
        assert kwargs == {
            "payload_keys": setup.SETUP_MODELS_PAYLOAD_KEYS,
            "require_existing": False,
        }
        calls.append("models")
        return models

    monkeypatch.setattr(setup, "resolve_models_root", resolve)
    monkeypatch.setattr(
        setup,
        "native_directory_path",
        lambda value, label, system, **kwargs: (
            calls.append("models-create")
            or models
        ),
    )
    monkeypatch.setattr(setup, "owned_model_directory", lambda *_args, **_kwargs: calls.append("owned") or model_dir)
    monkeypatch.setattr(setup, "invalidate_setup_state", lambda _path: calls.append("invalidate"))
    monkeypatch.setattr(setup, "create_or_reuse_venv", lambda *_args: calls.append("venv") or runtime_python)
    monkeypatch.setattr(setup, "verify_python_runtime", lambda *_args: calls.append("python") or (3, 12))
    monkeypatch.setattr(setup, "install_dependencies", lambda *_args: calls.append("dependencies"))
    monkeypatch.setattr(setup, "verify_runtime", lambda *_args: calls.append("health") or {"pythonMinor": "3.12", "cudaBf16": False})
    monkeypatch.setattr(setup, "ensure_snapshot", lambda *_args, **_kwargs: calls.append("snapshot") or model_dir)
    monkeypatch.setattr(setup, "write_setup_state", lambda *_args: calls.append("state"))
    assert setup.run_setup(context) == model_dir
    assert calls == [
        "bind",
        "flavor",
        "running",
        "models",
        "models-create",
        "owned",
        "invalidate",
        "venv",
        "python",
        "dependencies",
        "health",
        "snapshot",
        "state",
    ]


def test_cli_hides_unexpected_exception_details(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sensitive = "fixture-private-command-and-path"
    monkeypatch.setattr(
        setup,
        "run_setup",
        lambda _context: (_ for _ in ()).throw(RuntimeError(sensitive)),
    )
    assert setup.cli(["setup.py", "{}"]) == 1
    captured = capsys.readouterr()
    assert "SETUP_UNEXPECTED" in captured.err
    assert sensitive not in captured.err


def test_cli_reports_actionable_interruption_without_timeout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        setup,
        "run_setup",
        lambda _context: setup._raise_setup_interrupted(15, None),
    )
    assert setup.cli(["setup.py", "{}"]) == 1
    captured = capsys.readouterr()
    assert "SETUP_INTERRUPTED" in captured.err
    assert "run Repair again" in captured.err
    assert "SETUP_COMMAND_TIMEOUT" not in captured.err


def test_cli_restores_catchable_host_signal_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restored: list[list[tuple[int, object]]] = []
    previous = [(15, object())]
    monkeypatch.setattr(setup, "_install_termination_handlers", lambda: previous)
    monkeypatch.setattr(
        setup,
        "_restore_termination_handlers",
        lambda handlers: restored.append(handlers),
    )
    monkeypatch.setattr(setup, "run_setup", lambda _context: None)

    assert setup.cli(["setup.py", "{}"]) == 0
    assert restored == [previous]


def test_cli_preserves_asset_code_but_hides_internal_asset_detail(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sensitive = "fixture-internal-absolute-detail"
    monkeypatch.setattr(
        setup,
        "run_setup",
        lambda _context: (_ for _ in ()).throw(AssetError("ASSET_DOWNLOAD_FAILED", sensitive)),
    )
    assert setup.cli(["setup.py", "{}"]) == 1
    captured = capsys.readouterr()
    assert "ASSET_DOWNLOAD_FAILED" in captured.err
    assert sensitive not in captured.err
