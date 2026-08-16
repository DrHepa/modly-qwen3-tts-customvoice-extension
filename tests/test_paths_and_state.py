from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path, PurePosixPath
import platform
import stat
import sys
from types import SimpleNamespace

import pytest

from qwen3_tts_modly.constants import EXTENSION_ID
from qwen3_tts_modly.paths import (
    PathContractError,
    bind_extension,
    explicit_models_root,
    normalize_configured_directory_path,
    normalize_architecture,
    normalize_platform_name,
    owned_model_directory,
    storage_paths_overlap,
)
from qwen3_tts_modly.setup_support import PlatformFlavor
from qwen3_tts_modly import state


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def mock_distribution_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(state, "_validate_distributions", lambda _flavor: None)


def installed_layout(tmp_path: Path) -> tuple[Path, Path]:
    extension = tmp_path / "extensions" / EXTENSION_ID
    python = tmp_path / "bootstrap" / "python"
    extension.mkdir(parents=True)
    (extension / "manifest.json").write_bytes((ROOT / "manifest.json").read_bytes())
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python fixture")
    return extension, python


def test_path_normalization_matches_linux_and_fully_qualified_windows_contract() -> None:
    assert normalize_configured_directory_path("/srv/modly/../models", "models", "linux") == "/srv/models"
    assert normalize_configured_directory_path(r"C:\\Modly\\..\\Models", "models", "win32") == r"C:\Models"
    assert normalize_configured_directory_path(r"\\server\share\models", "models", "windows") == r"\\server\share\models"
    for invalid in ("models", r"C:models", r"\\server"):
        with pytest.raises(PathContractError, match="PATH_ABSOLUTE_REQUIRED"):
            normalize_configured_directory_path(invalid, "models", "win32")
    with pytest.raises(PathContractError, match="PATH_ABSOLUTE_REQUIRED"):
        normalize_configured_directory_path("models", "models", "linux")
    assert normalize_platform_name("Windows") == "win32"
    assert normalize_architecture("AMD64") == "x64"
    assert normalize_architecture("aarch64") == "arm64"


def test_binding_accepts_actual_repository_and_installed_id_basenames(tmp_path: Path) -> None:
    assert ROOT.name == "modly-qwen3-tts-customvoice-extension"
    repository_binding = bind_extension(
        {
            "ext_dir": str(ROOT),
            "python_exe": sys.executable,
            "platform": sys.platform,
        },
        ROOT,
    )
    assert repository_binding.extension_dir == ROOT

    extension, python = installed_layout(tmp_path)
    context = {
        "ext_dir": str(extension),
        "python_exe": str(python),
        "platform": sys.platform,
    }
    binding = bind_extension(context, extension)
    assert binding.extension_dir == extension
    assert binding.bootstrap_python == python


def test_binding_accepts_extension_root_alias_but_rejects_another_root(tmp_path: Path) -> None:
    extension, python = installed_layout(tmp_path)
    alias = tmp_path / "repository-local-link"
    try:
        alias.symlink_to(extension, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory aliases unavailable: {exc}")
    context = {
        "ext_dir": str(alias),
        "python_exe": str(python),
        "platform": sys.platform,
    }
    binding = bind_extension(context, extension)
    assert binding.extension_dir == alias

    other = tmp_path / "other" / EXTENSION_ID
    other.mkdir(parents=True)
    (other / "manifest.json").write_bytes((ROOT / "manifest.json").read_bytes())
    with pytest.raises(PathContractError, match="PATH_EXTENSION_BINDING_MISMATCH"):
        bind_extension({**context, "ext_dir": str(extension)}, other)


def test_binding_rejects_manifest_alias_and_identity_mismatch(tmp_path: Path) -> None:
    extension, python = installed_layout(tmp_path)
    context = {
        "ext_dir": str(extension),
        "python_exe": str(python),
        "platform": sys.platform,
    }
    manifest_path = extension / "manifest.json"
    outside_manifest = tmp_path / "outside-manifest.json"
    outside_manifest.write_bytes(manifest_path.read_bytes())
    manifest_path.unlink()
    try:
        manifest_path.symlink_to(outside_manifest)
    except OSError as exc:
        pytest.skip(f"file aliases unavailable: {exc}")
    with pytest.raises(PathContractError, match="PATH_MANIFEST_UNSAFE"):
        bind_extension(context, extension)

    manifest_path.unlink()
    manifest = json.loads(outside_manifest.read_text(encoding="utf-8"))
    manifest["id"] = "different-extension"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PathContractError, match="PATH_MANIFEST_ID_MISMATCH"):
        bind_extension(context, extension)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda manifest: manifest.update(type="model"),
        lambda manifest: manifest.update(entry="other.py"),
        lambda manifest: manifest["nodes"][0].update(id="other-node"),
    ],
)
def test_binding_rejects_manifest_process_contract_mismatch(
    tmp_path: Path,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    extension, python = installed_layout(tmp_path)
    manifest_path = extension / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PathContractError, match="PATH_MANIFEST_CONTRACT_MISMATCH"):
        bind_extension(
            {
                "ext_dir": str(extension),
                "python_exe": str(python),
                "platform": sys.platform,
            },
            extension,
        )


def test_models_root_is_required_explicitly_and_never_inferred(tmp_path: Path) -> None:
    with pytest.raises(PathContractError, match="PATH_MODELS_REQUIRED"):
        explicit_models_root({}, "models_dir", sys.platform, create=True)
    models = tmp_path / "configured models"
    result = explicit_models_root(
        {"models_dir": str(models), "unrelated": str(tmp_path / "other")},
        "models_dir",
        sys.platform,
        create=True,
    )
    assert result == models
    assert result.is_dir()


def test_models_root_alias_is_allowed_but_alias_below_it_is_rejected(tmp_path: Path) -> None:
    physical = tmp_path / "physical"
    alias = tmp_path / "configured"
    physical.mkdir()
    try:
        alias.symlink_to(physical, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory aliases unavailable: {exc}")
    models_root = explicit_models_root(
        {"models_dir": str(alias)}, "models_dir", sys.platform, create=False
    )
    model_dir = owned_model_directory(models_root, create=True)
    assert model_dir.resolve().is_relative_to(physical.resolve())
    model_dir.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    model_dir.symlink_to(outside, target_is_directory=True)
    with pytest.raises(PathContractError, match="PATH_OWNED_ALIAS"):
        owned_model_directory(models_root, create=False)


def test_storage_overlap_handles_linux_and_windows_case_semantics() -> None:
    linux_models = PurePosixPath("/opt/modly/models")
    linux_workspace = PurePosixPath("/opt/modly/workspace")
    assert storage_paths_overlap(linux_models, linux_models / "owned", "linux")
    assert storage_paths_overlap(linux_models, linux_models, "linux")
    assert not storage_paths_overlap(linux_models, linux_workspace, "linux")

    assert storage_paths_overlap(
        r"C:\Modly\Models",
        r"c:\modly\models\extension\node",
        "win32",
    )
    assert not storage_paths_overlap(
        r"C:\Modly\Models",
        r"C:\Modly\Workspace",
        "win32",
    )
    assert not storage_paths_overlap(
        r"C:\Modly\Models",
        r"D:\Modly\Models",
        "win32",
    )


def test_storage_overlap_resolves_existing_alias_parent_without_mutation(
    tmp_path: Path,
) -> None:
    physical = tmp_path / "physical"
    alias = tmp_path / "alias"
    physical.mkdir()
    sentinel = physical / "sentinel.txt"
    sentinel.write_text("untouched", encoding="utf-8")
    try:
        alias.symlink_to(physical, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory aliases unavailable: {exc}")

    assert storage_paths_overlap(
        alias / "prospective" / "models",
        physical / "prospective",
        normalize_platform_name(sys.platform),
    )
    assert sentinel.read_text(encoding="utf-8") == "untouched"
    assert not (physical / "prospective").exists()


def _current_flavor() -> PlatformFlavor:
    return PlatformFlavor(
        system=normalize_platform_name(sys.platform),
        arch=normalize_architecture(platform.machine()),
        accelerator="cpu",
        torch_variant="cpu",
        expected_cuda=None,
    )


def prepared_state_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    extension = tmp_path / "extensions" / EXTENSION_ID
    venv = extension / "venv"
    models = tmp_path / "models"
    model_dir = models / EXTENSION_ID / "generate-speech"
    venv.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    (extension / "manifest.json").write_bytes((ROOT / "manifest.json").read_bytes())
    return extension, models, model_dir


def health_record() -> dict[str, object]:
    return {
        "pythonMinor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "cudaBf16": False,
    }


def test_setup_state_is_atomic_pathless_and_rederives_explicit_models_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension, models, model_dir = prepared_state_layout(tmp_path)
    written = state.write_setup_state(extension, _current_flavor(), health_record())
    parsed = json.loads(written.read_text(encoding="utf-8"))
    serialized = json.dumps(parsed)
    assert str(tmp_path) not in serialized
    assert "binding" not in parsed
    assert set(parsed) == {"schema", "extension", "platform", "model"}
    monkeypatch.setattr(state, "verify_snapshot", lambda *_args, **_kwargs: [])
    validated = state.validate_runtime_state(extension, models)
    assert validated.model_dir == model_dir.resolve()
    assert validated.flavor == _current_flavor()


def test_runtime_state_rejects_assets_manifest_and_secret_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension, models, _model_dir = prepared_state_layout(tmp_path)
    written = state.write_setup_state(extension, _current_flavor(), health_record())
    monkeypatch.setattr(state, "verify_snapshot", lambda *_args, **_kwargs: ["fixture failure"])
    with pytest.raises(state.StateError, match="STATE_ASSET_INVENTORY_INVALID"):
        state.validate_runtime_state(extension, models)

    monkeypatch.setattr(state, "verify_snapshot", lambda *_args, **_kwargs: [])
    manifest = json.loads((extension / "manifest.json").read_text(encoding="utf-8"))
    manifest["version"] = "9.9.9"
    (extension / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(state.StateError, match="STATE_EXTENSION_DRIFT"):
        state.validate_runtime_state(extension, models)

    (extension / "manifest.json").write_bytes((ROOT / "manifest.json").read_bytes())
    parsed = json.loads(written.read_text(encoding="utf-8"))
    parsed["authorization"] = "fixture-secret"
    written.write_text(json.dumps(parsed), encoding="utf-8")
    with pytest.raises(state.StateError, match="STATE_PRIVACY_INVALID"):
        state.validate_runtime_state(extension, models)


def test_state_rejects_absolute_string_even_under_unknown_key(tmp_path: Path) -> None:
    extension, models, _model_dir = prepared_state_layout(tmp_path)
    written = state.write_setup_state(extension, _current_flavor(), health_record())
    parsed = json.loads(written.read_text(encoding="utf-8"))
    parsed["extra"] = str(models.resolve())
    written.write_text(json.dumps(parsed), encoding="utf-8")
    with pytest.raises(state.StateError, match="STATE_PRIVACY_INVALID"):
        state.validate_runtime_state(extension, models)


def test_state_invalidation_rejects_alias(tmp_path: Path) -> None:
    extension, _models, _model_dir = prepared_state_layout(tmp_path)
    destination = state.state_path(extension)
    target = tmp_path / "target-state"
    target.write_text("{}", encoding="utf-8")
    try:
        destination.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"file aliases unavailable: {exc}")
    with pytest.raises(state.StateError, match="STATE_PATH_UNSAFE"):
        state.invalidate_setup_state(extension)


def test_state_invalidation_rejects_aliased_venv_parent_without_deleting_external_state(
    tmp_path: Path,
) -> None:
    extension = tmp_path / "extension"
    outside_venv = tmp_path / "outside-venv"
    extension.mkdir()
    outside_venv.mkdir()
    external_state = outside_venv / state.STATE_FILENAME
    external_state.write_text("external state", encoding="utf-8")
    try:
        (extension / "venv").symlink_to(outside_venv, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory aliases unavailable: {exc}")

    with pytest.raises(state.StateError, match="STATE_VENV_UNSAFE"):
        state.invalidate_setup_state(extension)

    assert external_state.read_text(encoding="utf-8") == "external state"


def test_state_invalidation_allows_extension_root_alias_but_not_venv_alias(
    tmp_path: Path,
) -> None:
    physical = tmp_path / "physical-extension"
    alias = tmp_path / "extension-alias"
    venv = physical / "venv"
    venv.mkdir(parents=True)
    destination = venv / state.STATE_FILENAME
    destination.write_text("generated", encoding="utf-8")
    try:
        alias.symlink_to(physical, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"extension aliases unavailable: {exc}")

    state.invalidate_setup_state(alias)

    assert not destination.exists()


def test_state_alias_detection_includes_windows_reparse_attribute() -> None:
    regular_directory = SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0)
    windows_reparse = SimpleNamespace(
        st_mode=stat.S_IFDIR,
        st_file_attributes=state.WINDOWS_REPARSE_ATTRIBUTE,
    )
    assert not state._link_or_reparse(regular_directory)
    assert state._link_or_reparse(windows_reparse)


def test_distribution_drift_check_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    flavor = _current_flavor()
    expected = state.expected_distributions(flavor)
    monkeypatch.undo()
    monkeypatch.setattr(
        state.importlib_metadata,
        "version",
        lambda name: "0.0.0" if name == "qwen-tts" else expected[name],
    )
    with pytest.raises(state.StateError, match="STATE_DEPENDENCY_DRIFT"):
        state._validate_distributions(flavor)
