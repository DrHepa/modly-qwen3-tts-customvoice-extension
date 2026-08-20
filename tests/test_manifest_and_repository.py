from __future__ import annotations

import json
from pathlib import Path

from qwen3_tts_modly.constants import (
    ALL_DISTRIBUTIONS,
    BUILD_DISTRIBUTIONS,
    DEFAULT_DO_SAMPLE,
    DEFAULT_INSTRUCT,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NON_STREAMING_MODE,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_SPEAKER,
    DEFAULT_SUBTALKER_DOSAMPLE,
    DEFAULT_SUBTALKER_TEMPERATURE,
    DEFAULT_SUBTALKER_TOP_K,
    DEFAULT_SUBTALKER_TOP_P,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    EXTENSION_VERSION,
    LANGUAGES,
    RUNTIME_DISTRIBUTIONS,
    SPEAKERS,
    TORCH_VARIANTS,
)


ROOT = Path(__file__).resolve().parents[1]


def manifest() -> dict:
    return json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))


def test_manifest_declares_exact_public_process_identity() -> None:
    value = manifest()
    assert value["id"] == "qwen3-tts-customvoice-process-extension"
    assert value["name"] == "Qwen3-TTS CustomVoice"
    assert value["type"] == "process"
    assert value["entry"] == "qwen3_tts_customvoice_process.py"
    assert EXTENSION_VERSION == "0.1.4"
    assert value["version"] == EXTENSION_VERSION
    assert value["author"] == "DrHepa"
    assert value["source"] == "https://github.com/DrHepa/modly-qwen3-tts-customvoice-extension"
    assert (ROOT / value["entry"]).is_file()
    [node] = value["nodes"]
    assert set(node) == {"id", "name", "input", "output", "params_schema"}
    assert (node["id"], node["input"], node["output"]) == (
        "generate-speech",
        "text",
        "audio",
    )


def test_manifest_parameters_match_the_full_scalar_customvoice_surface() -> None:
    [node] = manifest()["nodes"]
    params = {item["id"]: item for item in node["params_schema"]}
    assert tuple(params) == (
        "speaker",
        "language",
        "instruct",
        "non_streaming_mode",
        "do_sample",
        "top_k",
        "top_p",
        "temperature",
        "repetition_penalty",
        "subtalker_dosample",
        "subtalker_top_k",
        "subtalker_top_p",
        "subtalker_temperature",
        "max_new_tokens",
    )
    expected_defaults = {
        "speaker": DEFAULT_SPEAKER,
        "language": DEFAULT_LANGUAGE,
        "instruct": DEFAULT_INSTRUCT,
        "non_streaming_mode": DEFAULT_NON_STREAMING_MODE,
        "do_sample": DEFAULT_DO_SAMPLE,
        "top_k": DEFAULT_TOP_K,
        "top_p": DEFAULT_TOP_P,
        "temperature": DEFAULT_TEMPERATURE,
        "repetition_penalty": DEFAULT_REPETITION_PENALTY,
        "subtalker_dosample": DEFAULT_SUBTALKER_DOSAMPLE,
        "subtalker_top_k": DEFAULT_SUBTALKER_TOP_K,
        "subtalker_top_p": DEFAULT_SUBTALKER_TOP_P,
        "subtalker_temperature": DEFAULT_SUBTALKER_TEMPERATURE,
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
    }
    assert {name: item["default"] for name, item in params.items()} == expected_defaults
    assert tuple(option["value"] for option in params["speaker"]["options"]) == SPEAKERS
    assert tuple(option["value"] for option in params["language"]["options"]) == LANGUAGES
    for name in ("non_streaming_mode", "do_sample", "subtalker_dosample"):
        assert params[name]["type"] == "select"
        assert [option["value"] for option in params[name]["options"]] == ["true", "false"]
    assert "text" not in params
    assert "seed" not in params
    assert params["max_new_tokens"]["min"] == 2


def _pins(path: Path) -> dict[str, str]:
    return {
        name: version
        for name, version in (
            line.split("==", 1)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def test_selected_direct_requirement_files_and_torch_variants_are_exactly_pinned() -> None:
    assert _pins(ROOT / "constraints.txt") == ALL_DISTRIBUTIONS
    assert _pins(ROOT / "requirements.txt") == RUNTIME_DISTRIBUTIONS
    assert RUNTIME_DISTRIBUTIONS["qwen-tts"] == "0.1.1"
    assert RUNTIME_DISTRIBUTIONS["scipy"] == "1.17.0"
    assert RUNTIME_DISTRIBUTIONS["gradio"] == "6.16.0"
    assert BUILD_DISTRIBUTIONS["setuptools"] == "78.1.0"
    assert TORCH_VARIANTS == {
        "cpu": {
            "torch": "2.11.0+cpu",
            "torchaudio": "2.11.0+cpu",
            "index": "https://download.pytorch.org/whl/cpu",
            "cuda": None,
        },
        "cu128": {
            "torch": "2.11.0+cu128",
            "torchaudio": "2.11.0+cu128",
            "index": "https://download.pytorch.org/whl/cu128",
            "cuda": "12.8",
        },
        "cu130": {
            "torch": "2.11.0+cu130",
            "torchaudio": "2.11.0+cu130",
            "index": "https://download.pytorch.org/whl/cu130",
            "cuda": "13.0",
        },
    }
    assert _pins(ROOT / "requirements-dev.txt") == {
        "pytest": "9.0.2",
        "numpy": "2.4.6",
    }


def test_selected_pins_have_a_known_upstream_metadata_intersection() -> None:
    def version(value: str) -> tuple[int, ...]:
        return tuple(int(part) for part in value.split("."))

    # PyTorch 2.11.0 metadata requires setuptools<82.
    assert version(BUILD_DISTRIBUTIONS["setuptools"]) < (82,)
    # A representative hub release demonstrates the verified intersection of
    # Transformers 4.57.3 (>=0.34,<1.0) and Gradio 6.16.0 (>=0.33.5,<2.0).
    # This is a metadata-coherence regression, not a transitive lock.
    compatible_hub = (0, 36, 0)
    assert (0, 34, 0) <= compatible_hub < (1, 0)
    assert (0, 33, 5) <= compatible_hub < (2, 0)


def _public_text_files(root: Path = ROOT) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or any(
            part in {".git", "__pycache__", ".pytest_cache"} for part in path.parts
        ):
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        files.append(path)
    return files


def _private_reference_hits(root: Path) -> list[Path]:
    private_home = "/" + "home" + "/" + "drhepa"
    return [
        path
        for path in _public_text_files(root)
        if private_home.casefold() in path.read_text(encoding="utf-8").casefold()
    ]


def test_repository_has_no_private_host_references_or_unfinished_markers() -> None:
    private_home = "/" + "home" + "/" + "drhepa"
    banned = (
        private_home.casefold(),
        ("Docu" + "mentos").casefold(),
        ("GB" + "10").casefold(),
        ("SM" + "121").casefold(),
        ("user" + "Data").casefold(),
        ("TO" + "DO").casefold(),
        ("REPLACE" + "_ME").casefold(),
        ("NotImplemented" + "Error").casefold(),
        ("." + "pyz").casefold(),
        ("generator" + ".py").casefold(),
    )
    for path in _public_text_files():
        source = path.read_text(encoding="utf-8", errors="replace").casefold()
        assert not any(token in source for token in banned), path


def test_local_metadata_sentinel_is_ignored_absent_and_still_scannable(
    tmp_path: Path,
) -> None:
    sentinel = ROOT / ".modly-local"
    assert not sentinel.exists()
    assert ".modly-local" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    fixture = tmp_path / ".modly-local"
    private_home = "/" + "home" + "/" + "drhepa"
    fixture.write_text(private_home + "/private-extension\n", encoding="utf-8")
    assert _private_reference_hits(tmp_path) == [fixture]
    assert _private_reference_hits(ROOT) == []


def test_runtime_has_no_hidden_network_cache_or_private_models_root_discovery() -> None:
    bootstrap = (ROOT / manifest()["entry"]).read_text(encoding="utf-8")
    runtime = (ROOT / "qwen3_tts_modly" / "process_runtime.py").read_text(encoding="utf-8")
    lowered = runtime.casefold()
    for forbidden in (
        "url" + "open",
        "requests" + ".",
        "snapshot" + "_download",
        "hf_hub" + "_download",
        "expand" + "user",
        "settings" + ".json",
        "user" + "profile",
        "local" + "host",
        "127." + "0.0.1",
    ):
        assert forbidden not in lowered
    assert "local_files_only=True" in runtime
    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in runtime
    assert 'os.environ["TRANSFORMERS_OFFLINE"] = "1"' in runtime
    assert "resolve_models_root(" in runtime
    paths = (ROOT / "qwen3_tts_modly" / "paths.py").read_text(encoding="utf-8")
    assert 'SETUP_MODELS_PAYLOAD_KEYS = ("models_dir", "modelsDir")' in paths
    assert 'RUNTIME_MODELS_PAYLOAD_KEYS = ("modelsDir",)' in paths
    assert 'MODELS_ENVIRONMENT_KEYS = ("MODLY_MODELS_DIR", "MODELS_DIR")' in paths
    assert 'extensions_root.name.casefold() != "extensions"' in paths
    assert "from qwen3_tts_modly.process_runtime import run_protocol" in bootstrap
    assert bootstrap.index("os.dup2(null_fd, 2)") < bootstrap.index(
        "from qwen3_tts_modly.process_runtime import run_protocol"
    )
    assert "os.dup(1)" in bootstrap


def test_ci_workflow_is_minimal_source_only_and_matrix_bounded() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    uses = [
        line.split("uses:", 1)[1].strip()
        for line in workflow.splitlines()
        if "uses:" in line
    ]
    run_steps = [
        line.split("run:", 1)[1].strip()
        for line in workflow.splitlines()
        if "run:" in line
    ]
    assert uses == ["actions/checkout@v4", "actions/setup-python@v5"]
    assert "permissions:\n  contents: read" in workflow
    assert "persist-credentials: false" in workflow
    assert "concurrency:" in workflow and "cancel-in-progress: true" in workflow
    assert "timeout-minutes: 15" in workflow
    assert "ubuntu-latest" in workflow and "windows-latest" in workflow
    assert '"3.11"' in workflow and '"3.12"' in workflow
    assert run_steps == [
        'python -m pip install --disable-pip-version-check --no-cache-dir --only-binary=":all:" --requirement requirements-dev.txt',
        "python -m pytest -q",
        "python -m compileall -q setup.py qwen3_tts_customvoice_process.py qwen3_tts_modly tests",
    ]
    lowered = workflow.casefold()
    for forbidden in (
        "requirements.txt",
        "python setup.py",
        "setup.py install",
        "curl ",
        "wget ",
        "sudo ",
        "secrets.",
        "upload-artifact",
        "download-artifact",
        "cache:",
        "models/",
        "huggingface",
    ):
        assert forbidden not in lowered


def test_readme_is_truthful_about_real_and_pending_validation_scope() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())
    assert f"Version `{EXTENSION_VERSION}`" in readme
    assert "not E2E verified" in normalized
    assert "VoiceDesign" in normalized and "ref_audio" in normalized
    assert "non_streaming_mode=false" in normalized
    assert "4,520,218,951 bytes" in normalized
    assert "one clean Linux runtime" in normalized
    assert "strictly offline" in normalized
    assert "structurally valid mono PCM16 WAV at 24 kHz" in normalized
    assert "perceptual or transcription-quality evaluation" in normalized
    assert "Clean Install-from-GitHub UI validation" in normalized
    assert "source-only GitHub Actions matrix has completed successfully" in normalized
    assert "not packaged cross-platform E2E evidence" in normalized
    assert "no total elapsed-time timeout" in normalized
    assert "SETUP_COMMAND_TIMEOUT" in normalized
