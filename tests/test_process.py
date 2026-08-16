from __future__ import annotations

import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import wave

import numpy as np
import pytest

from qwen3_tts_modly import process_runtime as process
from qwen3_tts_modly.setup_support import PlatformFlavor
from qwen3_tts_modly.state import RuntimeState


CPU_FLAVOR = PlatformFlavor("linux", "x64", "cpu", "cpu", None)
ROOT = Path(__file__).resolve().parents[1]


def payload(tmp_path: Path, *, text: object = "Hello from Qwen.", params: object = None) -> dict:
    models = tmp_path / "models"
    workspace = tmp_path / "workspace"
    temporary = tmp_path / "temp"
    models.mkdir(exist_ok=True)
    workspace.mkdir(exist_ok=True)
    temporary.mkdir(exist_ok=True)
    return {
        "input": {"text": text, "nodeId": "generate-speech"},
        "params": {} if params is None else params,
        "nodeId": "generate-speech",
        "modelsDir": str(models.resolve()),
        "workspaceDir": str(workspace.resolve()),
        "tempDir": str(temporary.resolve()),
    }


def runtime_state(model_dir: Path, flavor: PlatformFlavor = CPU_FLAVOR) -> RuntimeState:
    return RuntimeState(model_dir=model_dir.resolve(), flavor=flavor, payload={})


def fake_state(model_dir: Path, calls: list[str] | None = None):
    def validate(_root: Path, models_dir: Path) -> RuntimeState:
        if calls is not None:
            calls.append("state")
        assert models_dir.name == "models"
        return runtime_state(model_dir)

    return validate


def test_request_defaults_match_manifest_and_require_explicit_models_dir(tmp_path: Path) -> None:
    request = process.validate_request(payload(tmp_path))
    assert request.params == process.RequestParameters(
        speaker="Vivian",
        language="Auto",
        instruct="",
        non_streaming_mode=True,
        do_sample=True,
        top_k=50,
        top_p=1.0,
        temperature=0.9,
        repetition_penalty=1.05,
        subtalker_dosample=True,
        subtalker_top_k=50,
        subtalker_top_p=1.0,
        subtalker_temperature=0.9,
        max_new_tokens=8192,
    )
    missing = payload(tmp_path)
    missing.pop("modelsDir")
    with pytest.raises(process.ProcessFailure, match="REQUEST_PATH_INVALID"):
        process.validate_request(missing)


@pytest.mark.parametrize(
    "relationship",
    [
        "workspace-inside-owned",
        "temp-inside-owned",
        "owned-inside-workspace",
        "owned-inside-temp",
    ],
)
def test_runtime_storage_overlap_fails_before_state_or_output_mutation(
    tmp_path: Path,
    relationship: str,
) -> None:
    models = tmp_path / "models"
    workspace = tmp_path / "workspace"
    temporary = tmp_path / "temp"
    if relationship == "workspace-inside-owned":
        workspace = models / process.EXTENSION_ID / process.NODE_ID / "workspace"
    elif relationship == "temp-inside-owned":
        temporary = models / process.EXTENSION_ID / process.NODE_ID / "temp"
    elif relationship == "owned-inside-workspace":
        models = workspace / "models"
    elif relationship == "owned-inside-temp":
        models = temporary / "models"
    models.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=True, exist_ok=True)
    owned = models / process.EXTENSION_ID / process.NODE_ID
    owned.mkdir(parents=True, exist_ok=True)
    sentinel = owned / "sentinel.txt"
    sentinel.write_text("untouched", encoding="utf-8")
    value = {
        "input": {"text": "Storage boundary fixture.", "nodeId": process.NODE_ID},
        "params": {},
        "nodeId": process.NODE_ID,
        "modelsDir": str(models.resolve()),
        "workspaceDir": str(workspace.resolve()),
        "tempDir": str(temporary.resolve()),
    }

    def handler(request: dict, emitter: process.ProtocolEmitter) -> Path:
        return process.handle_request(
            request,
            emitter,
            state_validator=lambda *_args: pytest.fail(
                "storage overlap must fail before state or asset verification"
            ),
        )

    output = io.StringIO()
    assert process.run_protocol(io.StringIO(json.dumps(value) + "\n"), output, handler) == 1
    messages = [json.loads(line) for line in output.getvalue().splitlines()]
    assert messages[-1]["type"] == "error"
    assert "REQUEST_STORAGE_OVERLAP" in messages[-1]["message"]
    assert "Diagnostic:" not in messages[-1]["message"]
    assert sentinel.read_text(encoding="utf-8") == "untouched"
    assert not (workspace / "Workflows").exists()


def test_runtime_storage_overlap_resolves_models_alias_without_diagnostics(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    temporary = tmp_path / "temp"
    models_alias = tmp_path / "models-link"
    workspace.mkdir()
    temporary.mkdir()
    try:
        models_alias.symlink_to(workspace, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory aliases unavailable: {exc}")
    value = {
        "input": {"text": "Alias fixture.", "nodeId": process.NODE_ID},
        "params": {},
        "nodeId": process.NODE_ID,
        "modelsDir": str(models_alias.absolute()),
        "workspaceDir": str(workspace.resolve()),
        "tempDir": str(temporary.resolve()),
    }
    output = io.StringIO()

    assert process.run_protocol(io.StringIO(json.dumps(value) + "\n"), output) == 1

    [terminal] = [
        json.loads(line)
        for line in output.getvalue().splitlines()
        if json.loads(line).get("type") in {"done", "error"}
    ]
    assert terminal["type"] == "error"
    assert "REQUEST_STORAGE_OVERLAP" in terminal["message"]
    assert "Diagnostic:" not in terminal["message"]
    assert not (workspace / "Workflows").exists()


def test_runtime_storage_overlap_preserves_lexical_workspace_alias_boundary(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    owned = models / process.EXTENSION_ID / process.NODE_ID
    actual_workspace = tmp_path / "actual-workspace"
    temporary = tmp_path / "temp"
    owned.mkdir(parents=True)
    actual_workspace.mkdir()
    temporary.mkdir()
    workspace_alias = owned / "workspace-link"
    try:
        workspace_alias.symlink_to(actual_workspace, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory aliases unavailable: {exc}")
    sentinel = owned / "sentinel.txt"
    sentinel.write_text("untouched", encoding="utf-8")
    value = {
        "input": {"text": "Lexical alias fixture.", "nodeId": process.NODE_ID},
        "params": {},
        "nodeId": process.NODE_ID,
        "modelsDir": str(models.resolve()),
        "workspaceDir": str(workspace_alias.absolute()),
        "tempDir": str(temporary.resolve()),
    }

    def handler(request: dict, emitter: process.ProtocolEmitter) -> Path:
        return process.handle_request(
            request,
            emitter,
            state_validator=lambda *_args: pytest.fail(
                "lexical overlap must fail before state or asset verification"
            ),
        )

    output = io.StringIO()
    assert process.run_protocol(io.StringIO(json.dumps(value) + "\n"), output, handler) == 1
    messages = [json.loads(line) for line in output.getvalue().splitlines()]
    assert messages[-1]["type"] == "error"
    assert "REQUEST_STORAGE_OVERLAP" in messages[-1]["message"]
    assert "Diagnostic:" not in messages[-1]["message"]
    assert sentinel.read_text(encoding="utf-8") == "untouched"
    assert not (actual_workspace / "Workflows").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["input"].update(text=["one", "two"]),
        lambda value: value["input"].update(texts=["one", "two"]),
        lambda value: value.update(input=[{"text": "one"}]),
    ],
)
def test_batch_inputs_are_explicitly_rejected(tmp_path: Path, mutation: object) -> None:
    value = payload(tmp_path)
    mutation(value)
    with pytest.raises(process.ProcessFailure) as failure:
        process.validate_request(value)
    assert failure.value.code == "REQUEST_BATCH_UNSUPPORTED"


@pytest.mark.parametrize(
    "params",
    [
        {"speaker": "Unknown"},
        {"language": "Beijing"},
        {"non_streaming_mode": True},
        {"do_sample": "TRUE"},
        {"top_k": -1},
        {"top_k": 1.0},
        {"top_p": 0},
        {"top_p": 1.01},
        {"top_p": float("nan")},
        {"temperature": 0},
        {"repetition_penalty": -0.1},
        {"subtalker_dosample": False},
        {"subtalker_top_k": -1},
        {"subtalker_top_p": 0},
        {"subtalker_temperature": 0},
        {"max_new_tokens": 0},
        {"max_new_tokens": 1},
        {"max_new_tokens": True},
        {"seed": 7},
    ],
)
def test_parameter_guards_fail_closed(tmp_path: Path, params: dict) -> None:
    with pytest.raises(process.ProcessFailure) as failure:
        process.validate_request(payload(tmp_path, params=params))
    expected = "REQUEST_PARAM_UNSUPPORTED" if "seed" in params else "REQUEST_PARAM_INVALID"
    assert failure.value.code == expected


def test_full_custom_parameters_normalize_and_forward_exactly(tmp_path: Path) -> None:
    params = {
        "speaker": "Sohee",
        "language": "Korean",
        "instruct": "Calm and clear.",
        "non_streaming_mode": "false",
        "do_sample": "false",
        "top_k": 0,
        "top_p": 0.75,
        "temperature": 1.2,
        "repetition_penalty": 1.1,
        "subtalker_dosample": "false",
        "subtalker_top_k": 3,
        "subtalker_top_p": 0.8,
        "subtalker_temperature": 0.7,
        "max_new_tokens": 4096,
    }
    request = process.validate_request(payload(tmp_path, params=params))
    assert request.params.generation_kwargs(request.text) == {
        "text": "Hello from Qwen.",
        "speaker": "Sohee",
        "language": "Korean",
        "instruct": "Calm and clear.",
        "non_streaming_mode": False,
        "do_sample": False,
        "top_k": 0,
        "top_p": 0.75,
        "temperature": 1.2,
        "repetition_penalty": 1.1,
        "subtalker_dosample": False,
        "subtalker_top_k": 3,
        "subtalker_top_p": 0.8,
        "subtalker_temperature": 0.7,
        "max_new_tokens": 4096,
    }


def test_empty_instruction_is_omitted_from_upstream_kwargs(tmp_path: Path) -> None:
    request = process.validate_request(payload(tmp_path))
    kwargs = request.params.generation_kwargs(request.text)
    assert "instruct" not in kwargs
    assert kwargs["max_new_tokens"] == 8192


def test_cpu_and_cuda_dtype_policies_use_float32_and_probed_bf16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    float32 = object()
    bfloat16 = object()
    torch = SimpleNamespace(
        float32=float32,
        bfloat16=bfloat16,
        version=SimpleNamespace(cuda="13.0"),
        cuda=SimpleNamespace(is_available=lambda: True),
        device=lambda value: f"device:{value}",
    )
    assert process.runtime_device_policy(torch, CPU_FLAVOR) == ("cpu", float32)
    cuda_flavor = PlatformFlavor("linux", "x64", "cuda", "cu130", "13.0")
    monkeypatch.setattr(process, "_probe_cuda_bf16", lambda *_args: True)
    assert process.runtime_device_policy(torch, cuda_flavor) == ("cuda:0", bfloat16)
    monkeypatch.setattr(process, "_probe_cuda_bf16", lambda *_args: False)
    assert process.runtime_device_policy(torch, cuda_flavor) == ("cuda:0", float32)
    torch.version.cuda = "12.8"
    with pytest.raises(process.ProcessFailure, match="RUNTIME_ACCELERATOR_FAILED"):
        process.runtime_device_policy(torch, cuda_flavor)


def test_load_runtime_uses_only_local_snapshot_sdpa_and_cpu_float32(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict]] = []
    torch_module = ModuleType("torch")
    torch_module.float32 = object()
    qwen_module = ModuleType("qwen_tts")

    class FakeQwen:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> object:
            calls.append((path, kwargs))
            return object()

    qwen_module.Qwen3TTSModel = FakeQwen
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "qwen_tts", qwen_module)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    loaded_np, model = process.load_runtime(model_dir.resolve(), CPU_FLAVOR)
    assert loaded_np is np and model is not None
    assert calls == [
        (
            str(model_dir.resolve()),
            {
                "device_map": "cpu",
                "dtype": torch_module.float32,
                "attn_implementation": "sdpa",
                "local_files_only": True,
            },
        )
    ]


def test_valid_protocol_captures_third_party_output_and_writes_pcm16_wav(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    calls: list[dict[str, object]] = []

    class FakeModel:
        def generate_custom_voice(self, **kwargs: object) -> tuple[list[np.ndarray], int]:
            print("third-party stdout banner")
            print("third-party stderr banner", file=sys.stderr)
            calls.append(kwargs)
            return [np.array([-1.5, -0.5, 0.0, 0.5, 1.5], dtype=np.float32)], 24_000

    def loader(path: Path, flavor: PlatformFlavor) -> tuple[object, FakeModel]:
        print("loader banner")
        assert path == model_dir.resolve() and flavor == CPU_FLAVOR
        return np, FakeModel()

    def handler(value: dict, emitter: process.ProtocolEmitter) -> Path:
        return process.handle_request(
            value,
            emitter,
            state_validator=fake_state(model_dir),
            runtime_loader=loader,
        )

    output = io.StringIO()
    code = process.run_protocol(
        io.StringIO(json.dumps(payload(tmp_path)) + "\n"), output, handler
    )
    messages = [json.loads(line) for line in output.getvalue().splitlines()]
    assert code == 0
    assert messages[-1]["type"] == "done"
    assert sum(message["type"] in {"done", "error"} for message in messages) == 1
    assert set(messages[-1]["result"]) == {"filePath"}
    result = Path(messages[-1]["result"]["filePath"])
    assert result.is_absolute() and result.is_file()
    assert result.parent == tmp_path / "workspace" / "Workflows" / "Qwen3-TTS-CustomVoice"
    assert not list((tmp_path / "temp").iterdir())
    with wave.open(str(result), "rb") as handle:
        assert (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) == (1, 2, 24_000)
        assert np.frombuffer(handle.readframes(5), dtype="<i2").tolist() == [
            -32767,
            -16384,
            0,
            16384,
            32767,
        ]
    assert calls == [
        {
            "text": "Hello from Qwen.",
            "speaker": "Vivian",
            "language": "Auto",
            "non_streaming_mode": True,
            "do_sample": True,
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 0.9,
            "repetition_penalty": 1.05,
            "subtalker_dosample": True,
            "subtalker_top_k": 50,
            "subtalker_top_p": 1.0,
            "subtalker_temperature": 0.9,
            "max_new_tokens": 8192,
        }
    ]
    captured = capsys.readouterr()
    assert "third-party" not in captured.out
    assert "third-party" not in captured.err


def test_state_and_assets_are_validated_before_runtime_load(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    calls: list[str] = []

    class FakeModel:
        def generate_custom_voice(self, **_kwargs: object) -> tuple[list[np.ndarray], int]:
            calls.append("generate")
            return [np.array([0.0], dtype=np.float32)], 24_000

    def loader(_path: Path, _flavor: PlatformFlavor) -> tuple[object, FakeModel]:
        calls.append("load")
        return np, FakeModel()

    output = io.StringIO()
    value = payload(tmp_path)

    def handler(request: dict, emitter: process.ProtocolEmitter) -> Path:
        return process.handle_request(
            request,
            emitter,
            state_validator=fake_state(model_dir, calls),
            runtime_loader=loader,
        )

    assert process.run_protocol(io.StringIO(json.dumps(value) + "\n"), output, handler) == 0
    assert calls == ["state", "load", "generate"]


def test_runtime_error_emits_one_terminal_and_privacy_minimal_diagnostic(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    private_text = "Private sentence fixture."
    private_instruction = "Private instruction fixture."
    secret = "hf_abcdefghijk123456"
    value = payload(
        tmp_path,
        text=private_text,
        params={"speaker": "Vivian", "language": "English", "instruct": private_instruction},
    )

    class FailingModel:
        def generate_custom_voice(self, **_kwargs: object) -> object:
            print(f"library output {private_text} {private_instruction} {secret}")
            raise RuntimeError(f"exception {private_text} {secret} {value['workspaceDir']}")

    def handler(request: dict, emitter: process.ProtocolEmitter) -> Path:
        return process.handle_request(
            request,
            emitter,
            state_validator=fake_state(model_dir),
            runtime_loader=lambda _path, _flavor: (np, FailingModel()),
        )

    output = io.StringIO()
    assert process.run_protocol(io.StringIO(json.dumps(value) + "\n"), output, handler) == 1
    messages = [json.loads(line) for line in output.getvalue().splitlines()]
    terminals = [message for message in messages if message["type"] in {"done", "error"}]
    assert len(terminals) == 1 and terminals[0] == messages[-1]
    public = terminals[0]["message"]
    assert "GENERATION_RUNTIME_FAILED" in public
    assert "Workflows/Qwen3-TTS-CustomVoice/Diagnostics/" in public
    for forbidden in (private_text, private_instruction, secret, value["workspaceDir"]):
        assert forbidden not in public
    [diagnostic] = list(
        (tmp_path / "workspace" / "Workflows" / "Qwen3-TTS-CustomVoice" / "Diagnostics").glob("*.log")
    )
    content = diagnostic.read_text(encoding="utf-8")
    assert "code: GENERATION_RUNTIME_FAILED" in content
    assert set(line.split(":", 1)[0] for line in content.splitlines()[1:]) == {
        "run_id",
        "code",
        "stage",
        "action",
    }
    for forbidden in (private_text, private_instruction, secret, str(tmp_path), "RuntimeError"):
        assert forbidden not in content


@pytest.mark.parametrize(
    ("waveforms", "sample_rate", "code"),
    [
        ([], 24_000, "GENERATION_WAVEFORM_COUNT_INVALID"),
        ([np.array([0.0])], 22_050, "GENERATION_SAMPLE_RATE_INVALID"),
        ([np.array([0.0])], 24_000.5, "GENERATION_SAMPLE_RATE_INVALID"),
        ([np.array([[0.0]])], 24_000, "GENERATION_WAVEFORM_INVALID"),
        ([np.array([np.nan])], 24_000, "GENERATION_WAVEFORM_NONFINITE"),
    ],
)
def test_model_output_validation_is_fail_closed(
    tmp_path: Path,
    waveforms: object,
    sample_rate: int,
    code: str,
) -> None:
    request = process.validate_request(payload(tmp_path))

    class FakeModel:
        def generate_custom_voice(self, **_kwargs: object) -> tuple[object, int]:
            return waveforms, sample_rate

    with pytest.raises(process.ProcessFailure) as failure:
        process.synthesize(
            request,
            runtime_state(tmp_path),
            runtime_loader=lambda _path, _flavor: (np, FakeModel()),
        )
    assert failure.value.code == code


def test_output_rejects_workspace_alias_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    temporary = tmp_path / "temp"
    outside = tmp_path / "outside"
    workspace.mkdir()
    temporary.mkdir()
    outside.mkdir()
    try:
        (workspace / "Workflows").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory aliases unavailable: {exc}")
    with pytest.raises(process.ProcessFailure) as failure:
        process.write_wav(
            np,
            np.array([0.0, 0.5], dtype=np.float32),
            workspace,
            temporary,
        )
    assert failure.value.code == "OUTPUT_DIRECTORY_INVALID"
    assert list(outside.iterdir()) == []


def test_invalid_request_still_emits_exactly_one_terminal_error(tmp_path: Path) -> None:
    value = payload(tmp_path, text="")
    output = io.StringIO()
    assert process.run_protocol(io.StringIO(json.dumps(value) + "\n"), output) == 1
    messages = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [message["type"] for message in messages if message["type"] in {"done", "error"}] == [
        "error"
    ]
    assert "REQUEST_TEXT_EMPTY" in messages[-1]["message"]


def test_protocol_rejects_existing_result_outside_workflow_directory(tmp_path: Path) -> None:
    value = payload(tmp_path)
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"fixture")
    output = io.StringIO()
    assert process.run_protocol(
        io.StringIO(json.dumps(value) + "\n"),
        output,
        handler=lambda _payload, _emitter: outside.resolve(),
    ) == 1
    messages = [json.loads(line) for line in output.getvalue().splitlines()]
    assert messages[-1]["type"] == "error"
    assert "OUTPUT_RESULT_INVALID" in messages[-1]["message"]
    assert str(outside) not in messages[-1]["message"]


def test_offline_environment_is_forced_at_import_boundary() -> None:
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_DATASETS_OFFLINE"] == "1"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"


def _terminal_messages(stdout: str) -> list[dict[str, object]]:
    messages = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    return [message for message in messages if message.get("type") in {"done", "error"}]


def _isolated_subprocess_environment() -> dict[str, str]:
    environment = {
        "PYTHONPATH": "",
        "PYTHONNOUSERSITE": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    if "SYSTEMROOT" in os.environ:
        environment["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    return environment


def test_bootstrap_import_failure_emits_one_sanitized_terminal(
    tmp_path: Path,
) -> None:
    entry = tmp_path / "qwen3_tts_customvoice_process.py"
    shutil.copyfile(ROOT / "qwen3_tts_customvoice_process.py", entry)

    completed = subprocess.run(
        [sys.executable, str(entry)],
        input="{}\n",
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
        env=_isolated_subprocess_environment(),
        timeout=20,
    )

    terminals = _terminal_messages(completed.stdout)
    assert completed.returncode == 1
    assert completed.stderr == ""
    assert terminals == [
        {
            "type": "error",
            "message": (
                "[PROCESS_BOOTSTRAP_FAILED] Qwen3-TTS CustomVoice initialization failed. "
                "Run Repair and try again."
            ),
        }
    ]
    assert len(completed.stdout.splitlines()) == 1
    assert str(tmp_path) not in completed.stdout
    assert "Traceback" not in completed.stdout


def test_bootstrap_discards_native_fd_writes_during_import_and_runtime(
    tmp_path: Path,
) -> None:
    entry = tmp_path / "qwen3_tts_customvoice_process.py"
    shutil.copyfile(ROOT / "qwen3_tts_customvoice_process.py", entry)
    package = tmp_path / "qwen3_tts_modly"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    secret = "NATIVE_FD_PRIVATE_FIXTURE"
    (package / "process_runtime.py").write_text(
        "import os\n"
        f"os.write(1, b'{secret}_IMPORT_OUT')\n"
        f"os.write(2, b'{secret}_IMPORT_ERR')\n"
        "def run_protocol(input_stream, output_stream):\n"
        f"    os.write(1, b'{secret}_RUN_OUT')\n"
        f"    os.write(2, b'{secret}_RUN_ERR')\n"
        f"    raise RuntimeError('{secret}_EXCEPTION')\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, str(entry)],
        input="{}\n",
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
        env=_isolated_subprocess_environment(),
        timeout=20,
    )

    terminals = _terminal_messages(completed.stdout)
    assert completed.returncode == 1
    assert completed.stderr == ""
    assert len(terminals) == 1
    assert terminals[0]["type"] == "error"
    assert "PROCESS_BOOTSTRAP_FAILED" in str(terminals[0]["message"])
    assert len(completed.stdout.splitlines()) == 1
    assert secret not in completed.stdout
    assert secret not in completed.stderr
