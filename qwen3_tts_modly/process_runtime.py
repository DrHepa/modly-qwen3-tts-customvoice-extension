"""Internal offline Qwen3-TTS CustomVoice PROCESS implementation."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
from typing import Any, Callable, Mapping, TextIO
import uuid
import wave


# Set offline policy before qwen_tts, Transformers, or Hugging Face Hub can import.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

ROOT = Path(os.path.normpath(os.path.abspath(__file__))).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen3_tts_modly.constants import (
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
    DIAGNOSTICS_DIRECTORY_NAME,
    EXTENSION_ID,
    EXTENSION_NAME,
    LANGUAGES,
    NODE_ID,
    OUTPUT_RELATIVE_DIRECTORY,
    SAMPLE_RATE,
    SPEAKERS,
    STATE_FILENAME,
)
from qwen3_tts_modly.diagnostics import (
    BoundedTextCapture,
    ensure_workspace_subdirectory,
    write_diagnostic,
)
from qwen3_tts_modly.paths import (
    PathContractError,
    current_platform_name,
    native_directory_path,
    require_storage_disjoint,
)
from qwen3_tts_modly.setup_support import PlatformFlavor
from qwen3_tts_modly.state import RuntimeState, validate_runtime_state


ALLOWED_PARAMS = frozenset(
    {
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
    }
)
WINDOWS_REPARSE_ATTRIBUTE = 0x400


ERROR_CATALOG: dict[str, tuple[str, str]] = {
    "REQUEST_STDIN_EMPTY": ("request validation", "Run one generate-speech node from Modly."),
    "REQUEST_COUNT_INVALID": ("request validation", "Run one generate-speech execution at a time."),
    "REQUEST_JSON_INVALID": ("request validation", "Run the node again from Modly."),
    "REQUEST_TYPE_INVALID": ("request validation", "Run the node again from Modly."),
    "REQUEST_INPUT_INVALID": ("request validation", "Connect one text input and try again."),
    "REQUEST_BATCH_UNSUPPORTED": ("request validation", "Provide one scalar text input."),
    "REQUEST_NODE_INVALID": ("request validation", "Run this extension's generate-speech node."),
    "REQUEST_TEXT_EMPTY": ("request validation", "Provide non-empty text and try again."),
    "REQUEST_PARAMS_INVALID": ("request validation", "Reset the node parameters and try again."),
    "REQUEST_PARAM_UNSUPPORTED": ("request validation", "Reset unsupported node parameters."),
    "REQUEST_PARAM_INVALID": ("request validation", "Correct the node parameter values and try again."),
    "REQUEST_PATH_INVALID": ("request validation", "Restart Modly and verify its configured storage paths."),
    "REQUEST_STORAGE_OVERLAP": (
        "storage validation",
        "Configure separate model, workspace, and temporary storage roots.",
    ),
    "STATE_VALIDATION_FAILED": ("setup validation", "Run Repair for this extension and try again."),
    "RUNTIME_ACCELERATOR_FAILED": ("runtime validation", "Verify the selected accelerator runtime or run Repair."),
    "GENERATION_RUNTIME_FAILED": ("model generation", "Run Repair; then try generation again."),
    "GENERATION_SAMPLE_RATE_INVALID": ("model output validation", "Run Repair and try generation again."),
    "GENERATION_WAVEFORM_COUNT_INVALID": ("model output validation", "Run one scalar text generation again."),
    "GENERATION_WAVEFORM_INVALID": ("model output validation", "Try different text; then run Repair if needed."),
    "GENERATION_WAVEFORM_NONFINITE": ("model output validation", "Run Repair and try generation again."),
    "OUTPUT_DIRECTORY_INVALID": ("output preparation", "Verify workspace and temporary-directory permissions."),
    "OUTPUT_WRITE_FAILED": ("WAV write", "Check workspace storage and permissions, then try again."),
    "OUTPUT_WAV_INVALID": ("WAV validation", "Run Repair and try generation again."),
    "OUTPUT_RESULT_INVALID": ("result validation", "Check workspace permissions and try again."),
    "GENERATION_UNEXPECTED": ("generation", "Run Repair; then try generation again."),
}


class ProcessFailure(RuntimeError):
    """An allowlisted public failure with no raw exception or request data."""

    def __init__(self, code: str) -> None:
        if code not in ERROR_CATALOG:
            code = "GENERATION_UNEXPECTED"
        self.code = code
        self.stage, self.action = ERROR_CATALOG[code]
        super().__init__(code)

    def public_message(self, diagnostic: str | None) -> str:
        suffix = f" Diagnostic: {diagnostic}." if diagnostic else ""
        return f"[{self.code}] {EXTENSION_NAME} {self.stage} failed. {self.action}{suffix}"


@dataclass(frozen=True)
class RequestParameters:
    speaker: str
    language: str
    instruct: str
    non_streaming_mode: bool
    do_sample: bool
    top_k: int
    top_p: float
    temperature: float
    repetition_penalty: float
    subtalker_dosample: bool
    subtalker_top_k: int
    subtalker_top_p: float
    subtalker_temperature: float
    max_new_tokens: int

    def generation_kwargs(self, text: str) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "text": text,
            "speaker": self.speaker,
            "language": self.language,
            "non_streaming_mode": self.non_streaming_mode,
            "do_sample": self.do_sample,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "subtalker_dosample": self.subtalker_dosample,
            "subtalker_top_k": self.subtalker_top_k,
            "subtalker_top_p": self.subtalker_top_p,
            "subtalker_temperature": self.subtalker_temperature,
            "max_new_tokens": self.max_new_tokens,
        }
        if self.instruct:
            kwargs["instruct"] = self.instruct
        return kwargs


@dataclass(frozen=True)
class ValidatedRequest:
    text: str
    params: RequestParameters
    models_dir: Path
    workspace_dir: Path
    temp_dir: Path


def validate_runtime_storage_disjoint(
    models_dir: Path,
    workspace_dir: Path,
    temp_dir: Path,
    *,
    code_root: Path = ROOT,
) -> Path:
    """Validate runtime storage ownership before state checks or output creation."""

    owned_model_root = models_dir / EXTENSION_ID / NODE_ID
    workflows = workspace_dir / "Workflows"
    output = workspace_dir.joinpath(*OUTPUT_RELATIVE_DIRECTORY)
    mutable_runtime_paths = (
        workspace_dir,
        workflows,
        output,
        output / DIAGNOSTICS_DIRECTORY_NAME,
        temp_dir,
        code_root / "venv",
        code_root / "venv" / STATE_FILENAME,
    )
    try:
        require_storage_disjoint(
            (models_dir, owned_model_root),
            mutable_runtime_paths,
            current_platform_name(),
            code="RUNTIME_STORAGE_OVERLAP",
            public_message="runtime model storage overlaps mutable execution storage",
        )
    except PathContractError as exc:
        if exc.code == "RUNTIME_STORAGE_OVERLAP":
            raise ProcessFailure("REQUEST_STORAGE_OVERLAP") from exc
        raise ProcessFailure("REQUEST_PATH_INVALID") from exc
    return owned_model_root


class ProtocolEmitter:
    """Reserve stdout for NDJSON and enforce exactly one terminal event."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._terminal = False

    def _write(self, payload: dict[str, Any]) -> None:
        if self._terminal:
            raise RuntimeError("protocol output attempted after terminal event")
        print(json.dumps(payload, ensure_ascii=True), file=self._stream, flush=True)

    def progress(self, percent: int, label: str) -> None:
        self._write({"type": "progress", "percent": percent, "label": label})

    def log(self, message: str) -> None:
        self._write({"type": "log", "message": message})

    def done(self, path: Path) -> None:
        self._write({"type": "done", "result": {"filePath": str(path)}})
        self._terminal = True

    def error(self, message: str) -> None:
        self._write({"type": "error", "message": message})
        self._terminal = True


def _read_one_payload(stream: TextIO) -> dict[str, Any]:
    line = stream.readline()
    if not line:
        raise ProcessFailure("REQUEST_STDIN_EMPTY")
    if stream.read().strip():
        raise ProcessFailure("REQUEST_COUNT_INVALID")
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProcessFailure("REQUEST_JSON_INVALID") from exc
    if not isinstance(payload, dict):
        raise ProcessFailure("REQUEST_TYPE_INVALID")
    return payload


def _absolute_directory(payload: Mapping[str, Any], key: str) -> Path:
    try:
        return native_directory_path(
            payload.get(key),
            key,
            current_platform_name(),
            must_exist=True,
        )
    except BaseException as exc:
        raise ProcessFailure("REQUEST_PATH_INVALID") from exc


def _parse_bool(value: object) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ProcessFailure("REQUEST_PARAM_INVALID")


def _parse_int(value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProcessFailure("REQUEST_PARAM_INVALID")
    return value


def _parse_float(
    value: object,
    *,
    positive: bool = False,
    probability: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProcessFailure("REQUEST_PARAM_INVALID")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ProcessFailure("REQUEST_PARAM_INVALID")
    if positive and parsed <= 0:
        raise ProcessFailure("REQUEST_PARAM_INVALID")
    if probability and not (0 < parsed <= 1):
        raise ProcessFailure("REQUEST_PARAM_INVALID")
    return parsed


def _parse_parameters(params: Mapping[str, Any]) -> RequestParameters:
    unknown = set(params) - ALLOWED_PARAMS
    if unknown:
        raise ProcessFailure("REQUEST_PARAM_UNSUPPORTED")
    speaker = params.get("speaker", DEFAULT_SPEAKER)
    language = params.get("language", DEFAULT_LANGUAGE)
    instruct = params.get("instruct", DEFAULT_INSTRUCT)
    if not isinstance(speaker, str) or speaker not in SPEAKERS:
        raise ProcessFailure("REQUEST_PARAM_INVALID")
    if not isinstance(language, str) or language not in LANGUAGES:
        raise ProcessFailure("REQUEST_PARAM_INVALID")
    if not isinstance(instruct, str):
        raise ProcessFailure("REQUEST_PARAM_INVALID")
    return RequestParameters(
        speaker=speaker,
        language=language,
        instruct=instruct,
        non_streaming_mode=_parse_bool(
            params.get("non_streaming_mode", DEFAULT_NON_STREAMING_MODE)
        ),
        do_sample=_parse_bool(params.get("do_sample", DEFAULT_DO_SAMPLE)),
        top_k=_parse_int(params.get("top_k", DEFAULT_TOP_K), minimum=0),
        top_p=_parse_float(params.get("top_p", DEFAULT_TOP_P), probability=True),
        temperature=_parse_float(
            params.get("temperature", DEFAULT_TEMPERATURE), positive=True
        ),
        repetition_penalty=_parse_float(
            params.get("repetition_penalty", DEFAULT_REPETITION_PENALTY), positive=True
        ),
        subtalker_dosample=_parse_bool(
            params.get("subtalker_dosample", DEFAULT_SUBTALKER_DOSAMPLE)
        ),
        subtalker_top_k=_parse_int(
            params.get("subtalker_top_k", DEFAULT_SUBTALKER_TOP_K), minimum=0
        ),
        subtalker_top_p=_parse_float(
            params.get("subtalker_top_p", DEFAULT_SUBTALKER_TOP_P), probability=True
        ),
        subtalker_temperature=_parse_float(
            params.get("subtalker_temperature", DEFAULT_SUBTALKER_TEMPERATURE), positive=True
        ),
        max_new_tokens=_parse_int(
            params.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS), minimum=2
        ),
    )


def validate_request(payload: Mapping[str, Any]) -> ValidatedRequest:
    input_data = payload.get("input")
    if isinstance(input_data, list):
        raise ProcessFailure("REQUEST_BATCH_UNSUPPORTED")
    if not isinstance(input_data, dict):
        raise ProcessFailure("REQUEST_INPUT_INVALID")
    if "texts" in input_data or isinstance(input_data.get("text"), list):
        raise ProcessFailure("REQUEST_BATCH_UNSUPPORTED")
    declared_ids = [value for value in (payload.get("nodeId"), input_data.get("nodeId")) if value not in (None, "")]
    if not declared_ids or any(not isinstance(value, str) or value != NODE_ID for value in declared_ids):
        raise ProcessFailure("REQUEST_NODE_INVALID")
    text = input_data.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ProcessFailure("REQUEST_TEXT_EMPTY")
    params = payload.get("params", {})
    if not isinstance(params, dict):
        raise ProcessFailure("REQUEST_PARAMS_INVALID")
    models_dir = _absolute_directory(payload, "modelsDir")
    workspace_candidate = _absolute_directory(payload, "workspaceDir")
    temp_candidate = _absolute_directory(payload, "tempDir")
    # Preserve the host's lexical paths until both lexical and canonical
    # overlap checks have completed. Resolving an aliased workspace or temp
    # path first could otherwise hide that its lexical entry lives inside the
    # extension-owned snapshot.
    validate_runtime_storage_disjoint(models_dir, workspace_candidate, temp_candidate)
    workspace_dir = workspace_candidate.resolve(strict=True)
    temp_dir = temp_candidate.resolve(strict=True)
    return ValidatedRequest(
        text=text,
        params=_parse_parameters(params),
        models_dir=models_dir,
        workspace_dir=workspace_dir,
        temp_dir=temp_dir,
    )


def _probe_cuda_bf16(torch: Any, device: Any) -> bool:
    try:
        if not torch.cuda.is_bf16_supported():
            return False
        import torch.nn.functional as functional

        a = torch.randn((8, 8), device=device, dtype=torch.bfloat16)
        b = torch.randn((8, 8), device=device, dtype=torch.bfloat16)
        matmul = a @ b
        conv = functional.conv2d(
            torch.randn((1, 2, 8, 8), device=device, dtype=torch.bfloat16),
            torch.randn((4, 2, 3, 3), device=device, dtype=torch.bfloat16),
        )
        q = torch.randn((1, 2, 4, 8), device=device, dtype=torch.bfloat16)
        sdpa = functional.scaled_dot_product_attention(q, q, q)
        torch.cuda.synchronize(device)
        return all(
            bool(torch.isfinite(value.float()).all().item())
            for value in (matmul, conv, sdpa)
        )
    except Exception:
        return False


def runtime_device_policy(torch: Any, flavor: PlatformFlavor) -> tuple[str, Any]:
    if flavor.accelerator == "cpu":
        return "cpu", torch.float32
    if flavor.accelerator != "cuda" or not torch.cuda.is_available():
        raise ProcessFailure("RUNTIME_ACCELERATOR_FAILED")
    if str(torch.version.cuda or "") != str(flavor.expected_cuda or ""):
        raise ProcessFailure("RUNTIME_ACCELERATOR_FAILED")
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if _probe_cuda_bf16(torch, device) else torch.float32
    return "cuda:0", dtype


def load_runtime(model_dir: Path, flavor: PlatformFlavor) -> tuple[Any, Any]:
    """Load one absolute local snapshot with SDPA and no network fallback."""

    if not model_dir.is_absolute() or not model_dir.is_dir():
        raise ProcessFailure("STATE_VALIDATION_FAILED")

    import numpy as np
    import torch
    from qwen_tts import Qwen3TTSModel

    device_map, dtype = runtime_device_policy(torch, flavor)
    model = Qwen3TTSModel.from_pretrained(
        str(model_dir),
        device_map=device_map,
        dtype=dtype,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    return np, model


RuntimeLoader = Callable[[Path, PlatformFlavor], tuple[Any, Any]]
StateValidator = Callable[[Path, Path], RuntimeState]


def _validate_generation_output(np: Any, waveforms: object, sample_rate: object) -> Any:
    try:
        numeric_rate = float(sample_rate)
        valid_rate = (
            not isinstance(sample_rate, bool)
            and math.isfinite(numeric_rate)
            and numeric_rate == float(SAMPLE_RATE)
        )
    except (TypeError, ValueError, OverflowError):
        valid_rate = False
    if not valid_rate:
        raise ProcessFailure("GENERATION_SAMPLE_RATE_INVALID")
    if not isinstance(waveforms, (list, tuple)) or len(waveforms) != 1:
        raise ProcessFailure("GENERATION_WAVEFORM_COUNT_INVALID")
    try:
        samples = np.asarray(waveforms[0], dtype=np.float32)
    except Exception as exc:
        raise ProcessFailure("GENERATION_WAVEFORM_INVALID") from exc
    if samples.ndim != 1 or samples.size == 0:
        raise ProcessFailure("GENERATION_WAVEFORM_INVALID")
    if not bool(np.isfinite(samples).all()):
        raise ProcessFailure("GENERATION_WAVEFORM_NONFINITE")
    return samples


def synthesize(
    request: ValidatedRequest,
    runtime_state: RuntimeState,
    *,
    runtime_loader: RuntimeLoader = load_runtime,
) -> tuple[Any, Any]:
    capture = BoundedTextCapture()
    try:
        with redirect_stdout(capture), redirect_stderr(capture):
            np, model = runtime_loader(runtime_state.model_dir, runtime_state.flavor)
            waveforms, sample_rate = model.generate_custom_voice(
                **request.params.generation_kwargs(request.text)
            )
            samples = _validate_generation_output(np, waveforms, sample_rate)
    except ProcessFailure:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        raise ProcessFailure("GENERATION_RUNTIME_FAILED") from exc
    return np, samples


def _validate_wav(path: Path, workspace_root: Path) -> None:
    try:
        info = path.lstat()
        attributes = getattr(info, "st_file_attributes", 0)
        if (
            stat.S_ISLNK(info.st_mode)
            or attributes & WINDOWS_REPARSE_ATTRIBUTE
            or not stat.S_ISREG(info.st_mode)
        ):
            raise OSError("invalid output type")
        resolved = path.resolve(strict=True)
        if workspace_root not in resolved.parents:
            raise OSError("output containment failure")
        with wave.open(str(resolved), "rb") as handle:
            valid = (
                handle.getnchannels() == 1
                and handle.getsampwidth() == 2
                and handle.getframerate() == SAMPLE_RATE
                and handle.getnframes() > 0
            )
    except (OSError, EOFError, wave.Error) as exc:
        raise ProcessFailure("OUTPUT_WAV_INVALID") from exc
    if not valid:
        raise ProcessFailure("OUTPUT_WAV_INVALID")


def write_wav(np: Any, samples: Any, workspace_dir: Path, temp_dir: Path) -> Path:
    try:
        workspace_root, output_dir = ensure_workspace_subdirectory(
            workspace_dir, OUTPUT_RELATIVE_DIRECTORY
        )
        temp_root = temp_dir.resolve(strict=True)
        if not temp_root.is_dir():
            raise OSError("temporary root unavailable")
    except OSError as exc:
        raise ProcessFailure("OUTPUT_DIRECTORY_INVALID") from exc

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    destination = output_dir / f"qwen3-tts-{stamp}-{uuid.uuid4().hex[:12]}.wav"
    staged = output_dir / f".{destination.name}.{uuid.uuid4().hex}.part"
    temporary = temp_root / f"qwen3-tts-{uuid.uuid4().hex}.wav.tmp"
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.rint(clipped * 32767.0).astype("<i2", copy=False)
    try:
        with temporary.open("xb") as raw:
            with wave.open(raw, "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(SAMPLE_RATE)
                handle.writeframes(pcm.tobytes(order="C"))
            raw.flush()
            os.fsync(raw.fileno())
        if temp_root not in temporary.resolve(strict=True).parents:
            raise OSError("temporary containment failure")
        with temporary.open("rb") as source, staged.open("xb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        if output_dir.resolve(strict=True) != staged.parent.resolve(strict=True):
            raise OSError("output staging containment failure")
        os.replace(staged, destination)
    except (OSError, wave.Error) as exc:
        staged.unlink(missing_ok=True)
        raise ProcessFailure("OUTPUT_WRITE_FAILED") from exc
    finally:
        temporary.unlink(missing_ok=True)
    try:
        _validate_wav(destination, workspace_root)
    except ProcessFailure:
        destination.unlink(missing_ok=True)
        raise
    return destination.resolve(strict=True)


def handle_request(
    payload: Mapping[str, Any],
    emitter: ProtocolEmitter,
    *,
    state_validator: StateValidator = validate_runtime_state,
    runtime_loader: RuntimeLoader = load_runtime,
) -> Path:
    emitter.progress(5, "Validating scalar request")
    request = validate_request(payload)
    emitter.progress(15, "Verifying local setup and model assets")
    try:
        runtime_state = state_validator(ROOT, request.models_dir)
    except ProcessFailure:
        raise
    except BaseException as exc:
        raise ProcessFailure("STATE_VALIDATION_FAILED") from exc
    emitter.progress(30, "Loading the local Qwen3-TTS runtime")
    emitter.log("Generating one offline CustomVoice waveform")
    np, samples = synthesize(request, runtime_state, runtime_loader=runtime_loader)
    emitter.progress(92, "Writing mono PCM16 WAV")
    output = write_wav(np, samples, request.workspace_dir, request.temp_dir)
    emitter.progress(100, "Speech generation complete")
    return output


Handler = Callable[[Mapping[str, Any], ProtocolEmitter], Path]


def _workspace_for_diagnostic(payload: Mapping[str, Any] | None) -> Path | None:
    if not isinstance(payload, Mapping):
        return None
    try:
        workspace_candidate = native_directory_path(
            payload.get("workspaceDir"),
            "workspaceDir",
            current_platform_name(),
            must_exist=True,
        )
        models = native_directory_path(
            payload.get("modelsDir"),
            "modelsDir",
            current_platform_name(),
            must_exist=True,
        )
        temporary_candidate = native_directory_path(
            payload.get("tempDir"),
            "tempDir",
            current_platform_name(),
            must_exist=True,
        )
        validate_runtime_storage_disjoint(models, workspace_candidate, temporary_candidate)
        return workspace_candidate.resolve(strict=True)
    except BaseException:
        return None


def _valid_result_path(output: Path, payload: Mapping[str, Any] | None) -> bool:
    workspace = _workspace_for_diagnostic(payload)
    if workspace is None or not output.is_absolute():
        return False
    try:
        info = output.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or bool(getattr(info, "st_file_attributes", 0) & WINDOWS_REPARSE_ATTRIBUTE)
            or not stat.S_ISREG(info.st_mode)
        ):
            return False
        resolved = output.resolve(strict=True)
        expected_parent = workspace.joinpath(*OUTPUT_RELATIVE_DIRECTORY).resolve(strict=True)
        return resolved.parent == expected_parent and workspace in resolved.parents
    except OSError:
        return False


def run_protocol(
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
    handler: Handler = handle_request,
) -> int:
    emitter = ProtocolEmitter(output_stream)
    payload: dict[str, Any] | None = None
    run_id = uuid.uuid4().hex
    try:
        payload = _read_one_payload(input_stream)
        output = handler(payload, emitter)
        if not _valid_result_path(output, payload):
            raise ProcessFailure("OUTPUT_RESULT_INVALID")
        emitter.done(output)
        return 0
    except BaseException as exc:
        failure = exc if isinstance(exc, ProcessFailure) else ProcessFailure("GENERATION_UNEXPECTED")
        diagnostic = write_diagnostic(
            _workspace_for_diagnostic(payload),
            run_id,
            code=failure.code,
            stage=failure.stage,
            action=failure.action,
        )
        public = failure.public_message(diagnostic)
        emitter.error(public)
        return 1
