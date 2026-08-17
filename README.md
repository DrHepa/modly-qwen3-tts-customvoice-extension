# Qwen3-TTS CustomVoice for Modly

An open-source Python PROCESS extension by **DrHepa** that maps one Modly text
input to one mono PCM16 WAV file at 24 kHz using the built-in speakers from
`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`.

## Status

Version `0.1.2` has static, unit, mocked protocol, and limited real-runtime
validation. In one clean Linux runtime, setup completed, the immutable
4,520,218,951-byte model inventory was fully size/hash verified, and one real
scalar PROCESS request loaded the pinned snapshot strictly offline and produced
a structurally valid mono PCM16 WAV at 24 kHz.

Production Install-from-GitHub evidence also showed that a multi-gigabyte
PyTorch installation can remain active beyond the former shared one-hour wall
limit. Version `0.1.2` gives each dependency-install command the separate
three-hour bound documented below without changing dependency or model pins.

That evidence applies only to the tested Linux runtime configuration. It is not
a perceptual or transcription-quality evaluation, and it does not establish
Windows, Linux x64, Linux ARM64, CPU, CUDA 12.8, or CUDA 13.0 support as a whole.
Clean Install-from-GitHub UI validation and packaged Windows/Linux E2E remain
pending. The source-only GitHub Actions matrix has completed successfully on
Ubuntu and Windows with Python 3.11 and 3.12; that is not packaged native or
runtime E2E evidence.

## Requirements and compatibility

This extension requires a Modly build whose Python PROCESS contract supplies:

- absolute `models_dir` in the one-object setup JSON; and
- absolute `modelsDir` in each runtime request.

JSON setup is the only successful setup contract. The historical positional
forms `<python_exe> <ext_dir> <gpu_sm>` and
`<python_exe> <ext_dir> <gpu_sm> <cuda_version>` have no model-storage field;
they are parsed only to report the stable `SETUP_MODELS_DIR_REQUIRED` upgrade
error instead of guessing a directory.

The extension uses only those explicit values. It does not inspect Modly
settings, user profiles, environment variables, sibling directories, or a
global Hugging Face cache to discover model storage.

### Candidate platform matrix

| Platform | CPU | NVIDIA CUDA 12.8 | NVIDIA CUDA 13.0 | Status |
| --- | --- | --- | --- | --- |
| Windows x64 | `torch==2.11.0+cpu` | `torch==2.11.0+cu128` | `torch==2.11.0+cu130` | Candidate; not E2E verified |
| Linux x64 | `torch==2.11.0+cpu` | `torch==2.11.0+cu128` | `torch==2.11.0+cu130` | Candidate; not E2E verified |
| Linux ARM64 | Same selected flavors where every official pinned wheel exists | Same | Same | Wheel-availability candidate only |

Python 3.11 and 3.12 are accepted. Other operating systems, Windows ARM64,
other architectures, non-CPU/non-CUDA accelerators, and CUDA hints other than
12.8 or 13.0 are rejected before dependency installation. CUDA compute
capability is probed rather than equality-gated. Runtime uses float32 on CPU;
on CUDA it uses BF16 only after a synchronized capability/runtime probe and
otherwise falls back to float32. SDPA is the baseline attention implementation;
FlashAttention is neither installed nor required.

Candidate compatibility means the setup logic and command construction exist.
It does not guarantee that every pinned third-party wheel is published for
every Python/platform combination.

## Installation and Repair

When this repository is published and the matching Modly host contract is
available, use Modly's **Install from GitHub** flow:

1. Install `https://github.com/DrHepa/modly-qwen3-tts-customvoice-extension`
   through Modly's extension installer.
   The checkout or install-directory basename is not the extension identity
   and may differ from `qwen3-tts-customvoice-process-extension`. Setup binds
   `ext_dir` canonically to the running extension root and verifies the regular,
   non-aliased root `manifest.json`, including its exact ID and PROCESS entry
   contract, before making changes.
2. Setup creates or repairs exactly `<extension>/venv` using Modly's Python.
   A venv is reused in place only when Python minor, implementation, cache tag,
   SOABI, operating system, and architecture match, and the interpreter reports
   that venv as its active prefix. Otherwise setup verifies a fixed sibling
   staging venv, activates it with rollback-safe renames, and removes the
   temporary backup. Normal POSIX `venv/bin/python` interpreter symlinks are
   supported after this prefix and ABI verification.
3. It installs exact selected direct and native package versions, rejects
   unexpected source builds, runs `pip check` plus audio/tensor health checks,
   and downloads the immutable public model snapshot.
4. If setup is interrupted, run **Repair**. Valid files are reused and
   same-directory `.part` files resume only after HTTP Range validation.
   Repair reconciles obsolete files, directories, aliases, and interrupted
   readiness temporaries only inside the exact extension-owned model directory;
   it never follows aliases or duplicates the full snapshot in a sibling.

Setup never invokes `apt`, `sudo`, a system package manager, Hugging Face CLI,
or `from_pretrained` to obtain weights. It does not read or log a Hugging Face
token because the pinned repository is public.

The Python `sox==1.5.0` wrapper is pinned. An external `sox` executable is a
real upstream dependency for some SoX-based utilities, so setup reports its
absence as a warning. The scalar `generate_custom_voice` path implemented here
does not directly invoke that executable; it is therefore not a setup hard
gate. The Python SoX wrapper is the one explicit source-distribution exception;
every other setup install command requires binary distributions.

The selected direct pins include `setuptools==78.1.0` and `gradio==6.16.0`.
They preserve the verified metadata intersection required by PyTorch 2.11.0,
Transformers 4.57.3, and Gradio without claiming a complete transitive lock.
Setup still runs normal `pip check`. The sole compatibility exception is the
official `nvidia-cusparselt-cu13==0.8.0` Linux ARM64 CUDA 13.0 wheel whose
payload is ARM64 but whose internal tag is exactly
`py3-none-manylinux2014_sbsa`. It is accepted only when that is the single
canonical `pip check` diagnostic and the installed distribution, inventory,
and WHEEL metadata pass local regular-file, containment, version, and exact-tag
validation. CUDA tensor/audio health checks remain mandatory afterward.

## Usage

Connect one Modly text artifact to **Generate Speech**, select the desired
speaker, language, and generation controls, then run the node. A successful
run returns one audio artifact backed by the validated WAV described below.

## Model storage

Setup owns this directory under the explicit Modly models root:

```text
<modelsDir>/qwen3-tts-customvoice-process-extension/generate-speech
```

The exact snapshot is 4,520,218,951 bytes (about 4.52 GB decimal), excluding
the Python environment and filesystem overhead. Every declared file has a
pinned size and SHA-256 digest. Readiness is marked atomically only after the
complete inventory verifies.

Setup state stores public version, platform-flavor, and inventory records only;
it does not persist configured absolute paths, request data, or credentials.

Storage roots must be canonically disjoint. Before setup creates, removes, or
repairs anything, it rejects a `models_dir` or owned snapshot that is the same
as, above, or below the extension's venv, venv recovery directories, state, or
other extension-managed mutable paths. Runtime applies the same bidirectional
rule between `modelsDir`/the owned snapshot and `workspaceDir`, `Workflows`,
`tempDir`, output, diagnostics, and setup-state paths. Existing symlink or
junction parents and not-yet-created descendants are checked without creating
them. Ordinary disjoint sibling roots remain supported.

With the intended separate Modly models root, the snapshot is outside the
extension source directory. Modly uninstall may therefore leave it in place.
Removal is extension-managed: after the extension is no longer running, remove
the owned directory shown above if the storage is no longer wanted. Never
remove the shared `<modelsDir>` root.

## Node: Generate Speech

- **Input port:** `text` (one non-empty scalar string)
- **Output port:** `audio` (one final WAV artifact)
- **Built-in speakers:** `Vivian`, `Serena`, `Uncle_Fu`, `Dylan`, `Eric`,
  `Ryan`, `Aiden`, `Ono_Anna`, `Sohee`
- **Languages:** `Auto`, `Chinese`, `English`, `German`, `Italian`,
  `Portuguese`, `Spanish`, `Japanese`, `Korean`, `French`, `Russian`

Text is an input port and is not duplicated as a parameter. Lists and batch
requests are rejected; the PROCESS does not concatenate, loop over, or package
multiple results.

### Parameters

| Parameter | Type | Default | Validation / meaning |
| --- | --- | --- | --- |
| `speaker` | select | `Vivian` | Exact built-in speaker value |
| `language` | select | `Auto` | Exact supported language value |
| `instruct` | string | empty | Optional CustomVoice style instruction |
| `non_streaming_mode` | select string | `true` | Strict `true` / `false` |
| `do_sample` | select string | `true` | Strict `true` / `false` |
| `top_k` | integer | `50` | At least `0` |
| `top_p` | float | `1.0` | Greater than `0`, at most `1` |
| `temperature` | float | `0.9` | Greater than `0` |
| `repetition_penalty` | float | `1.05` | Greater than `0` |
| `subtalker_dosample` | select string | `true` | Exact upstream spelling; strict `true` / `false` |
| `subtalker_top_k` | integer | `50` | At least `0` |
| `subtalker_top_p` | float | `1.0` | Greater than `0`, at most `1` |
| `subtalker_temperature` | float | `0.9` | Greater than `0` |
| `max_new_tokens` | integer | `8192` | At least `2` (the upstream core forces this bound) |

These normalized values are forwarded to the official
`generate_custom_voice` API. An empty `instruct` is omitted. Setting
`non_streaming_mode=false` changes upstream text conditioning, but the Modly
PROCESS contract still returns only one final WAV; it does not stream audio
artifacts.

CustomVoice has no seed argument, so this extension does not invent one.
VoiceDesign and Base-model voice cloning (`ref_audio`/reference text) are
separate upstream capabilities and are intentionally out of scope for this
CustomVoice node. Beijing and Sichuan dialect labels are not part of this
model's supported language list and are not exposed.

## Outputs

Runtime requires the explicit `modelsDir`, re-derives the owned snapshot,
validates pathless setup state and every asset before importing Qwen, and sets
Hugging Face/Transformers offline flags before those libraries can import. It
loads only the absolute local snapshot with `local_files_only=True` and never
falls back to a remote service.

Successful runs atomically publish a unique file beneath:

```text
<workspaceDir>/Workflows/Qwen3-TTS-CustomVoice/
```

The file is validated as non-empty mono PCM16 audio at 24 kHz. The terminal
result is exactly `{ "filePath": "<absolute-existing-path>" }`.

## Troubleshooting

The declared entry duplicates the Modly NDJSON descriptor and redirects OS file
descriptors 1 and 2 before importing the internal runtime or third-party code.
Intentional progress/log/terminal messages use only the dedicated channel;
Python-level and native stdout/stderr writes are discarded so they cannot
corrupt the protocol or expose request data. Once the Python bootstrap starts,
it emits exactly one terminal `done` or `error` event for handled requests and
import/runtime failures. Interpreter startup and syntax failures occur before
Python can establish this boundary and therefore cannot be converted into a
terminal NDJSON error.

Errors use stable public codes and actions. If a valid workspace is available,
a bounded diagnostic is written under
`Workflows/Qwen3-TTS-CustomVoice/Diagnostics/` and referenced relatively.
Diagnostics contain only run ID, stable code, stage, and public action. They do
not contain input text, instructions, arbitrary absolute paths, exceptions,
tracebacks, environment values, commands, settings, tokens, or secrets.

Run **Repair** after dependency changes, interrupted setup, or model-file
modification. Correct `REQUEST_*` values in the node. For `OUTPUT_*`, verify
workspace storage and permissions.

`SETUP_STORAGE_OVERLAP` means the configured models root intersects
extension-managed mutable storage. `REQUEST_STORAGE_OVERLAP` means model
storage intersects runtime workspace or temporary storage. Select separate
roots and retry; the public error intentionally does not echo configured
absolute paths.

Dependency setup diagnostics are bounded and classified without exposing raw
package-manager output. `SETUP_DEPENDENCY_CONFLICT` indicates incompatible
metadata, `SETUP_WHEEL_UNAVAILABLE` indicates a missing binary wheel,
`SETUP_NETWORK_FAILED` indicates download connectivity failure, and
`SETUP_STORAGE_FULL` indicates insufficient storage. `SETUP_COMMAND_FAILED` is
the privacy-safe fallback. Correct the reported condition and run **Repair**;
the extension intentionally omits commands, indexes, URLs, local paths,
credentials, and raw resolver output from public logs.

Each multi-gigabyte dependency-install command has a separate three-hour wall
limit; environment preparation, dependency checks, metadata audits, and runtime
health probes retain their shorter bounded limits. `SETUP_COMMAND_TIMEOUT`
means one of those bounds expired. Verify network and system responsiveness,
then run **Repair**; setup does not continue from an unverified environment.

Runtime health failures also use a bounded, descriptor-isolated JSON channel.
`SETUP_HEALTH_CUDA_OOM` means a CUDA allocation failed under current GPU-memory
pressure: release GPU memory and run **Repair** again; setup does not silently
retry or weaken the probe. `SETUP_HEALTH_IMPORT_FAILED`,
`SETUP_HEALTH_CUDA_UNAVAILABLE`, `SETUP_HEALTH_NATIVE_PROBE_FAILED`, and
`SETUP_HEALTH_INVALID_RESULT` distinguish pinned imports, accelerator
availability, tensor/audio probes, and identity/result validation. Messages
include only an allowlisted health substage. Malformed or oversized health
output becomes `SETUP_HEALTH_FAILED`; raw exceptions, tracebacks, device names,
paths, environment values, and third-party output are never promoted. Missing
external SoX remains a warning after otherwise successful health checks.

## Limitations

- The successful Linux runtime validation covers one configuration only. Every
  listed platform/flavor combination remains a candidate until it is packaged
  and validated independently end to end.
- Clean Install-from-GitHub UI validation and remote GitHub Actions evidence
  remain pending.
- The generated WAV passed structural audio checks; perceptual quality and
  transcription accuracy were not evaluated.
- Requests are scalar. Batch input and streamed audio artifacts are not part
  of the current PROCESS contract.
- VoiceDesign, Base-model cloning, and reference-audio conditioning are outside
  this CustomVoice node.
- Peak memory and complete installed-environment size have not yet been
  measured on the candidate platforms.
- External SoX remains an upstream boundary for utilities outside the scalar
  generation path used here.
- The complete transitive dependency closure still needs clean validation on
  every candidate Python/platform combination; this repository intentionally
  does not invent a lock from one development machine.

## Development validation

The repository tests use mocks and local fixtures only; they do not install
Qwen/PyTorch, access the network, download weights, or run inference:

```bash
python3 -m pytest -q
python3 -m compileall -q setup.py qwen3_tts_customvoice_process.py qwen3_tts_modly tests
```

The strict Modly extension validator must be run with its audited audio I/O
allowance. `.github/workflows/ci.yml` defines the same source-only pytest and
compile checks for Ubuntu and Windows with Python 3.11 and 3.12. The workflow
has been added for publication but has not yet produced a remote CI result.

Separately from those static tiers, one clean Linux runtime setup, immutable
asset verification, offline local model load, scalar PROCESS generation, and
structural WAV validation have passed. That single execution is functional
evidence for its tested runtime only, not packaged cross-platform E2E evidence.

## Credits

- Model: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- Immutable revision: `0c0e3051f131929182e2c023b9537f8b1c68adfe`
- Modly integration author: DrHepa
- Upstream software and model: Qwen team

Modly is created by Lightning Pixel and is the host application for this
extension. Modly is not redistributed by this repository.

## License

DrHepa's integration code is MIT-licensed; see `LICENSE`. Qwen3-TTS software
and downloaded model files retain their Apache License 2.0 terms and upstream
attribution; see `NOTICE`, `THIRD_PARTY_NOTICES.md`,
`LICENSES/Apache-2.0.txt`, and the upstream model card. Dependency packages
retain their respective licenses. Modly is the host application and is not
redistributed by this repository.
