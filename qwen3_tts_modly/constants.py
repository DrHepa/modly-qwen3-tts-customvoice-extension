"""Immutable extension, dependency, and model-snapshot identities."""

from __future__ import annotations

from dataclasses import dataclass


EXTENSION_ID = "qwen3-tts-customvoice-process-extension"
EXTENSION_NAME = "Qwen3-TTS CustomVoice"
EXTENSION_VERSION = "0.1.1"
NODE_ID = "generate-speech"
ENTRY_FILENAME = "qwen3_tts_customvoice_process.py"

MODEL_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
MODEL_REVISION = "0c0e3051f131929182e2c023b9537f8b1c68adfe"
READY_MARKER_FILENAME = ".modly-qwen3-tts-assets-v1.json"
STATE_FILENAME = ".modly-qwen3-tts-state-v1.json"
STATE_SCHEMA = "modly.qwen3-tts-customvoice.setup-state.v1"
READY_SCHEMA = "modly.qwen3-tts-customvoice.assets.v1"

SAMPLE_RATE = 24_000
OUTPUT_RELATIVE_DIRECTORY = ("Workflows", "Qwen3-TTS-CustomVoice")
DIAGNOSTICS_DIRECTORY_NAME = "Diagnostics"

SPEAKERS = (
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
)
LANGUAGES = (
    "Auto",
    "Chinese",
    "English",
    "German",
    "Italian",
    "Portuguese",
    "Spanish",
    "Japanese",
    "Korean",
    "French",
    "Russian",
)
DEFAULT_SPEAKER = "Vivian"
DEFAULT_LANGUAGE = "Auto"
DEFAULT_INSTRUCT = ""
DEFAULT_NON_STREAMING_MODE = "true"
DEFAULT_DO_SAMPLE = "true"
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 1.0
DEFAULT_TEMPERATURE = 0.9
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_SUBTALKER_DOSAMPLE = "true"
DEFAULT_SUBTALKER_TOP_K = 50
DEFAULT_SUBTALKER_TOP_P = 1.0
DEFAULT_SUBTALKER_TEMPERATURE = 0.9
DEFAULT_MAX_NEW_TOKENS = 8192


@dataclass(frozen=True)
class AssetSpec:
    """One immutable file in the pinned Hugging Face snapshot."""

    relative_path: str
    size: int
    sha256: str

    @property
    def resolve_url(self) -> str:
        from urllib.parse import quote

        encoded_path = "/".join(quote(part, safe="") for part in self.relative_path.split("/"))
        return (
            f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/"
            f"{encoded_path}?download=true"
        )


ASSETS = (
    AssetSpec(
        ".gitattributes",
        1_519,
        "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
    ),
    AssetSpec(
        "README.md",
        57_846,
        "64f65e809b51cc0c35f393fbbfcc2d735c0cb3fdbbc6f3fdc4a5e6ce55e9d088",
    ),
    AssetSpec(
        "config.json",
        4_908,
        "17a07f527a1c25ea30b4e023a184482a23d3e279d697b1dc81b1bde498d29cf9",
    ),
    AssetSpec(
        "generation_config.json",
        245,
        "f1b90b4513f3b34c62851049e2492d7b4c5940daf1276f89c82b8ef04127f3aa",
    ),
    AssetSpec(
        "merges.txt",
        1_671_839,
        "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    ),
    AssetSpec(
        "model.safetensors",
        3_833_402_552,
        "38b1d5971bdbd982b561cccec982669a53b0537c3cf5e9bd4778ed07bb2f5137",
    ),
    AssetSpec(
        "preprocessor_config.json",
        127,
        "efdde1022ea9d76928bf7a9cd53139138f5ba2e466e837f08f6105ab1af1c119",
    ),
    AssetSpec(
        "speech_tokenizer/config.json",
        2_336,
        "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167",
    ),
    AssetSpec(
        "speech_tokenizer/configuration.json",
        76,
        "6bc26d64eb5024b4d1dab5a52371958b429256d6c9d59787f1f5294a54e0cebd",
    ),
    AssetSpec(
        "speech_tokenizer/model.safetensors",
        682_293_092,
        "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258",
    ),
    AssetSpec(
        "speech_tokenizer/preprocessor_config.json",
        234,
        "fcb3805e597e786d4067706e602f6688524640f8d3396790e2e09b5942fcbdfb",
    ),
    AssetSpec(
        "tokenizer_config.json",
        7_344,
        "dc3c31c3bdaedd5016382bb3cbe07323026775ad51f5a4fb564505992ae4a670",
    ),
    AssetSpec(
        "vocab.json",
        2_776_833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
)

JSON_ASSET_PATHS = frozenset(
    spec.relative_path for spec in ASSETS if spec.relative_path.endswith(".json")
)

BUILD_DISTRIBUTIONS = {
    "pip": "26.1.2",
    "setuptools": "78.1.0",
    "wheel": "0.47.0",
}

RUNTIME_DISTRIBUTIONS = {
    "qwen-tts": "0.1.1",
    "transformers": "4.57.3",
    "accelerate": "1.12.0",
    "gradio": "6.16.0",
    "librosa": "0.11.0",
    "soundfile": "0.14.0",
    "sox": "1.5.0",
    "soxr": "1.0.0",
    "onnxruntime": "1.27.0",
    "einops": "0.8.2",
    "numpy": "2.4.6",
    "scipy": "1.17.0",
    "scikit-learn": "1.9.0",
    "numba": "0.66.0",
    "llvmlite": "0.48.0",
}

TORCH_BASE_VERSION = "2.11.0"
TORCH_VARIANTS = {
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

ALL_DISTRIBUTIONS = {**BUILD_DISTRIBUTIONS, **RUNTIME_DISTRIBUTIONS}
