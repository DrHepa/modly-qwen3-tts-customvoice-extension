from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from qwen3_tts_modly import assets
from qwen3_tts_modly.constants import (
    ASSETS as PINNED_ASSETS,
    EXTENSION_VERSION,
    AssetSpec,
)


class FakeResponse:
    def __init__(
        self,
        data: bytes,
        *,
        status: int = 200,
        content_range: str = "",
        include_length: bool = True,
    ) -> None:
        self._stream = io.BytesIO(data)
        self.status = status
        self.headers: dict[str, str] = {}
        if content_range:
            self.headers["Content-Range"] = content_range
        if include_length:
            self.headers["Content-Length"] = str(len(data))

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)


def spec_for(data: bytes, name: str = "asset.bin") -> AssetSpec:
    return AssetSpec(name, len(data), hashlib.sha256(data).hexdigest())


def test_pinned_snapshot_inventory_has_exact_paths_sizes_and_weight_hashes() -> None:
    inventory = {spec.relative_path: (spec.size, spec.sha256) for spec in PINNED_ASSETS}
    assert inventory == {
        ".gitattributes": (1_519, "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361"),
        "README.md": (57_846, "64f65e809b51cc0c35f393fbbfcc2d735c0cb3fdbbc6f3fdc4a5e6ce55e9d088"),
        "config.json": (4_908, "17a07f527a1c25ea30b4e023a184482a23d3e279d697b1dc81b1bde498d29cf9"),
        "generation_config.json": (245, "f1b90b4513f3b34c62851049e2492d7b4c5940daf1276f89c82b8ef04127f3aa"),
        "merges.txt": (1_671_839, "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"),
        "model.safetensors": (3_833_402_552, "38b1d5971bdbd982b561cccec982669a53b0537c3cf5e9bd4778ed07bb2f5137"),
        "preprocessor_config.json": (127, "efdde1022ea9d76928bf7a9cd53139138f5ba2e466e837f08f6105ab1af1c119"),
        "speech_tokenizer/config.json": (2_336, "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167"),
        "speech_tokenizer/configuration.json": (76, "6bc26d64eb5024b4d1dab5a52371958b429256d6c9d59787f1f5294a54e0cebd"),
        "speech_tokenizer/model.safetensors": (682_293_092, "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258"),
        "speech_tokenizer/preprocessor_config.json": (234, "fcb3805e597e786d4067706e602f6688524640f8d3396790e2e09b5942fcbdfb"),
        "tokenizer_config.json": (7_344, "dc3c31c3bdaedd5016382bb3cbe07323026775ad51f5a4fb564505992ae4a670"),
        "vocab.json": (2_776_833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
    }
    assert all(len(digest) == 64 for _size, digest in inventory.values())


def test_every_asset_url_uses_the_immutable_public_revision() -> None:
    revision = "0c0e3051f131929182e2c023b9537f8b1c68adfe"
    for spec in PINNED_ASSETS:
        assert spec.resolve_url.startswith("https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/resolve/")
        assert f"/resolve/{revision}/" in spec.resolve_url
        assert "/main/" not in spec.resolve_url
        assert "token=" not in spec.resolve_url.casefold()


def test_known_truncated_vocab_size_is_not_accepted(tmp_path: Path) -> None:
    spec = next(item for item in PINNED_ASSETS if item.relative_path == "vocab.json")
    (tmp_path / "vocab.json").write_bytes(b"x" * 913_408)
    valid, reason = assets.verify_asset(tmp_path / "vocab.json", spec)
    assert not valid
    assert "expected 2776833" in reason


def test_valid_asset_is_idempotent_without_network(tmp_path: Path) -> None:
    data = b"verified asset"
    spec = spec_for(data)
    (tmp_path / spec.relative_path).write_bytes(data)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("network must not be opened")

    result = assets.ensure_asset(tmp_path, spec, opener=forbidden, log=lambda _message: None)
    assert result.read_bytes() == data


def test_valid_asset_removes_stale_regular_partial_without_network(tmp_path: Path) -> None:
    data = b"verified asset"
    spec = spec_for(data)
    (tmp_path / spec.relative_path).write_bytes(data)
    part = tmp_path / f"{spec.relative_path}.part"
    part.write_bytes(b"stale")
    assets.ensure_asset(
        tmp_path,
        spec,
        opener=lambda *_args, **_kwargs: pytest.fail("network must not be opened"),
        log=lambda _message: None,
    )
    assert not part.exists()


def test_incomplete_partial_resumes_with_exact_range_and_promotes_atomically(tmp_path: Path) -> None:
    data = b'{"a":123}'
    spec = spec_for(data, "vocab.json")
    part = tmp_path / "vocab.json.part"
    part.write_bytes(data[:4])
    requests: list[object] = []

    def opener(request: object, *, timeout: float) -> FakeResponse:
        requests.append(request)
        assert request.headers["Range"] == "bytes=4-"
        assert request.get_header("User-agent") == (
            f"Modly-Qwen3-TTS-CustomVoice/{EXTENSION_VERSION}"
        )
        assert "Authorization" not in request.headers
        return FakeResponse(data[4:], status=206, content_range="bytes 4-8/9")

    result = assets.ensure_asset(
        tmp_path,
        spec,
        opener=opener,
        log=lambda _message: None,
        retry_delay=0,
    )
    assert result.read_bytes() == data
    assert not part.exists()
    assert len(requests) == 1


def test_equal_size_corrupt_partial_restarts_without_range(tmp_path: Path) -> None:
    data = b"correct data"
    spec = spec_for(data)
    (tmp_path / "asset.bin.part").write_bytes(b"x" * len(data))

    def opener(request: object, *, timeout: float) -> FakeResponse:
        assert "Range" not in request.headers
        return FakeResponse(data)

    result = assets.ensure_asset(tmp_path, spec, opener=opener, log=lambda _message: None)
    assert result.read_bytes() == data


def test_resume_rejects_wrong_content_range_and_preserves_useful_partial(tmp_path: Path) -> None:
    data = b"abcdefghij"
    spec = spec_for(data)
    part = tmp_path / "asset.bin.part"
    part.write_bytes(data[:3])
    with pytest.raises(assets.AssetError, match="ASSET_DOWNLOAD_FAILED"):
        assets.ensure_asset(
            tmp_path,
            spec,
            opener=lambda *_args, **_kwargs: FakeResponse(
                data[3:], status=206, content_range="bytes 2-9/10"
            ),
            log=lambda _message: None,
            retries=1,
        )
    assert part.read_bytes() == data[:3]


def test_http_416_restarts_once_from_zero(tmp_path: Path) -> None:
    data = b"abcdefghij"
    spec = spec_for(data)
    (tmp_path / "asset.bin.part").write_bytes(data[:3])
    requests: list[object] = []

    def opener(request: object, *, timeout: float) -> FakeResponse:
        requests.append(request)
        if len(requests) == 1:
            raise HTTPError(spec.resolve_url, 416, "range rejected", None, None)
        assert "Range" not in request.headers
        return FakeResponse(data)

    result = assets.ensure_asset(tmp_path, spec, opener=opener, log=lambda _message: None)
    assert result.read_bytes() == data
    assert len(requests) == 2


def test_snapshot_writes_ready_only_after_exact_json_and_hash_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = json.dumps({"model": "fixture"}).encode()
    spec = spec_for(data, "config.json")
    monkeypatch.setattr(assets, "ASSETS", (spec,))
    monkeypatch.setattr(assets, "JSON_ASSET_PATHS", frozenset({"config.json"}))

    opened = 0

    def opener(*_args: object, **_kwargs: object) -> FakeResponse:
        nonlocal opened
        opened += 1
        return FakeResponse(data)

    result = assets.ensure_snapshot(tmp_path, opener=opener, log=lambda _message: None)
    assert result == tmp_path.resolve()
    marker = tmp_path / assets.READY_MARKER_FILENAME
    assert marker.is_file()
    assert json.loads(marker.read_text(encoding="utf-8"))["extensionVersion"] == (
        EXTENSION_VERSION
    )
    assert assets.verify_snapshot(tmp_path) == []
    assert opened == 1

    assets.ensure_snapshot(
        tmp_path,
        opener=lambda *_args, **_kwargs: pytest.fail("ready snapshot must not use network"),
        log=lambda _message: None,
    )


def test_corrupt_download_never_writes_ready_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = b"correct"
    spec = spec_for(data)
    monkeypatch.setattr(assets, "ASSETS", (spec,))
    monkeypatch.setattr(assets, "JSON_ASSET_PATHS", frozenset())
    with pytest.raises(assets.AssetError):
        assets.ensure_snapshot(
            tmp_path,
            opener=lambda *_args, **_kwargs: FakeResponse(b"wrong!!"),
            log=lambda _message: None,
        )
    assert not (tmp_path / assets.READY_MARKER_FILENAME).exists()


def test_repair_removes_unexpected_regular_file_and_converges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = b"correct"
    spec = spec_for(data)
    monkeypatch.setattr(assets, "ASSETS", (spec,))
    monkeypatch.setattr(assets, "JSON_ASSET_PATHS", frozenset())
    (tmp_path / "unexpected.bin").write_bytes(b"unexpected")
    assets.ensure_snapshot(
        tmp_path,
        opener=lambda *_args, **_kwargs: FakeResponse(data),
        log=lambda _message: None,
    )
    assert not (tmp_path / "unexpected.bin").exists()
    assert assets.verify_snapshot(tmp_path) == []


def test_repair_removes_unexpected_empty_directory_and_converges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = b"correct"
    spec = spec_for(data)
    monkeypatch.setattr(assets, "ASSETS", (spec,))
    monkeypatch.setattr(assets, "JSON_ASSET_PATHS", frozenset())
    (tmp_path / "unexpected-directory").mkdir()
    assets.ensure_snapshot(
        tmp_path,
        opener=lambda *_args, **_kwargs: FakeResponse(data),
        log=lambda _message: None,
    )
    assert not (tmp_path / "unexpected-directory").exists()
    assert assets.verify_snapshot(tmp_path) == []


def test_repair_cleans_owned_residue_without_following_aliases_or_staging_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = b"verified"
    spec = spec_for(data, "nested/asset.bin")
    monkeypatch.setattr(assets, "ASSETS", (spec,))
    monkeypatch.setattr(assets, "JSON_ASSET_PATHS", frozenset())
    expected = tmp_path / "nested" / "asset.bin"
    expected.parent.mkdir()
    expected.write_bytes(data)

    outside_file = tmp_path.parent / f"{tmp_path.name}-outside-file"
    outside_dir = tmp_path.parent / f"{tmp_path.name}-outside-dir"
    outside_file.write_bytes(b"outside")
    outside_dir.mkdir()
    (outside_dir / "keep.txt").write_text("keep", encoding="utf-8")
    obsolete = tmp_path / "obsolete-layout"
    obsolete.mkdir()
    (obsolete / "stale.bin").write_bytes(b"stale")
    try:
        (obsolete / "outside-file-link").symlink_to(outside_file)
        (tmp_path / "outside-dir-link").symlink_to(outside_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"filesystem aliases unavailable: {exc}")
    (tmp_path / "obsolete.bin.part").write_bytes(b"stale")
    marker_temp = tmp_path / f".{assets.READY_MARKER_FILENAME}.interrupted.tmp"
    marker_temp.write_text("interrupted", encoding="utf-8")
    messages: list[str] = []

    assets.ensure_snapshot(
        tmp_path,
        opener=lambda *_args, **_kwargs: pytest.fail("valid expected file must be reused"),
        log=messages.append,
    )

    assert expected.read_bytes() == data
    assert outside_file.read_bytes() == b"outside"
    assert (outside_dir / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert assets.verify_snapshot(tmp_path) == []
    assert not obsolete.exists()
    assert not marker_temp.exists()
    assert not (tmp_path / "obsolete.bin.part").exists()
    assert not (tmp_path / "outside-dir-link").exists()
    assert not any("staging" in path.name.casefold() for path in tmp_path.parent.iterdir())
    assert all(str(tmp_path) not in message for message in messages)


def test_repair_replaces_expected_asset_alias_without_touching_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = b"correct"
    spec = spec_for(data)
    monkeypatch.setattr(assets, "ASSETS", (spec,))
    monkeypatch.setattr(assets, "JSON_ASSET_PATHS", frozenset())
    outside = tmp_path.parent / f"{tmp_path.name}-outside-target"
    outside.write_bytes(b"outside")
    try:
        (tmp_path / spec.relative_path).symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"file aliases unavailable: {exc}")

    assets.ensure_snapshot(
        tmp_path,
        opener=lambda *_args, **_kwargs: FakeResponse(data),
        log=lambda _message: None,
    )

    assert (tmp_path / spec.relative_path).read_bytes() == data
    assert not (tmp_path / spec.relative_path).is_symlink()
    assert outside.read_bytes() == b"outside"


def test_repair_rejects_aliased_owned_root_without_touching_target(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    alias = tmp_path / "owned"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    try:
        alias.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory aliases unavailable: {exc}")
    with pytest.raises(assets.AssetError, match="ASSET_OWNED_ROOT_INVALID"):
        assets.ensure_snapshot(
            alias,
            opener=lambda *_args, **_kwargs: pytest.fail("unsafe root must not use network"),
            log=lambda _message: None,
        )
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "keep"
