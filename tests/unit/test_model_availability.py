"""Unit tests for lib.model_availability (pure helpers, no network, no real disk).

Tests cover:
  - Ollama model-name parsing, canonical names and exact matching
  - Registry manifest sizing: deduplicated digests, blobs already in the
    store, ``-partial`` downloads never counted as present (nor touched)
  - fetch_pull_estimate outcomes (ok, 404, auth, HTTP errors, network
    errors, bad JSON, manifest lists) via a mock session
  - Model store resolution: $OLLAMA_MODELS, ~/.ollama/models, the Linux
    systemd service store
  - Disk checks, safety margin, the mid-pull disk guard, formatting and
    local/remote URL classification
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

import lib.model_availability as ma
from lib.model_availability import (
    DEFAULT_MIN_FREE_BYTES,
    GB,
    MANIFEST_ACCEPT,
    MANIFEST_LIST_TYPE,
    MAX_VARIANT_FETCHES,
    GiB,
    ModelRef,
    PullEstimate,
    blob_is_present,
    check_disk,
    existing_ancestor,
    fetch_pull_estimate,
    format_gb,
    format_gib,
    is_local_url,
    make_disk_guard,
    manifest_blob_sizes,
    normalize_model_name,
    ollama_models_dir,
    parse_model_ref,
    required_margin_bytes,
    sum_manifest_sizes,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "manifests"
QWEN3_VL_TOTAL = 20_910_286_445
QWEN3_VL_MODEL_DIGEST = "sha256:6e416d39200aae1cec3ea197c5a5ebbaf214ccddc9561bcc0ec7157c83b2a99b"
QWEN3_VL_MODEL_SIZE = 20_910_274_496


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Fail loudly if a test forgets to pass a mock session."""

    def no_http(*_args, **_kwargs):
        raise AssertionError("tests must pass a mock session")

    monkeypatch.setattr("lib.model_availability.requests.get", no_http)


@pytest.fixture
def qwen3_vl_manifest() -> dict:
    """The real registry manifest of qwen3-vl:32b (duplicate license layer)."""
    return json.loads((FIXTURES / "qwen3-vl_32b.json").read_text())


def _resp(status=200, payload=None, json_error=None):
    resp = MagicMock()
    resp.status_code = status
    if json_error is not None:
        resp.json.side_effect = json_error
    else:
        resp.json.return_value = payload if payload is not None else {}
    return resp


def _session(*responses):
    session = MagicMock()
    session.get.side_effect = list(responses)
    return session


def _usage(free):
    return lambda _path: SimpleNamespace(total=10 * free, used=9 * free, free=free)


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


class TestParseModelRef:
    def test_library_default_tag(self):
        ref = parse_model_ref("gemma4")
        assert ref == ModelRef("registry.ollama.ai", "library", "gemma4", "latest")
        assert ref.canonical == "gemma4:latest"
        assert ref.manifest_url == "https://registry.ollama.ai/v2/library/gemma4/manifests/latest"

    def test_library_with_tag(self):
        ref = parse_model_ref("qwen3.6:35b-a3b")
        assert (ref.repo, ref.tag) == ("qwen3.6", "35b-a3b")
        assert ref.canonical == "qwen3.6:35b-a3b"

    def test_namespaced(self):
        ref = parse_model_ref("Drews54/llama3.2-vision-abliterated")
        assert ref.namespace == "Drews54"
        assert ref.canonical == "Drews54/llama3.2-vision-abliterated:latest"

    def test_hf(self):
        ref = parse_model_ref("hf.co/mradermacher/dolphin-2.9.2-Phi-3-Medium-abliterated-i1-GGUF:Q6_K")
        assert ref.host == "hf.co"
        assert ref.tag == "Q6_K"
        assert ref.manifest_url == (
            "https://hf.co/v2/mradermacher/dolphin-2.9.2-Phi-3-Medium-abliterated-i1-GGUF"
            "/manifests/Q6_K"
        )
        assert ref.canonical.startswith("hf.co/mradermacher/")

    def test_host_port_is_preserved(self):
        ref = parse_model_ref("localhost:5000/ns/m:tag")
        assert ref.host == "localhost:5000"
        assert (ref.namespace, ref.repo, ref.tag) == ("ns", "m", "tag")
        assert ref.manifest_url == "https://localhost:5000/v2/ns/m/manifests/tag"

    def test_ollama_com_alias_uses_registry_host(self):
        ref = parse_model_ref("ollama.com/library/gemma4:26b")
        assert ref.canonical == "gemma4:26b"
        assert ref.manifest_url.startswith("https://registry.ollama.ai/v2/library/gemma4/")

    def test_manifest_url_for_digest(self):
        ref = parse_model_ref("gemma4:26b")
        assert ref.manifest_url_for("sha256:abc") == (
            "https://registry.ollama.ai/v2/library/gemma4/manifests/sha256:abc"
        )

    @pytest.mark.parametrize("name", ["", "   ", "a/b/c/d", "x@sha256:abc", "ns//m", "m:"])
    def test_rejects(self, name):
        with pytest.raises(ValueError, match="Unrecognised model name"):
            parse_model_ref(name)


class TestNormalizeModelName:
    def test_equivalent_spellings(self):
        names = ["registry.ollama.ai/library/qwen3-vl:32b", "QWEN3-VL:32b", "qwen3-vl:32b"]
        assert len({normalize_model_name(n) for n in names}) == 1

    def test_implicit_latest(self):
        assert normalize_model_name("Drews54/llama3.2-vision-abliterated") == normalize_model_name(
            "Drews54/llama3.2-vision-abliterated:latest"
        )

    def test_prefix_is_not_a_match(self):
        assert normalize_model_name("qwen3") != normalize_model_name("qwen3-vl:32b")
        assert normalize_model_name("qwen3") != normalize_model_name("qwen3-coder:latest")

    def test_unparseable_falls_back(self):
        assert normalize_model_name("  A/B/C/D ") == "a/b/c/d"


# ---------------------------------------------------------------------------
# Manifest sizing
# ---------------------------------------------------------------------------


class TestManifestSizes:
    def test_dedupes_duplicate_layers(self, qwen3_vl_manifest):
        sizes = manifest_blob_sizes(qwen3_vl_manifest)
        assert len(sizes) == 4  # config + model + license (once) + params
        assert sum(sizes.values()) == QWEN3_VL_TOTAL

    def test_ignores_malformed_entries(self):
        manifest = {
            "config": None,
            "layers": [{"digest": "sha256:a", "size": 10}, {"digest": "sha256:b"}, "junk",
                       {"digest": "sha256:c", "size": True}],
        }
        assert manifest_blob_sizes(manifest) == {"sha256:a": 10}

    def test_nothing_present_without_blobs_dir(self, qwen3_vl_manifest):
        assert sum_manifest_sizes(qwen3_vl_manifest, None) == (QWEN3_VL_TOTAL, QWEN3_VL_TOTAL)

    def test_subtracts_present_blobs(self, qwen3_vl_manifest, tmp_path):
        blobs = tmp_path / "blobs"
        blobs.mkdir()
        (blobs / QWEN3_VL_MODEL_DIGEST.replace(":", "-")).write_bytes(b"x")
        total, missing = sum_manifest_sizes(qwen3_vl_manifest, blobs)
        assert total == QWEN3_VL_TOTAL
        assert missing == QWEN3_VL_TOTAL - QWEN3_VL_MODEL_SIZE

    def test_partial_blob_is_not_present_and_not_touched(self, qwen3_vl_manifest, tmp_path):
        blobs = tmp_path / "blobs"
        blobs.mkdir()
        stem = QWEN3_VL_MODEL_DIGEST.replace(":", "-")
        partials = [blobs / f"{stem}-partial", blobs / f"{stem}-partial-0"]
        for p in partials:
            p.write_bytes(b"half")
        assert blob_is_present(blobs, QWEN3_VL_MODEL_DIGEST) is False
        assert sum_manifest_sizes(qwen3_vl_manifest, blobs) == (QWEN3_VL_TOTAL, QWEN3_VL_TOTAL)
        assert all(p.read_bytes() == b"half" for p in partials)  # never deleted or modified

    def test_directory_named_like_blob_is_not_present(self, tmp_path):
        (tmp_path / "sha256-abc").mkdir()
        assert blob_is_present(tmp_path, "sha256:abc") is False


class TestFetchPullEstimate:
    def test_ok_sends_accept_and_timeout(self, qwen3_vl_manifest):
        session = _session(_resp(200, qwen3_vl_manifest))
        est = fetch_pull_estimate("qwen3-vl:32b", session=session)
        assert est.known
        assert est.total_bytes == est.missing_bytes == QWEN3_VL_TOTAL
        assert est.source == "https://registry.ollama.ai/v2/library/qwen3-vl/manifests/32b"
        call = session.get.call_args
        assert call.args[0] == est.source
        assert call.kwargs["headers"]["Accept"] == MANIFEST_ACCEPT
        assert call.kwargs["timeout"] == 10.0

    def test_subtracts_local_blobs(self, qwen3_vl_manifest, tmp_path):
        (tmp_path / QWEN3_VL_MODEL_DIGEST.replace(":", "-")).write_bytes(b"x")
        est = fetch_pull_estimate(
            "qwen3-vl:32b", blobs_dir=tmp_path, session=_session(_resp(200, qwen3_vl_manifest))
        )
        assert est.missing_bytes == QWEN3_VL_TOTAL - QWEN3_VL_MODEL_SIZE

    def test_404_is_not_found(self):
        est = fetch_pull_estimate("qwen2-vl:7b", session=_session(_resp(404)))
        assert est.not_found is True
        assert not est.known
        assert "HTTP 404" in est.error

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_required_is_unknown(self, status):
        est = fetch_pull_estimate("hf.co/private/x:Q4", session=_session(_resp(status)))
        assert not est.known and not est.not_found
        assert "authentication" in est.error

    def test_server_error_is_unknown(self):
        est = fetch_pull_estimate("gemma4", session=_session(_resp(503)))
        assert not est.known
        assert est.error == "registry returned HTTP 503"

    def test_connection_error_is_unknown(self):
        session = MagicMock()
        session.get.side_effect = requests.ConnectionError("offline")
        est = fetch_pull_estimate("gemma4", session=session)
        assert not est.known and not est.not_found
        assert "offline" in est.error

    def test_bad_json_is_unknown(self):
        est = fetch_pull_estimate("gemma4", session=_session(_resp(200, json_error=ValueError("bad"))))
        assert not est.known
        assert "invalid manifest JSON" in est.error

    def test_manifest_without_layers_is_unknown(self):
        est = fetch_pull_estimate("gemma4", session=_session(_resp(200, {"layers": []})))
        assert not est.known

    def test_invalid_name_is_unknown_without_request(self):
        session = MagicMock()
        est = fetch_pull_estimate("a/b/c/d", session=session)
        assert not est.known
        session.get.assert_not_called()

    @staticmethod
    def _list_manifest():
        return {
            "mediaType": MANIFEST_LIST_TYPE,
            "manifests": [
                {"digest": "sha256:gguf", "runner": "llamacpp", "format": "gguf"},
                {"digest": "sha256:mlx", "runner": "mlx", "format": "safetensors"},
            ],
        }

    @staticmethod
    def _child(size):
        return {"config": {"digest": f"sha256:c{size}", "size": 1},
                "layers": [{"digest": f"sha256:l{size}", "size": size}]}

    def test_manifest_list_prefers_mlx_on_apple_silicon(self):
        session = _session(_resp(200, self._list_manifest()), _resp(200, self._child(500)))
        est = fetch_pull_estimate("qwen3.8:27b", session=session, system="Darwin", machine="arm64")
        assert session.get.call_args_list[1].args[0].endswith("/manifests/sha256:mlx")
        assert est.missing_bytes == 501

    def test_manifest_list_prefers_gguf_on_linux(self):
        session = _session(_resp(200, self._list_manifest()), _resp(200, self._child(700)))
        est = fetch_pull_estimate("qwen3.8:27b", session=session, system="Linux", machine="x86_64")
        assert session.get.call_args_list[1].args[0].endswith("/manifests/sha256:gguf")
        assert est.missing_bytes == 701

    def test_manifest_list_no_match_takes_max(self):
        listing = {"manifests": [{"digest": "sha256:a"}, {"digest": "sha256:b"}]}
        session = _session(_resp(200, listing), _resp(200, self._child(100)),
                           _resp(200, self._child(900)))
        est = fetch_pull_estimate("m", session=session, system="Linux", machine="x86_64")
        assert session.get.call_count == 3
        assert est.missing_bytes == 901

    def test_manifest_list_unknown_platform_takes_max(self):
        session = _session(_resp(200, self._list_manifest()), _resp(200, self._child(100)),
                           _resp(200, self._child(900)))
        est = fetch_pull_estimate("m", session=session, system=None, machine=None)
        assert est.missing_bytes == 901

    def test_manifest_list_no_match_compares_every_child(self):
        # The largest child sits after the first three: it must still be seen.
        listing = {"manifests": [{"digest": f"sha256:{i}"} for i in range(5)]}
        sizes = [100, 200, 300, 900, 50]
        session = _session(_resp(200, listing), *(_resp(200, self._child(s)) for s in sizes))
        est = fetch_pull_estimate("m", session=session, system=None, machine=None)
        assert session.get.call_count == 6
        assert est.missing_bytes == 901

    def test_manifest_list_too_many_unmatched_variants_is_unknown(self):
        listing = {"manifests": [{"digest": f"sha256:{i}"} for i in range(MAX_VARIANT_FETCHES + 1)]}
        session = _session(_resp(200, listing))
        est = fetch_pull_estimate("m", session=session, system=None, machine=None)
        assert not est.known
        assert "size unknown" in est.error
        assert session.get.call_count == 1

    def test_manifest_list_child_failure_is_unknown(self):
        session = _session(_resp(200, self._list_manifest()), _resp(500))
        est = fetch_pull_estimate("m", session=session, system="Linux", machine="x86_64")
        assert not est.known
        assert "sha256:gguf" in est.error

    def test_pull_estimate_known_property(self):
        assert PullEstimate("m", 1, 1).known is True
        assert PullEstimate("m", error="x").known is False


# ---------------------------------------------------------------------------
# Model store and disk
# ---------------------------------------------------------------------------


class TestOllamaModelsDir:
    def test_env_is_expanded(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        assert ollama_models_dir({"OLLAMA_MODELS": "~/x"}) == tmp_path / "x"

    def test_blank_env_is_ignored(self, tmp_path):
        assert ollama_models_dir({"OLLAMA_MODELS": "  "}, system="Darwin", home=tmp_path) == (
            tmp_path / ".ollama" / "models"
        )

    def test_default_uses_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("OLLAMA_MODELS", raising=False)
        assert ollama_models_dir(system="Darwin") == tmp_path / ".ollama" / "models"

    def test_linux_service_store_when_user_store_missing(self, monkeypatch, tmp_path):
        service = tmp_path / "usr-share-ollama" / ".ollama" / "models"
        service.mkdir(parents=True)
        monkeypatch.setattr(ma, "LINUX_SERVICE_MODELS_DIR", service)
        home = tmp_path / "home"
        assert ollama_models_dir({}, system="Linux", home=home) == service

    def test_linux_user_store_wins_when_present(self, monkeypatch, tmp_path):
        service = tmp_path / "svc"
        service.mkdir()
        monkeypatch.setattr(ma, "LINUX_SERVICE_MODELS_DIR", service)
        user_store = tmp_path / "home" / ".ollama" / "models"
        user_store.mkdir(parents=True)
        assert ollama_models_dir({}, system="Linux", home=tmp_path / "home") == user_store

    def test_linux_without_any_store_uses_home(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ma, "LINUX_SERVICE_MODELS_DIR", tmp_path / "missing")
        assert ollama_models_dir({}, system="Linux", home=tmp_path) == tmp_path / ".ollama" / "models"

    @pytest.mark.skipif(
        not hasattr(os, "geteuid") or os.geteuid() == 0,
        reason="root bypasses directory permissions",
    )
    def test_linux_private_service_home_counts_as_present(self, monkeypatch, tmp_path):
        # Real filesystem: the service home is mode 000, so stat() of the
        # store fails with EACCES. Newer pathlib returns False from
        # Path.exists() here instead of raising, so this must not rely on it.
        private_home = tmp_path / "usr-share-ollama"
        service = private_home / ".ollama" / "models"
        service.mkdir(parents=True)
        monkeypatch.setattr(ma, "LINUX_SERVICE_MODELS_DIR", service)
        private_home.chmod(0o000)
        try:
            assert ollama_models_dir({}, system="Linux", home=tmp_path / "nohome") == service
            # Free space is measured on the nearest accessible ancestor, which
            # is the private home itself (same filesystem as the store).
            assert existing_ancestor(service) == private_home
        finally:
            private_home.chmod(0o755)

    def test_linux_eacces_on_service_store_counts_as_present(self, monkeypatch, tmp_path):
        # Same case without relying on file modes (e.g. when running as root).
        service = tmp_path / "private" / ".ollama" / "models"
        real_stat = os.stat

        def fake_stat(path, *args, **kwargs):
            if Path(path) == service:
                raise PermissionError(13, "Permission denied", str(path))
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(ma, "LINUX_SERVICE_MODELS_DIR", service)
        monkeypatch.setattr(ma.os, "stat", fake_stat)
        assert ollama_models_dir({}, system="Linux", home=tmp_path / "nohome") == service

    def test_missing_paths_are_absent(self, tmp_path):
        assert ma._path_exists(tmp_path)
        assert not ma._path_exists(tmp_path / "missing" / "x")
        assert not ma._path_exists_or_hidden(tmp_path / "missing")
        # A file used as a directory component is ENOTDIR, not "hidden".
        (tmp_path / "f").write_text("x")
        assert not ma._path_exists_or_hidden(tmp_path / "f" / "models")

    def test_service_store_ignored_on_macos(self, monkeypatch, tmp_path):
        service = tmp_path / "svc"
        service.mkdir()
        monkeypatch.setattr(ma, "LINUX_SERVICE_MODELS_DIR", service)
        assert ollama_models_dir({}, system="Darwin", home=tmp_path / "h") == (
            tmp_path / "h" / ".ollama" / "models"
        )

    def test_env_beats_linux_service_store(self, monkeypatch, tmp_path):
        service = tmp_path / "svc"
        service.mkdir()
        monkeypatch.setattr(ma, "LINUX_SERVICE_MODELS_DIR", service)
        assert ollama_models_dir({"OLLAMA_MODELS": str(tmp_path / "custom")}, system="Linux") == (
            tmp_path / "custom"
        )


class TestDisk:
    def test_existing_ancestor_walks_up(self, tmp_path):
        assert existing_ancestor(tmp_path / "a" / "b" / "c") == tmp_path
        assert existing_ancestor(tmp_path) == tmp_path

    def test_check_disk_ok_and_margin(self, tmp_path):
        disk = check_disk(tmp_path, QWEN3_VL_TOTAL, disk_usage=_usage(28_150_000_000))
        assert disk.ok
        assert disk.margin_bytes == DEFAULT_MIN_FREE_BYTES
        assert disk.required_bytes == QWEN3_VL_TOTAL + 5 * GiB
        assert disk.shortfall_bytes == 0

    def test_check_disk_insufficient_measures_existing_ancestor(self, tmp_path):
        usage = MagicMock(return_value=SimpleNamespace(free=12_000_000_000))
        store = tmp_path / "missing" / "models"
        disk = check_disk(store, QWEN3_VL_TOTAL, disk_usage=usage)
        assert not disk.ok
        assert disk.shortfall_bytes == disk.required_bytes - 12_000_000_000 > 0
        usage.assert_called_once_with(tmp_path)
        assert disk.path == store and disk.measured_path == tmp_path

    def test_margin_fraction_dominates_for_large_downloads(self):
        assert required_margin_bytes(100 * GB) == 10 * GB
        assert required_margin_bytes(1 * GB) == DEFAULT_MIN_FREE_BYTES
        assert required_margin_bytes(1 * GB, min_free_bytes=0) == GB // 10

    def test_disk_guard_trips_below_min(self, tmp_path):
        free = {"value": 10 * GiB}
        guard = make_disk_guard(
            tmp_path, 5 * GiB, disk_usage=lambda _p: SimpleNamespace(free=free["value"])
        )
        assert guard() is None
        free["value"] = 4 * GiB
        reason = guard()
        assert reason is not None and "fell to" in reason and str(tmp_path) in reason
        assert "--min-free-gb is 5 GiB" in reason  # same unit as the flag

    def test_disk_guard_ignores_measurement_errors(self, tmp_path):
        def broken(_path):
            raise OSError("stat failed")

        assert make_disk_guard(tmp_path, 5 * GiB, disk_usage=broken)() is None


# ---------------------------------------------------------------------------
# Formatting and hosts
# ---------------------------------------------------------------------------


class TestFormattingAndHosts:
    def test_format_gb(self):
        assert format_gb(QWEN3_VL_TOTAL) == "20.9 GB"
        assert format_gb(0) == "0.0 GB"

    def test_format_gib(self):
        assert format_gib(5 * GiB) == "5 GiB"
        assert format_gib(10 * GiB) == "10 GiB"
        assert format_gib(int(2.5 * GiB)) == "2.5 GiB"
        assert format_gib(0) == "0 GiB"

    def test_margin_text_shows_floor_in_gib(self, tmp_path):
        # --min-free-gb is GiB: the default floor reads "5 GiB", not "5.4 GB".
        disk = check_disk(tmp_path, QWEN3_VL_TOTAL, disk_usage=_usage(100 * GB))
        assert disk.min_free_bytes == DEFAULT_MIN_FREE_BYTES
        assert disk.margin_text == "5 GiB (--min-free-gb)"
        disk = check_disk(tmp_path, 1 * GB, min_free_bytes=10 * GiB, disk_usage=_usage(100 * GB))
        assert disk.margin_text == "10 GiB (--min-free-gb)"

    def test_margin_text_shows_fraction_in_gb(self, tmp_path):
        disk = check_disk(tmp_path, 100 * GB, disk_usage=_usage(200 * GB))
        assert disk.margin_bytes == 10 * GB
        assert disk.margin_text == "10.0 GB (10% of the download)"

    @pytest.mark.parametrize(
        "url",
        ["http://localhost:11434", "http://127.0.0.1:11434", "http://0.0.0.0:11434",
         "http://[::1]:11434", "http://LOCALHOST:11434"],
    )
    def test_local_urls(self, url):
        assert is_local_url(url) is True

    def test_remote_url(self, monkeypatch):
        monkeypatch.setattr("lib.model_availability.socket.gethostname", lambda: "my-laptop")
        assert is_local_url("http://gpu-box:11434") is False

    def test_own_hostname_is_local(self, monkeypatch):
        monkeypatch.setattr("lib.model_availability.socket.gethostname", lambda: "My-Laptop")
        assert is_local_url("http://my-laptop:11434") is True
