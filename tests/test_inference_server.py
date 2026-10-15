import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).parent.parent
KERNEL_DIR = REPO_ROOT / "src" / "kernels" / "inference-server"
SCRIPTS_DIR = REPO_ROOT / "scripts"
FIXTURES_DIR = Path(__file__).parent / "fixtures"

MINIMAL_CONFIG = {
    "hostname": "test-host",
    "tailscale_auth_key": "tskey-test",
    "hf_token": "hf_test",
    "model": "Qwen/Qwen3-VL-4B-Instruct",
    "served_model_name": "qwen",
}


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def core() -> ModuleType:
    target = KERNEL_DIR / "core.template.py"
    if not target.exists():
        target = KERNEL_DIR / "core.py"
    return load_module("kernel_core", target)


@pytest.fixture(scope="module")
def orchestrator() -> ModuleType:
    return load_module("host_inference_server", SCRIPTS_DIR / "host_inference_server.py")


class TestServerConfig:
    def test_from_dict_required_fields(self, core: ModuleType) -> None:
        config = core.ServerConfig.from_dict(MINIMAL_CONFIG)
        assert config.hostname == "test-host"
        assert config.tailscale_auth_key == "tskey-test"
        assert config.hf_token == "hf_test"
        assert config.model == "Qwen/Qwen3-VL-4B-Instruct"
        assert config.served_model_name == "qwen"

    def test_from_dict_applies_defaults(self, core: ModuleType) -> None:
        config = core.ServerConfig.from_dict(MINIMAL_CONFIG)
        assert config.port == 8000
        assert config.host == "0.0.0.0"
        assert config.max_model_len == 16384
        assert config.tensor_parallel_size == 1
        assert config.socks5_port == 1055
        assert config.skip_install is False

    def test_from_dict_accepts_overrides(self, core: ModuleType) -> None:
        config = core.ServerConfig.from_dict(
            {**MINIMAL_CONFIG, "tensor_parallel_size": 2, "port": 9000}
        )
        assert config.tensor_parallel_size == 2
        assert config.port == 9000

    def test_from_dict_ignores_unknown_keys(self, core: ModuleType) -> None:
        config = core.ServerConfig.from_dict({**MINIMAL_CONFIG, "unknown_field": "value"})
        assert config.hostname == "test-host"

    def test_load_config_from_embedded_json(
        self, core: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(core, "CONFIG_JSON", json.dumps(MINIMAL_CONFIG))
        config = core.load_config()
        assert config.hostname == MINIMAL_CONFIG["hostname"]
        assert config.hf_token == MINIMAL_CONFIG["hf_token"]
        assert config.model == MINIMAL_CONFIG["model"]

    def test_load_config_empty_raises(
        self, core: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(core, "CONFIG_JSON", "{}")
        with pytest.raises(RuntimeError, match="Embedded CONFIG_JSON is empty"):
            core.load_config()


class TestKernelMetadataTemplate:
    def test_template_is_valid_json(self) -> None:
        template_path = KERNEL_DIR / "kernel-metadata.template.json"
        assert template_path.exists(), "kernel-metadata.template.json not found"
        data = json.loads(template_path.read_text())
        assert isinstance(data, dict)

    def test_template_has_required_fields(self) -> None:
        data = json.loads((KERNEL_DIR / "kernel-metadata.template.json").read_text())
        for field in ("title", "code_file", "language", "kernel_type"):
            assert field in data, f"Missing required field: {field}"

    def test_template_has_no_id(self) -> None:
        data = json.loads((KERNEL_DIR / "kernel-metadata.template.json").read_text())
        assert "id" not in data, "Template must not contain 'id' — it is injected at push time."

    def test_template_has_no_env(self) -> None:
        data = json.loads((KERNEL_DIR / "kernel-metadata.template.json").read_text())
        assert "env" not in data, "Template must not contain 'env'."

    def test_template_code_file_points_to_core(self) -> None:
        data = json.loads((KERNEL_DIR / "kernel-metadata.template.json").read_text())
        assert data["code_file"] == "core.py"


class TestPatchMetadata:
    def test_injects_id(self, orchestrator: ModuleType) -> None:
        result = orchestrator.patch_metadata("alice", "my-kernel", "NvidiaTeslaT4")
        assert result["id"] == "alice/my-kernel"

    def test_injects_machine_shape(self, orchestrator: ModuleType) -> None:
        result = orchestrator.patch_metadata("alice", "my-kernel", "NvidiaL4")
        assert result["machine_shape"] == "NvidiaL4"

    def test_has_no_env(self, orchestrator: ModuleType) -> None:
        result = orchestrator.patch_metadata("alice", "my-kernel", "NvidiaTeslaT4")
        assert "env" not in result

    def test_preserves_template_fields(self, orchestrator: ModuleType) -> None:
        result = orchestrator.patch_metadata("alice", "my-kernel", "NvidiaTeslaT4")
        assert result["code_file"] == "core.py"
        assert result["language"] == "python"


class TestEmbedConfig:
    def test_embed_config_replaces_empty_placeholder(self, orchestrator: ModuleType) -> None:
        template = 'CONFIG_JSON: str = """{}"""\n'
        result = orchestrator.embed_config(template, MINIMAL_CONFIG)
        assert MINIMAL_CONFIG["hostname"] in result

    def test_embed_config_replaces_existing_config(self, orchestrator: ModuleType) -> None:
        old_template = 'CONFIG_JSON: str = """\n{"hostname": "old"}\n"""\n'
        result = orchestrator.embed_config(old_template, MINIMAL_CONFIG)
        assert MINIMAL_CONFIG["hostname"] in result
        assert '"old"' not in result

    def test_embed_config_missing_placeholder_raises(self, orchestrator: ModuleType) -> None:
        template = "x = 1\n"
        with pytest.raises(RuntimeError, match="CONFIG_JSON definition not found"):
            orchestrator.embed_config(template, MINIMAL_CONFIG)


class TestApplyBackendDefaults:
    def test_kaggle_sets_tensor_parallel_2(self, orchestrator: ModuleType) -> None:
        config: dict = {}
        orchestrator.apply_backend_defaults(config, "kaggle")
        assert config["tensor_parallel_size"] == 2

    def test_colab_sets_tensor_parallel_1(self, orchestrator: ModuleType) -> None:
        config: dict = {}
        orchestrator.apply_backend_defaults(config, "colab")
        assert config["tensor_parallel_size"] == 1

    def test_explicit_value_not_overridden(self, orchestrator: ModuleType) -> None:
        config = {"tensor_parallel_size": 4}
        orchestrator.apply_backend_defaults(config, "kaggle")
        assert config["tensor_parallel_size"] == 4


class TestFixtureConfig:
    def test_dummy_config_is_valid(self, core: ModuleType) -> None:
        data = json.loads((FIXTURES_DIR / "dummy_config.json").read_text())
        config = core.ServerConfig.from_dict(data)
        assert config.hostname == "test-host"


class TestColabSession:
    def test_stop_colab_session_calls_cli(
        self, orchestrator: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called_cmd: list[list[str]] = []

        def mock_run(cmd: list[str], check: bool = False) -> None:
            called_cmd.append(cmd)

        monkeypatch.setattr(orchestrator.subprocess, "run", mock_run)
        orchestrator.stop_colab_session("test-session")
        assert called_cmd == [["colab", "stop", "-s", "test-session"]]

    def test_parse_args_defaults_session(
        self, orchestrator: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            ["host_inference_server.py", "--config", "cfg.json", "--backend", "colab"],
        )
        args = orchestrator.parse_args()
        assert args.session == "inference-server"
