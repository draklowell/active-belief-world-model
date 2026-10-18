import dataclasses
import json
import logging
import os
import subprocess
import sys
import threading
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
    force=True,
)


@dataclasses.dataclass
class ServerConfig:
    """Configuration for the vLLM + Tailscale inference server.

    Attributes:
        hostname: Tailscale hostname to advertise.
        tailscale_auth_key: Tailscale auth key for login.
        hf_token: Hugging Face token for model downloads.
        model: Hugging Face model identifier to serve.
        served_model_name: API alias exposed by the OpenAI-compatible endpoint.
        port: Port the inference server listens on.
        host: Bind address for the inference server.
        max_model_len: Maximum context length in tokens.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        socks5_port: SOCKS5 proxy port exposed by tailscaled.
        skip_install: Skip dependency installation steps when True.
        enforce_eager: Whether to enforce eager execution in vLLM (may reduce memory usage).
    """

    hostname: str
    tailscale_auth_key: str
    hf_token: str
    model: str
    served_model_name: str
    port: int = 8000
    host: str = "0.0.0.0"
    max_model_len: int = 16384
    tensor_parallel_size: int = 1
    socks5_port: int = 1055
    skip_install: bool = False
    enforce_eager: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "ServerConfig":
        """Construct a ServerConfig from a dictionary, ignoring unknown keys.

        Args:
            data: Mapping of config field names to values.

        Returns:
            A populated ServerConfig instance with defaults for omitted fields.
        """
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


CONFIG_JSON: str = """{}"""


def load_config() -> ServerConfig:
    """Load embedded server configuration.

    Returns:
        A ServerConfig instance parsed from the embedded CONFIG_JSON string.
    """
    data = json.loads(CONFIG_JSON)
    if not data:
        raise RuntimeError(
            "Embedded CONFIG_JSON is empty. Run through host_inference_server.py to embed config."
        )
    return ServerConfig.from_dict(data)


class TailscaleManager:
    """Manages the Tailscale daemon lifecycle including installation, login, and automatic restart.

    Attributes:
        hostname: The Tailscale hostname to advertise.
        auth_key: The Tailscale auth key used for login.
        socks5_port: The local SOCKS5 proxy port exposed by the daemon.

    Args:
        hostname: The Tailscale hostname to advertise.
        auth_key: The Tailscale auth key used for login.
        socks5_port: The local SOCKS5 proxy port for the daemon to listen on.
    """

    hostname: str
    auth_key: str
    socks5_port: int

    def __init__(self, hostname: str, auth_key: str, socks5_port: int = 1055) -> None:
        self.hostname = hostname
        self.auth_key = auth_key
        self.socks5_port = socks5_port
        self._log = logging.getLogger("TailscaleManager")
        self._stop_event = threading.Event()
        self._daemon_process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def install(self) -> None:
        """Install Tailscale via the official install script."""
        self._log.info("Installing Tailscale...")
        subprocess.run(
            "curl -fsSL https://tailscale.com/install.sh | sh",
            shell=True,
            check=True,
        )
        self._log.info("Tailscale installed successfully.")

    def _start_daemon(self) -> subprocess.Popen:
        self._log.info(
            "Starting tailscaled (userspace networking, SOCKS5 :%d)...", self.socks5_port
        )
        process = subprocess.Popen(
            [
                "tailscaled",
                "-tun=userspace-networking",
                f"--socks5-server=localhost:{self.socks5_port}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self._log.info("tailscaled started (pid=%d).", process.pid)
        return process

    def _login(self) -> None:
        self._log.info("Logging into Tailscale as '%s'...", self.hostname)
        subprocess.run(
            [
                "tailscale",
                "up",
                f"--hostname={self.hostname}",
                f"--auth-key={self.auth_key}",
            ],
            check=True,
        )
        self._log.info("Tailscale login successful.")

    def _stream_daemon_logs(self, process: subprocess.Popen) -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            self._log.info("[tailscaled] %s", line.decode(errors="replace").rstrip())

    def _run(self) -> None:
        self._daemon_process = self._start_daemon()
        time.sleep(2)
        self._login()

        log_thread = threading.Thread(
            target=self._stream_daemon_logs,
            args=(self._daemon_process,),
            daemon=True,
        )
        log_thread.start()

        while not self._stop_event.is_set():
            return_code = self._daemon_process.poll()
            if return_code is not None:
                self._log.warning("tailscaled exited with code %d, restarting...", return_code)
                self._daemon_process = self._start_daemon()
                log_thread = threading.Thread(
                    target=self._stream_daemon_logs,
                    args=(self._daemon_process,),
                    daemon=True,
                )
                log_thread.start()
            self._stop_event.wait(timeout=5)

        self._log.info(
            "Stop requested, terminating tailscaled (pid=%d)...", self._daemon_process.pid
        )
        self._daemon_process.terminate()
        try:
            self._daemon_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._log.warning("tailscaled did not exit cleanly, killing it.")
            self._daemon_process.kill()
        self._log.info("tailscaled stopped.")

    def start(self) -> None:
        """Start the Tailscale management thread (non-blocking)."""
        self._thread = threading.Thread(target=self._run, name="TailscaleManager", daemon=True)
        self._thread.start()
        self._log.info("TailscaleManager thread started.")

    def stop(self) -> None:
        """Signal the management thread to stop and wait for it to finish."""
        self._log.info("Stopping TailscaleManager...")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._log.info("TailscaleManager stopped.")

    def is_alive(self) -> bool:
        """Return whether the management thread is still running."""
        return self._thread is not None and self._thread.is_alive()


class VLLMManager:
    """Manages vLLM installation and the inference server process.

    The inference server is started once. If the vLLM process exits for any reason,
    the manager signals the caller via the provided stop event rather than restarting.

    Attributes:
        model: The Hugging Face model identifier to serve.
        served_model_name: The name exposed by the OpenAI-compatible API.
        hf_token: The Hugging Face token set when launching the vLLM subprocess.
        tensor_parallel_size: Number of GPUs used for tensor parallelism.
        port: The port the inference server listens on.
        host: The host address the inference server binds to.
        max_model_len: The maximum sequence length in tokens.
        enforce_eager: Whether to enforce eager execution in vLLM (may reduce memory usage).

    Args:
        model: The Hugging Face model identifier.
        served_model_name: The alias exposed via the API.
        hf_token: Hugging Face token for authenticated model downloads.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        port: Port number for the inference server.
        host: Bind address for the inference server.
        max_model_len: Maximum context length in tokens.
        enforce_eager: Whether to enforce eager execution in vLLM (may reduce memory usage).
    """

    model: str
    served_model_name: str
    hf_token: str
    tensor_parallel_size: int
    port: int
    host: str
    max_model_len: int
    enforce_eager: bool

    def __init__(
        self,
        model: str,
        served_model_name: str,
        hf_token: str,
        tensor_parallel_size: int = 1,
        port: int = 8000,
        host: str = "0.0.0.0",
        max_model_len: int = 16384,
        enforce_eager: bool = True,
    ) -> None:
        self.model = model
        self.served_model_name = served_model_name
        self.hf_token = hf_token
        self.tensor_parallel_size = tensor_parallel_size
        self.port = port
        self.host = host
        self.max_model_len = max_model_len
        self.enforce_eager = enforce_eager
        self._log = logging.getLogger("VLLMManager")
        self._process: subprocess.Popen | None = None
        self._managed_thread: threading.Thread | None = None

    def install(self) -> None:
        """Install vllm and vllm-omni via pip."""
        self._log.info("Installing vllm and vllm-omni...")
        subprocess.run(
            ["pip", "install", "vllm==0.30.0", "vllm-omni==0.30.0"],
            check=True,
        )
        self._log.info("vllm and vllm-omni installed successfully.")

    def _stream_logs(self, process: subprocess.Popen) -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            self._log.info("[vllm] %s", line.decode(errors="replace").rstrip())

    def _run(self, external_stop_event: threading.Event) -> None:
        self._log.info(
            "Starting vLLM inference server: model=%s, host=%s, port=%d, tensor_parallel_size=%d",
            self.model,
            self.host,
            self.port,
            self.tensor_parallel_size,
        )

        enforce_eager = []
        if self.enforce_eager:
            enforce_eager = ["--enforce-eager"]

        self._process = subprocess.Popen(
            [
                "vllm",
                "serve",
                self.model,
                "--port",
                str(self.port),
                "--host",
                self.host,
                *enforce_eager,
                "--max-model-len",
                str(self.max_model_len),
                "--served-model-name",
                self.served_model_name,
                "--tensor-parallel-size",
                str(self.tensor_parallel_size),
            ],
            env={**os.environ, "VLLM_TORCH_COMPILE": "0", "HF_TOKEN": self.hf_token},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self._log.info("vLLM server started (pid=%d).", self._process.pid)

        log_thread = threading.Thread(
            target=self._stream_logs,
            args=(self._process,),
            daemon=True,
        )
        log_thread.start()

        return_code = self._process.wait()
        self._log.error(
            "vLLM server exited unexpectedly with code %d. Signalling shutdown.", return_code
        )
        external_stop_event.set()

    def start(self, external_stop_event: threading.Event) -> None:
        """Start the vLLM inference server in a background thread.

        Args:
            external_stop_event: Event set when vLLM exits, signalling the caller to shut down.
        """
        self._managed_thread = threading.Thread(
            target=self._run,
            args=(external_stop_event,),
            name="VLLMManager",
            daemon=True,
        )
        self._managed_thread.start()
        self._log.info("VLLMManager thread started.")

    def stop(self) -> None:
        """Terminate the vLLM inference server if it is running."""
        self._log.info("Stopping VLLMManager...")
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._log.warning("vLLM did not exit cleanly, killing it.")
                self._process.kill()
        if self._managed_thread is not None:
            self._managed_thread.join()
        self._log.info("VLLMManager stopped.")

    def is_alive(self) -> bool:
        """Return whether the vLLM server thread is still running."""
        return self._managed_thread is not None and self._managed_thread.is_alive()


def main() -> None:
    """Load config, start Tailscale and vLLM in parallel, and wait for shutdown."""
    log = logging.getLogger("main")
    config = load_config()
    log.info(
        "Loaded config: model=%s, hostname=%s, tensor_parallel_size=%d",
        config.model,
        config.hostname,
        config.tensor_parallel_size,
    )

    tailscale = TailscaleManager(
        hostname=config.hostname,
        auth_key=config.tailscale_auth_key,
        socks5_port=config.socks5_port,
    )
    vllm = VLLMManager(
        model=config.model,
        served_model_name=config.served_model_name,
        hf_token=config.hf_token,
        tensor_parallel_size=config.tensor_parallel_size,
        port=config.port,
        host=config.host,
        max_model_len=config.max_model_len,
        enforce_eager=config.enforce_eager,
    )

    if not config.skip_install:
        tailscale.install()
        vllm.install()

    shutdown_event = threading.Event()

    tailscale.start()
    time.sleep(3)
    vllm.start(external_stop_event=shutdown_event)

    log.info("Both services running. Waiting for shutdown signal...")
    try:
        shutdown_event.wait()
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received, shutting down...")
    finally:
        log.info("Initiating shutdown sequence...")
        vllm.stop()
        tailscale.stop()
        log.info("Shutdown complete.")


if __name__ == "__main__":
    main()
