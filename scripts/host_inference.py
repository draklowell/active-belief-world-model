import argparse
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
            raise RuntimeError(
                "tailscaled stdout is not available for log streaming, use stdout=PIPE in Popen."
            )

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

    The inference server is started once. If vLLM exits for any reason,
    the manager signals the caller via the stop event rather than restarting.

    Attributes:
        model: The Hugging Face model identifier to serve.
        served_model_name: The name exposed by the OpenAI-compatible API.
        port: The port the inference server listens on.
        host: The host address the inference server binds to.
        max_model_len: The maximum sequence length in tokens.

    Args:
        model: The Hugging Face model identifier.
        served_model_name: The alias exposed via the API.
        port: Port number for the inference server.
        host: Bind address for the inference server.
        max_model_len: Maximum context length in tokens.
    """

    model: str
    served_model_name: str
    port: int
    host: str
    max_model_len: int

    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-4B-Instruct",
        served_model_name: str = "gpt4-o",
        port: int = 8000,
        host: str = "0.0.0.0",
        max_model_len: int = 16384,
    ) -> None:
        self.model = model
        self.served_model_name = served_model_name
        self.port = port
        self.host = host
        self.max_model_len = max_model_len
        self._log = logging.getLogger("VLLMManager")
        self._process: subprocess.Popen | None = None
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None

    def install(self) -> None:
        """Create project, install vllm and vllm-omni via uv."""
        self._log.info("Creating project...")
        subprocess.run(
            ["uv", "init"],
            check=True,
        )
        self._log.info("Installing vllm and vllm-omni...")
        subprocess.run(
            ["uv", "add", "vllm==0.30.0", "vllm-omni==0.30.0"],
            check=True,
        )
        self._log.info("vllm and vllm-omni installed successfully.")

    def _stream_logs(self, process: subprocess.Popen) -> None:
        if process.stdout is None:
            raise RuntimeError(
                "vllm stdout is not available for log streaming, use stdout=PIPE in Popen."
            )

        for line in process.stdout:
            self._log.info("[vllm] %s", line.decode(errors="replace").rstrip())

    def _run(self, external_stop_event: threading.Event) -> None:
        self._log.info(
            "Starting vLLM inference server: model=%s, host=%s, port=%d",
            self.model,
            self.host,
            self.port,
        )
        self._process = subprocess.Popen(
            [
                "uv",
                "run",
                "vllm",
                "serve",
                self.model,
                "--port",
                str(self.port),
                "--host",
                self.host,
                "--enforce-eager",
                "--max-model-len",
                str(self.max_model_len),
                "--served-model-name",
                self.served_model_name,
            ],
            env={**os.environ, "VLLM_TORCH_COMPILE": "0"},
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
            external_stop_event: Event that gets set if vLLM exits, signalling the
                caller to shut down.
        """
        self._stop_event = external_stop_event
        self._thread = threading.Thread(
            target=self._run,
            args=(external_stop_event,),
            name="VLLMManager",
            daemon=True,
        )
        self._thread.start()
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
        if self._thread is not None:
            self._thread.join()
        self._log.info("VLLMManager stopped.")

    def is_alive(self) -> bool:
        """Return whether the vLLM server thread is still running."""
        return self._thread is not None and self._thread.is_alive()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Start a Tailscale-connected vLLM inference server."
    )
    parser.add_argument("--hostname", required=True, help="Tailscale hostname to advertise.")
    parser.add_argument("--auth-key", required=True, help="Tailscale auth key.")
    parser.add_argument(
        "--socks5-port",
        type=int,
        default=1055,
        help="SOCKS5 proxy port for tailscaled (default: 1055).",
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-VL-4B-Instruct", help="Hugging Face model ID to serve."
    )
    parser.add_argument(
        "--served-model-name",
        default="qwen",
        help="Model alias exposed via the API (default: qwen).",
    )
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port (default: 8000).")
    parser.add_argument(
        "--host", default="0.0.0.0", help="vLLM server bind address (default: 0.0.0.0)."
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="Maximum context length in tokens (default: 16384).",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip installation steps (useful when dependencies are already present).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: install dependencies, start Tailscale and vLLM, then wait."""
    log = logging.getLogger("main")
    args = parse_args()

    tailscale = TailscaleManager(
        hostname=args.hostname,
        auth_key=args.auth_key,
        socks5_port=args.socks5_port,
    )
    vllm = VLLMManager(
        model=args.model,
        served_model_name=args.served_model_name,
        port=args.port,
        host=args.host,
        max_model_len=args.max_model_len,
    )

    log.info("Starting Tailscale and vLLM managers...")

    if not args.skip_install:
        tailscale.install()
        vllm.install()

    shutdown_event = threading.Event()

    tailscale.start()
    time.sleep(3)

    vllm.start(external_stop_event=shutdown_event)

    log.info("Both services are running. Waiting for shutdown signal...")
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
