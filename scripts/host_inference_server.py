import argparse
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

from decouple import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
    force=True,
)

KERNEL_DIR = Path(__file__).parent.parent / "src" / "kernels" / "inference-server"
CORE_PATH = KERNEL_DIR / "core.py"
CORE_TEMPLATE_PATH = KERNEL_DIR / "core.template.py"
TEMPLATE_PATH = KERNEL_DIR / "kernel-metadata.template.json"
RUNTIME_METADATA_PATH = KERNEL_DIR / "kernel-metadata.json"


def read_kaggle_username() -> str:
    """Read the Kaggle username from the local Kaggle CLI configuration.

    Returns:
        The username string as configured in ~/.kaggle/kaggle.json.
    """
    result = subprocess.run(
        ["kaggle", "config", "view"],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        match = re.match(r"\s*-\s*username:\s*(\S+)", line)
        if match:
            return match.group(1)
    raise RuntimeError("Could not parse username from `kaggle config view` output.")


def apply_backend_defaults(server_config: dict, backend: str) -> dict:
    """Apply platform-specific defaults to a config dict when fields are absent.

    Args:
        server_config: Mutable config dictionary to augment.
        backend: Inference platform identifier, either "kaggle" or "colab".

    Returns:
        The same config dictionary with defaults applied in-place.
    """
    if "tensor_parallel_size" not in server_config:
        server_config["tensor_parallel_size"] = 2 if backend == "kaggle" else 1
    return server_config


def load_secrets() -> dict:
    """Load secrets from environment variables via decouple.

    Returns:
        Dictionary mapping secret keys to their values.
    """
    return {
        "tailscale_auth_key": config("TAILSCALE_AUTH_KEY", default=""),
        "hf_token": config("HF_TOKEN", default=""),
    }


def embed_config(source_code: str, server_config: dict) -> str:
    """Embed the configuration dictionary as a JSON string in Python source code.

    Args:
        source_code: Python code containing a CONFIG_JSON definition.
        server_config: Configuration dictionary to serialize and embed.

    Returns:
        Updated source code with serialized JSON embedded in CONFIG_JSON.
    """
    config_json = json.dumps(server_config, indent=2)
    pattern = r'CONFIG_JSON:\s*str\s*=\s*""".*?"""'
    updated_code, count = re.subn(
        pattern,
        lambda _: f'CONFIG_JSON: str = """\n{config_json}\n"""',
        source_code,
        flags=re.DOTALL,
    )
    if count == 0:
        raise RuntimeError("CONFIG_JSON definition not found in kernel template code.")
    return updated_code


def embed_secrets(source_code: str, secrets: dict) -> str:
    """Embed the secrets dictionary as a JSON string in Python source code.

    Args:
        source_code: Python code containing a SECRETS_JSON definition.
        secrets: Secrets dictionary to serialize and embed.

    Returns:
        Updated source code with serialized JSON embedded in SECRETS_JSON.
    """
    secrets_json = json.dumps(secrets, indent=2)
    pattern = r'SECRETS_JSON:\s*str\s*=\s*""".*?"""'
    updated_code, count = re.subn(
        pattern,
        lambda _: f'SECRETS_JSON: str = """\n{secrets_json}\n"""',
        source_code,
        flags=re.DOTALL,
    )
    if count == 0:
        raise RuntimeError("SECRETS_JSON definition not found in kernel template code.")
    return updated_code


def embed_config_to_core(server_config: dict, secrets: dict) -> None:
    """Generate core.py by embedding configuration and secrets into core.template.py.

    Args:
        server_config: Configuration dictionary to embed.
        secrets: Secrets dictionary to embed.
    """
    source_code = CORE_TEMPLATE_PATH.read_text()
    source_code = embed_config(source_code, server_config)
    source_code = embed_secrets(source_code, secrets)
    CORE_PATH.write_text(source_code)


def patch_metadata(username: str, kernel_slug: str, accelerator: str | None = None) -> dict:
    """Build a runtime kernel-metadata dict by patching the committed template.

    Args:
        username: Kaggle username to prefix the kernel id.
        kernel_slug: Kaggle kernel slug (the part after the username).
        accelerator: Kaggle machine shape string (e.g. "NvidiaTeslaT4").

    Returns:
        Patched metadata dictionary ready to serialize as kernel-metadata.json.
    """
    template = json.loads(TEMPLATE_PATH.read_text())
    template["id"] = f"{username}/{kernel_slug}"
    template["machine_shape"] = accelerator
    return template


def run_kaggle(args: argparse.Namespace) -> None:
    """Push and monitor a Kaggle kernel, deleting it on interrupt.

    Args:
        args: Parsed CLI arguments.
    """
    log = logging.getLogger("kaggle")

    username = read_kaggle_username()
    log.info("Detected Kaggle username: %s", username)

    server_config = json.loads(Path(args.config).read_text())
    apply_backend_defaults(server_config, "kaggle")
    secrets = load_secrets()

    embed_config_to_core(server_config, secrets)
    log.info("Embedded config and secrets from core.template.py into core.py.")

    accelerator = args.accelerator or "NvidiaTeslaT4"

    metadata = patch_metadata(username, args.session, accelerator)
    RUNTIME_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    log.info("Wrote runtime kernel-metadata.json.")

    kernel_ref = f"{username}/{args.session}"
    log.info("Pushing kernel %s...", kernel_ref)
    subprocess.run(["kaggle", "kernels", "push", "-p", str(KERNEL_DIR)], check=True)
    log.info("Kernel pushed. Monitoring (press Ctrl-C to stop and delete the kernel)...")

    try:
        while True:
            result = subprocess.run(
                ["kaggle", "kernels", "status", kernel_ref],
                capture_output=True,
                text=True,
            )
            log.info("[kaggle status] %s", result.stdout.strip())
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        log.info("Interrupted. Deleting kernel %s...", kernel_ref)
        subprocess.run(["kaggle", "kernels", "delete", kernel_ref, "--yes"])
        log.info("Kernel deleted.")


def stop_colab_session(session_name: str) -> None:
    """Stop and remove a Colab session via the Colab CLI.

    Args:
        session_name: The name or identifier of the session to remove.
    """
    log = logging.getLogger("colab")
    log.info("Stopping Colab session '%s'...", session_name)
    subprocess.run(["colab", "stop", "-s", session_name], check=False)
    log.info("Colab session '%s' stopped.", session_name)


def run_colab(args: argparse.Namespace) -> None:
    """Run core.py via the Colab CLI and stream its output.

    Args:
        args: Parsed CLI arguments.
    """
    log = logging.getLogger("colab")

    server_config = json.loads(Path(args.config).read_text())
    apply_backend_defaults(server_config, "colab")
    secrets = load_secrets()

    embed_config_to_core(server_config, secrets)
    log.info("Embedded config and secrets from core.template.py into core.py.")

    if args.accelerator == "cpu":
        accelerator = tuple()
    else:
        accelerator_type, accelerator_model = (args.accelerator or "gpu/t4").split("/", 1)
        accelerator = (f"--{accelerator_type}", accelerator_model)

    session_name = args.session
    log.info("Starting Colab run (session=%s)...", session_name)
    process = subprocess.Popen(
        ["colab", "run", *accelerator, "-s", session_name, str(CORE_PATH)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        if process.stdout is not None:
            for line in process.stdout:
                decoded_line = line.decode(errors="replace").rstrip()
                log.info("[colab] %s", decoded_line)
                match = re.match(
                    r"^\[colab\] Session READY \(([^)]+)\). Executing", decoded_line, re.IGNORECASE
                )
                if match:
                    session_name = match.group(1)
        process.wait()
        if process.returncode != 0:
            log.error("Colab run exited with code %d.", process.returncode)
            sys.exit(process.returncode)
    except KeyboardInterrupt:
        log.info("Interrupted. Terminating Colab run...")
        process.terminate()
        process.wait()
        log.info("Colab run terminated.")
    finally:
        stop_colab_session(session_name)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the inference server orchestrator."""
    parser = argparse.ArgumentParser(description="Launch an inference server on Kaggle or Colab.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON file containing ServerConfig fields.",
    )
    parser.add_argument(
        "--backend",
        choices=["kaggle", "colab"],
        required=True,
        help="Inference platform to target.",
    )
    parser.add_argument(
        "--accelerator",
        default=None,
        help="Kaggle or Colab machine shape (default: NvidiaTeslaT4 "
        "for kaggle and gpu/t4 for colab).",
    )
    parser.add_argument(
        "--session",
        default="inference-server",
        help="Colab session name or Kaggle kernel slug (default: inference-server).",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between Kaggle status polls (default: 30).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: dispatch to the Kaggle or Colab backend."""
    log = logging.getLogger("main")
    args = parse_args()
    log.info("Starting inference server orchestrator (backend=%s)...", args.backend)

    if args.backend == "kaggle":
        run_kaggle(args)
    else:
        run_colab(args)


if __name__ == "__main__":
    main()
