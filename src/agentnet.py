import struct
import zlib
from pathlib import Path

import httpx
import polars as pl

REPO_ID = "xlangai/AgentNet"
DATA_DIR = Path(__file__).resolve().parents[1] / "data/raw/agentnet"
REVISION_FILE = DATA_DIR / "REVISION.txt"
SUBSET_FILES = {
    "meta": "meta_data_merged.jsonl",
    "ubuntu": "agentnet_ubuntu_5k.jsonl",
    "win_mac": "agentnet_win_mac_18k.jsonl",
}
IMAGES_DIR = DATA_DIR / "images"
IMAGE_INDEX_FILE = DATA_DIR / "ubuntu_images_index.csv"
IMAGE_MANIFEST_FILE = DATA_DIR / "ubuntu_images_manifest.csv"
IMAGE_SUBDIR = "ubuntu_images"
INDEX_TAIL_BYTES = 4 * 1024 * 1024
LOCAL_HEADER_SLACK = 512
ACTION_PATTERN = r"(\w+(?:\.\w+)*)\("
STATUS_PATTERN = r"status=['\"](\w+)['\"]"


def scan_jsonl(path: Path) -> pl.LazyFrame:
    """Lazily scan a JSON Lines file into a Polars frame.

    Args:
        path: Path to a JSON Lines file.

    Returns:
        Lazy frame with the schema inferred from every row.
    """
    return pl.scan_ndjson(path, infer_schema_length=None)


def task_table(trajectories: pl.LazyFrame) -> pl.LazyFrame:
    """Reduce trajectories to one row per task with its step count.

    Args:
        trajectories: Lazy frame with one row per task and a `traj` list column.

    Returns:
        Lazy frame without `traj` and with an `n_steps` column.
    """
    return trajectories.with_columns(pl.col("traj").list.len().alias("n_steps")).drop("traj")


def step_table(trajectories: pl.LazyFrame) -> pl.LazyFrame:
    """Flatten trajectories into one row per step with parsed action fields.

    Args:
        trajectories: Lazy frame with one row per task and a `traj` list column.

    Returns:
        Lazy frame with one row per step, holding the action name, pointer coordinates and status.
    """
    return (
        trajectories.select("task_id", "traj")
        .explode("traj")
        .unnest("traj")
        .unnest("value")
        .with_columns(
            pl.col("code").str.extract(ACTION_PATTERN, 1).alias("action_name"),
            pl.col("thought").str.len_chars().alias("thought_chars"),
            pl.col("code").str.extract(STATUS_PATTERN, 1).alias("terminate_status"),
            pl.col("code").str.extract(r"x=([0-9.]+)", 1).cast(pl.Float64).alias("x"),
            pl.col("code").str.extract(r"y=([0-9.]+)", 1).cast(pl.Float64).alias("y"),
            pl.col("observation").str.len_chars().alias("observation_chars"),
            pl.col("reflection").str.len_chars().alias("reflection_chars"),
            pl.col("marks").list.len().alias("n_marks"),
        )
        .drop("marks", "thought", "observation", "reflection", strict=False)
    )


def pinned_commit() -> str:
    """Read the dataset commit sha recorded in the revision file.

    Returns:
        Commit sha that the dataset files are pinned to.
    """
    return REVISION_FILE.read_text().strip().split("@")[1]


def segment_url(commit: str, disk: int, last_disk: int) -> str:
    """Build the download URL of one segment of the image archive.

    Args:
        commit: Dataset commit sha.
        disk: Zero based index of the segment.
        last_disk: Index of the final segment, which is stored as `images.zip`.

    Returns:
        Direct download URL of the segment.
    """
    part = "images.zip" if disk == last_disk else f"images.z{disk + 1:02d}"
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/{commit}/{IMAGE_SUBDIR}/{part}"


def parse_central_directory(data: bytes) -> pl.DataFrame:
    """Parse the entries of a zip central directory held in memory.

    Args:
        data: Raw bytes of the central directory.

    Returns:
        One row per entry with its name, CRC32, sizes, segment index and local header offset.
    """
    rows = []
    position = 0
    while data[position : position + 4] == b"PK\x01\x02":
        fields = struct.unpack("<IHHHHHHIIIHHHHHII", data[position : position + 46])
        crc, compressed, uncompressed, name_len, extra_len, comment_len, disk = fields[7:14]
        offset = fields[16]
        name = data[position + 46 : position + 46 + name_len].decode()
        extra = data[position + 46 + name_len : position + 46 + name_len + extra_len]
        cursor = 0
        while cursor + 4 <= len(extra):
            tag, size = struct.unpack("<HH", extra[cursor : cursor + 4])
            if tag == 1:
                body = extra[cursor + 4 : cursor + 4 + size]
                offset_in_body = 0
                if uncompressed == 0xFFFFFFFF:
                    (uncompressed,) = struct.unpack("<Q", body[offset_in_body : offset_in_body + 8])
                    offset_in_body += 8
                if compressed == 0xFFFFFFFF:
                    (compressed,) = struct.unpack("<Q", body[offset_in_body : offset_in_body + 8])
                    offset_in_body += 8
                if offset == 0xFFFFFFFF:
                    (offset,) = struct.unpack("<Q", body[offset_in_body : offset_in_body + 8])
                    offset_in_body += 8
                if disk == 0xFFFF:
                    (disk,) = struct.unpack("<I", body[offset_in_body : offset_in_body + 4])
            cursor += 4 + size
        rows.append((name, crc, compressed, uncompressed, disk, offset))
        position += 46 + name_len + extra_len + comment_len
    return pl.DataFrame(
        rows,
        schema=["name", "crc32", "compressed_bytes", "uncompressed_bytes", "disk", "offset"],
        orient="row",
    )


def read_zip_index(client: httpx.Client, url: str) -> pl.DataFrame:
    """Fetch and parse the central directory of a remote zip with range requests.

    Args:
        client: HTTP client used for range requests.
        url: URL of the final segment of the archive, which holds the central directory.

    Returns:
        One row per archive entry, as produced by parse_central_directory.
    """
    size = int(client.head(url).headers["content-length"])
    tail = client.get(url, headers={"Range": f"bytes={size - INDEX_TAIL_BYTES}-{size - 1}"}).content
    locator = tail.rfind(b"PK\x06\x07")
    record = tail.rfind(b"PK\x06\x06", 0, locator)
    directory_size, directory_offset = struct.unpack("<QQ", tail[record + 40 : record + 56])
    directory = client.get(
        url, headers={"Range": f"bytes={directory_offset}-{directory_offset + directory_size - 1}"}
    ).content
    return parse_central_directory(directory)


def inflate_entry(chunk: bytes, compressed_bytes: int, crc32: int) -> bytes:
    """Decompress one zip entry from its local header and verify its CRC32.

    Args:
        chunk: Bytes starting at the local file header of the entry.
        compressed_bytes: Length of the compressed payload.
        crc32: Expected CRC32 of the uncompressed content.

    Returns:
        Uncompressed bytes of the entry.
    """
    name_len, extra_len = struct.unpack("<HH", chunk[26:30])
    start = 30 + name_len + extra_len
    data = zlib.decompress(chunk[start : start + compressed_bytes], -15)
    if zlib.crc32(data) != crc32:
        raise ValueError("CRC32 mismatch for downloaded zip entry")
    return data


def fetch_entry(
    client: httpx.Client, url: str, offset: int, compressed_bytes: int, crc32: int
) -> bytes:
    """Download one zip entry with a range request and return its uncompressed bytes.

    Args:
        client: HTTP client used for range requests.
        url: URL of the segment that holds the entry.
        offset: Byte offset of the local file header of the entry.
        compressed_bytes: Length of the compressed payload.
        crc32: Expected CRC32 of the uncompressed content.

    Returns:
        Uncompressed bytes of the entry.
    """
    end = offset + 30 + compressed_bytes + LOCAL_HEADER_SLACK
    chunk = client.get(url, headers={"Range": f"bytes={offset}-{end}"}).content
    return inflate_entry(chunk, compressed_bytes, crc32)


def png_size(path: Path) -> tuple[int, int]:
    """Read the width and height of a PNG file from its IHDR chunk.

    Args:
        path: Path to a PNG file.

    Returns:
        Width and height in pixels.
    """
    with path.open("rb") as file:
        header = file.read(24)
    width, height = struct.unpack(">II", header[16:24])
    return width, height
