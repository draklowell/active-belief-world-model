import io
import struct
import zipfile
from pathlib import Path

import pytest

from agentnet import inflate_entry, parse_central_directory, png_size, segment_url


def build_archive(files: dict[str, bytes]) -> bytes:
    """Build an in-memory zip archive from named byte contents.

    Args:
        files: Mapping of archive member names to their content.

    Returns:
        Bytes of the zip archive.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def central_directory(archive: bytes) -> bytes:
    """Extract the central directory bytes from a zip archive.

    Args:
        archive: Bytes of a zip archive.

    Returns:
        Bytes of the central directory.
    """
    end = archive.rfind(b"PK\x05\x06")
    size, offset = struct.unpack("<II", archive[end + 12 : end + 20])
    return archive[offset : offset + size]


FILES = {"a.png": b"alpha" * 500, "b.png": bytes(range(256)) * 20}


def test_parse_central_directory_matches_zipfile() -> None:
    """Check that the parsed index agrees with the zipfile module."""
    archive = build_archive(FILES)
    index = parse_central_directory(central_directory(archive))
    with zipfile.ZipFile(io.BytesIO(archive)) as reference:
        infos = {info.filename: info for info in reference.infolist()}
    assert index.height == len(FILES)
    for row in index.iter_rows(named=True):
        info = infos[row["name"]]
        assert row["crc32"] == info.CRC
        assert row["compressed_bytes"] == info.compress_size
        assert row["uncompressed_bytes"] == info.file_size
        assert row["offset"] == info.header_offset


def test_inflate_entry_roundtrip_and_crc_check() -> None:
    """Check that entries inflate to their original content and reject a wrong CRC32."""
    archive = build_archive(FILES)
    index = parse_central_directory(central_directory(archive))
    for row in index.iter_rows(named=True):
        chunk = archive[row["offset"] :]
        assert inflate_entry(chunk, row["compressed_bytes"], row["crc32"]) == FILES[row["name"]]
        with pytest.raises(ValueError):
            inflate_entry(chunk, row["compressed_bytes"], row["crc32"] ^ 1)


def test_segment_url_uses_zip_for_last_disk() -> None:
    """Check that segment URLs use the .zNN names and the final .zip name."""
    assert segment_url("abc", 0, 13).endswith("/ubuntu_images/images.z01")
    assert segment_url("abc", 12, 13).endswith("/ubuntu_images/images.z13")
    assert segment_url("abc", 13, 13).endswith("/ubuntu_images/images.zip")


def test_png_size_reads_ihdr(tmp_path: Path) -> None:
    """Check that the dimensions are read from the IHDR chunk."""
    ihdr = struct.pack(">II", 1920, 1080) + b"\x08\x02\x00\x00\x00"
    png = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr
    path = tmp_path / "shot.png"
    path.write_bytes(png)
    assert png_size(path) == (1920, 1080)
