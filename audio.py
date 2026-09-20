from __future__ import annotations

import importlib
import io
import math
import shutil
import struct
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SILENCE_RMS = 0.002
DEFAULT_SILENCE_PEAK = 0.008
DEFAULT_SILENCE_MIN_SAMPLES = 160


@dataclass(frozen=True, slots=True)
class AudioFormat:
    mime_type: str | None
    suffix: str
    is_silk: bool = False


@dataclass(frozen=True, slots=True)
class NormalizedAudio:
    data: bytes
    mime_type: str
    suffix: str
    # When the bytes are unchanged, retain MaiBot's already decoded base64 so
    # the inline path does not perform an unnecessary encode/decode roundtrip.
    original_base64: str | None = None


def detect_audio(data: bytes) -> AudioFormat:
    """Detect formats Gemini can receive directly and common QQ inputs."""
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return AudioFormat("audio/wav", ".wav")
    if data.startswith(b"OggS"):
        return AudioFormat("audio/ogg", ".ogg")
    if data.startswith(b"fLaC"):
        return AudioFormat("audio/flac", ".flac")
    if data.startswith(b"ID3"):
        return AudioFormat("audio/mpeg", ".mp3")
    # AAC ADTS also starts with 0xFFF, so detect it before generic MPEG frames.
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xF6) == 0xF0:
        return AudioFormat("audio/aac", ".aac")
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return AudioFormat("audio/mpeg", ".mp3")
    if data.startswith(b"FORM") and len(data) >= 12 and data[8:12] in {b"AIFF", b"AIFC"}:
        return AudioFormat("audio/aiff", ".aiff")
    if data.startswith(b"\x1aE\xdf\xa3"):
        return AudioFormat("audio/webm", ".webm")
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return AudioFormat("audio/m4a", ".m4a")
    if data.startswith(b"#!SILK_V3") or data.startswith(b"\x02#!SILK_V3"):
        return AudioFormat(None, ".silk", is_silk=True)
    if data.startswith(b"#!AMR\n") or data.startswith(b"#!AMR-WB\n"):
        return AudioFormat(None, ".amr")
    return AudioFormat(None, ".bin")


def _load_silk_decoder() -> Any | None:
    """Load the dedicated decoder lazily so non-SILK audio stays dependency-free."""
    for module_name in ("pilk_nogil", "pilk"):
        try:
            return importlib.import_module(module_name)
        except Exception:
            continue
    return None


def _pcm_to_wav(pcm: bytes, *, sample_rate: int = 24000) -> bytes:
    if not pcm:
        raise RuntimeError("SILK decoder 返回了空 PCM")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return output.getvalue()


def _decode_silk_with_decoder(data: bytes, decoder: Any) -> bytes:
    # pilk-nogil >= 0.4 exposes a memory-safe silk_to_wav helper.  It also
    # understands Tencent's leading 0x02 marker used by QQ/WeChat.
    silk_to_wav = getattr(decoder, "silk_to_wav", None)
    if callable(silk_to_wav):
        try:
            wav_data = silk_to_wav(data, None, rate=24000)
        except TypeError:
            wav_data = silk_to_wav(data, None)
        if isinstance(wav_data, (bytes, bytearray)) and wav_data:
            return bytes(wav_data)

    decode = getattr(decoder, "decode", None)
    if not callable(decode):
        raise RuntimeError("SILK decoder 没有 decode/silk_to_wav 接口")

    # Newer bindings accept bytes and return PCM directly.
    for kwargs in ({"pcm_rate": 24000}, {}):
        try:
            pcm_data = decode(data, None, **kwargs)
        except Exception:
            continue
        if isinstance(pcm_data, (bytes, bytearray)) and pcm_data:
            return _pcm_to_wav(bytes(pcm_data))

    # Older pilk releases only accept paths.  Keep this compatibility fallback
    # local to SILK; ordinary Gemini-compatible audio never touches disk.
    with tempfile.TemporaryDirectory(prefix="maibot-silk-") as tmpdir:
        source_path = Path(tmpdir) / "input.silk"
        pcm_path = Path(tmpdir) / "output.pcm"
        source_path.write_bytes(data)
        try:
            decode(str(source_path), str(pcm_path), pcm_rate=24000)
        except TypeError:
            decode(str(source_path), str(pcm_path))
        pcm = pcm_path.read_bytes() if pcm_path.exists() else b""
    return _pcm_to_wav(pcm)


def _convert_with_ffmpeg(data: bytes, suffix: str, timeout: float) -> bytes:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "收到的语音不是 Gemini 可直接接受的格式，而且系统没有找到 ffmpeg。"
            "请安装 ffmpeg，或让上游 Adapter 输出 WAV/MP3/OGG/WebM。"
        )

    # Pipe conversion avoids a temporary input/output pair for the common
    # fallback path.  Some old ffmpeg builds cannot sniff SILK/AMR from stdin;
    # retain a path-based retry for those builds and for backward compatibility.
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        "pipe:1",
    ]
    proc = subprocess.run(
        command,
        input=data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=max(10.0, timeout),
        check=False,
    )
    if proc.returncode == 0 and proc.stdout:
        return proc.stdout

    with tempfile.TemporaryDirectory(prefix="maibot-ffmpeg-") as tmpdir:
        source_path = Path(tmpdir) / f"input{suffix or '.bin'}"
        output_path = Path(tmpdir) / "converted.wav"
        source_path.write_bytes(data)
        proc = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=max(10.0, timeout),
            check=False,
        )
        converted = output_path.read_bytes() if output_path.exists() else b""

    if proc.returncode != 0 or not converted:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        if len(detail) > 1200:
            detail = detail[-1200:]
        raise RuntimeError(
            "ffmpeg 无法把语音转换为 WAV。若这是 QQ/Tencent SILK，普通 ffmpeg 构建可能不带 SILK 解码器；"
            "建议安装可选的 pilk-nogil，或让 NapCat/Adapter 先转成 WAV/OGG/MP3。"
            + (f" ffmpeg: {detail}" if detail else "")
        )
    return converted


def normalize_audio(
    data: bytes,
    *,
    timeout: float,
    original_base64: str | None = None,
) -> NormalizedAudio:
    """Return audio in a Gemini-compatible form, preferring a dedicated SILK decoder."""
    detected = detect_audio(data)
    if detected.mime_type:
        return NormalizedAudio(
            data=data,
            mime_type=detected.mime_type,
            suffix=detected.suffix,
            original_base64=original_base64,
        )

    converted: bytes | None = None
    if detected.is_silk:
        decoder = _load_silk_decoder()
        if decoder is not None:
            try:
                converted = _decode_silk_with_decoder(data, decoder)
            except Exception:
                # A malformed/variant Tencent sample should still get the
                # existing FFmpeg fallback rather than failing early.
                converted = None
    if converted is None:
        converted = _convert_with_ffmpeg(data, detected.suffix, timeout)
    return NormalizedAudio(data=converted, mime_type="audio/wav", suffix=".wav")


def _pcm_samples(raw: bytes, sample_width: int, channels: int) -> list[float]:
    if sample_width not in {1, 2, 3, 4} or channels <= 0:
        return []
    frame_width = sample_width * channels
    if frame_width <= 0:
        return []
    sample_count = len(raw) // frame_width
    # A bounded sample set keeps the gate inexpensive for unexpectedly long
    # WAVs while preserving a representative spread across the recording.
    target_samples = 200_000
    frame_stride = max(1, math.ceil(sample_count / target_samples))
    values: list[float] = []
    for frame_index in range(0, sample_count, frame_stride):
        frame = raw[frame_index * frame_width : (frame_index + 1) * frame_width]
        channel_values: list[float] = []
        for channel in range(channels):
            start = channel * sample_width
            chunk = frame[start : start + sample_width]
            if len(chunk) != sample_width:
                continue
            if sample_width == 1:
                value = (chunk[0] - 128) / 128.0
            elif sample_width == 2:
                value = struct.unpack_from("<h", chunk)[0] / 32768.0
            elif sample_width == 3:
                integer = int.from_bytes(chunk, "little", signed=False)
                if integer & 0x800000:
                    integer -= 1 << 24
                value = integer / 8_388_608.0
            else:
                value = struct.unpack_from("<i", chunk)[0] / 2_147_483_648.0
            channel_values.append(value)
        if channel_values:
            # Taking the loudest channel avoids a false silence when stereo
            # channels cancel each other during averaging.
            values.append(max(channel_values, key=abs))
    return values


def is_clearly_silent(
    data: bytes,
    mime_type: str,
    *,
    rms_threshold: float = DEFAULT_SILENCE_RMS,
    peak_threshold: float = DEFAULT_SILENCE_PEAK,
    min_samples: int = DEFAULT_SILENCE_MIN_SAMPLES,
) -> bool | None:
    """Return True only for high-confidence PCM silence, False/None otherwise.

    This is intentionally an energy gate, not a noise detector or VAD.  Loud
    white noise therefore remains ``False`` and is sent to Gemini; compressed
    formats that would require a forced transcode return ``None``.
    """
    if mime_type != "audio/wav":
        return None
    try:
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            if wav_file.getcomptype() != "NONE":
                return None
            frame_count = wav_file.getnframes()
            raw = wav_file.readframes(frame_count)
    except (EOFError, OSError, wave.Error):
        return None
    if frame_count <= 0 or not raw:
        return True
    values = _pcm_samples(raw, sample_width, channels)
    if not values:
        return None
    peak = max(abs(value) for value in values)
    if len(values) < min_samples:
        return True if peak <= peak_threshold / 2 else None
    rms = math.sqrt(sum(value * value for value in values) / len(values))
    if rms <= rms_threshold and peak <= peak_threshold:
        return True
    return False
