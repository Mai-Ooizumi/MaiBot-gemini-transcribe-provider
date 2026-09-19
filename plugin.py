from __future__ import annotations

import asyncio
import base64
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from google import genai
from maibot_sdk import LLMProvider, MaiBotPlugin


CLIENT_TYPE = "gemini35.transcribe"
DEFAULT_MODEL = "gemini-3.5-transcribe"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def _detect_audio(data: bytes) -> tuple[str | None, str | None]:
    """Return (mime_type, suffix). None means Google Transcribe may not accept it directly."""
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav", ".wav"
    if data.startswith(b"OggS"):
        return "audio/ogg", ".ogg"
    if data.startswith(b"fLaC"):
        return "audio/flac", ".flac"
    if data.startswith(b"ID3"):
        return "audio/mpeg", ".mp3"
    # AAC ADTS also starts with 0xFFF, so detect it before generic MPEG audio frames.
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xF6) == 0xF0:
        return "audio/aac", ".aac"
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return "audio/mpeg", ".mp3"
    if data.startswith(b"FORM") and len(data) >= 12 and data[8:12] in {b"AIFF", b"AIFC"}:
        return "audio/aiff", ".aiff"
    if data.startswith(b"\x1aE\xdf\xa3"):
        return "audio/webm", ".webm"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        # M4A/MP4-family audio container. Use the MIME accepted by Gemini Transcribe.
        return "audio/m4a", ".m4a"
    if data.startswith(b"#!SILK_V3") or data.startswith(b"\x02#!SILK_V3"):
        return None, ".silk"
    if data.startswith(b"#!AMR\n") or data.startswith(b"#!AMR-WB\n"):
        return None, ".amr"
    return None, ".bin"


def _convert_to_wav(src: Path, dst: Path, timeout: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "收到的语音不是 Gemini 3.5 Transcribe 可直接接受的格式，而且系统没有找到 ffmpeg。"
            "请安装 ffmpeg，或让上游 Adapter 输出 WAV/MP3/OGG/Opus/WebM。"
        )
    proc = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=max(10.0, timeout),
        check=False,
    )
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        if len(detail) > 1200:
            detail = detail[-1200:]
        raise RuntimeError(
            "ffmpeg 无法把语音转换为 WAV。若这是 QQ/Tencent SILK，普通 ffmpeg 构建可能不带 SILK 解码器；"
            "建议让 NapCat/Adapter 先转成 WAV/OGG/MP3。"
            + (f" ffmpeg: {detail}" if detail else "")
        )


def _extract_api_key(request: dict[str, Any], settings: dict[str, Any]) -> str:
    provider = _as_dict(request.get("api_provider"))
    provider_key = str(provider.get("api_key") or "").strip()
    if provider_key and provider_key.lower() not in {"none", "unused", "placeholder"}:
        return provider_key

    gemini_cfg = _as_dict(settings.get("gemini"))
    config_key = str(gemini_cfg.get("api_key") or "").strip()
    if config_key:
        return config_key

    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()


def _extract_model(request: dict[str, Any], settings: dict[str, Any]) -> str:
    model_info = _as_dict(request.get("model_info"))
    configured = str(model_info.get("model_identifier") or "").strip()
    if configured:
        return configured
    gemini_cfg = _as_dict(settings.get("gemini"))
    return str(gemini_cfg.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _build_transcription_config(settings: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    gemini_cfg = _as_dict(settings.get("gemini"))
    extra = _as_dict(request.get("extra_params"))

    language_codes = _as_str_list(extra.get("language_codes", gemini_cfg.get("language_codes", [])))
    custom_vocabulary = _as_str_list(extra.get("custom_vocabulary", gemini_cfg.get("custom_vocabulary", [])))
    mode = str(extra.get("transcription_mode", gemini_cfg.get("mode", "verbatim"))).strip().lower() or "verbatim"
    diarization = bool(extra.get("speaker_diarization", gemini_cfg.get("speaker_diarization", False)))
    word_timestamps = bool(extra.get("word_timestamps", gemini_cfg.get("word_timestamps", False)))

    if mode not in {"verbatim", "smart"}:
        raise ValueError("gemini.mode / transcription_mode 只能是 'verbatim' 或 'smart'")
    if mode == "smart" and (diarization or word_timestamps):
        raise ValueError("Gemini 3.5 Transcribe 的 smart 模式不能与 speaker diarization / word timestamps 同时使用")
    if custom_vocabulary and (diarization or word_timestamps):
        raise ValueError("Gemini 3.5 Transcribe 的 custom_vocabulary 不能与 speaker diarization / word timestamps 同时使用")

    config: dict[str, Any] = {}
    if language_codes:
        config["language_codes"] = language_codes
    if custom_vocabulary:
        config["custom_vocabulary"] = custom_vocabulary[:1000]

    if mode == "smart":
        config["mode"] = "smart"
    elif diarization or word_timestamps:
        mode_cfg: dict[str, Any] = {"type": "verbatim"}
        if diarization:
            mode_cfg["diarization_mode"] = "speaker"
        if word_timestamps:
            mode_cfg["timestamp_granularities"] = ["word"]
        config["mode"] = mode_cfg
    # For plain verbatim, omit mode: it is the API default.
    return config


def _extract_interaction_text(interaction: Any) -> str:
    """兼容 google-genai 不同版本的 Interactions 文本输出结构。"""
    direct = str(getattr(interaction, "output_text", "") or "").strip()
    if direct:
        return direct

    chunks: list[str] = []
    for output in getattr(interaction, "outputs", []) or []:
        if getattr(output, "type", None) == "text":
            text = str(getattr(output, "text", "") or "").strip()
            if text:
                chunks.append(text)
    if chunks:
        return "\n".join(chunks).strip()

    # 某些 SDK / API 版本把 model output 放在 steps[].content[] 中。
    for step in getattr(interaction, "steps", []) or []:
        for content in getattr(step, "content", []) or []:
            if getattr(content, "type", None) == "text":
                text = str(getattr(content, "text", "") or "").strip()
                if text:
                    chunks.append(text)
    return "\n".join(chunks).strip()


def _interaction_diagnostic(interaction: Any) -> str:
    """只输出无敏感信息的响应摘要，便于定位 Google 的静默空结果。"""
    status = str(getattr(interaction, "status", "") or "unknown")
    interaction_id = str(getattr(interaction, "id", "") or "")

    output_types: list[str] = []
    for output in getattr(interaction, "outputs", []) or []:
        output_types.append(str(getattr(output, "type", type(output).__name__)))

    usage = getattr(interaction, "usage", None)
    total_input = getattr(usage, "total_input_tokens", None) if usage is not None else None
    total_output = getattr(usage, "total_output_tokens", None) if usage is not None else None

    parts = [f"status={status}"]
    if interaction_id:
        parts.append(f"id={interaction_id}")
    if total_input is not None:
        parts.append(f"input_tokens={total_input}")
    if total_output is not None:
        parts.append(f"output_tokens={total_output}")
    parts.append(f"output_types={output_types or []}")
    return ", ".join(parts)


def _fallback_transcription(
    *,
    client: Any,
    uploaded: Any,
    fallback_model: str,
    prompt: str,
) -> str:
    response = client.models.generate_content(
        model=fallback_model,
        contents=[prompt, uploaded],
    )
    return str(getattr(response, "text", "") or "").strip()


def _run_transcription(
    *,
    api_key: str,
    model: str,
    audio_path: Path,
    mime_type: str,
    transcription_config: dict[str, Any],
    delete_uploaded_file: bool,
    fallback_on_empty: bool,
    fallback_model: str,
    fallback_prompt: str,
) -> tuple[str, str | None]:
    client = genai.Client(api_key=api_key)
    uploaded = None
    try:
        uploaded = client.files.upload(file=str(audio_path))
        uploaded_mime = str(getattr(uploaded, "mime_type", "") or mime_type)
        uploaded_uri = str(getattr(uploaded, "uri", "") or "")
        if not uploaded_uri:
            raise RuntimeError("Google Files API 上传成功但没有返回 file URI")

        kwargs: dict[str, Any] = {
            "model": model,
            "input": [
                {
                    "type": "audio",
                    "uri": uploaded_uri,
                    "mime_type": uploaded_mime,
                }
            ],
        }
        if transcription_config:
            kwargs["generation_config"] = {"transcription_config": transcription_config}

        interaction = client.interactions.create(**kwargs)
        text = _extract_interaction_text(interaction)
        if text:
            return text, None

        diagnostic = _interaction_diagnostic(interaction)
        if fallback_on_empty and fallback_model:
            fallback_text = _fallback_transcription(
                client=client,
                uploaded=uploaded,
                fallback_model=fallback_model,
                prompt=fallback_prompt,
            )
            if fallback_text:
                return fallback_text, diagnostic
            raise RuntimeError(
                "Gemini 3.5 Transcribe 返回空结果，且 fallback 也为空。"
                f" Dedicated response: {diagnostic}; fallback_model={fallback_model}"
            )

        raise RuntimeError(
            "Gemini 3.5 Transcribe 返回了空转写结果。"
            f" Google response summary: {diagnostic}"
        )
    finally:
        if delete_uploaded_file and uploaded is not None:
            name = str(getattr(uploaded, "name", "") or "")
            if name:
                try:
                    client.files.delete(name=name)
                except Exception:
                    # Cleanup failure must not discard a successful transcription.
                    pass
        try:
            client.close()
        except Exception:
            pass


class Gemini35TranscribePlugin(MaiBotPlugin):
    def __init__(self) -> None:
        super().__init__()
        self._settings: dict[str, Any] = {}

    async def on_load(self) -> None:
        # Runner 在 on_load() 前已注入插件自身 config.toml；直接读取本地配置快照，
        # 不走 config.get_all capability RPC，避免无权限令牌时报 E_CAPABILITY_DENIED。
        self._settings = self.get_plugin_config_data()
        self.ctx.logger.info("Gemini 3.5 Transcribe Provider 已加载 (%s)", CLIENT_TYPE)

    async def on_unload(self) -> None:
        return None

    async def on_config_update(self, scope: str, config_data: dict[str, object], version: str) -> None:
        del version
        if scope == "self":
            self._settings = dict(config_data)

    @LLMProvider(
        CLIENT_TYPE,
        name="Gemini 3.5 Transcribe",
        description="通过 Google Files API + Interactions API 调用 gemini-3.5-transcribe",
    )
    async def handle_llm_provider(self, operation: str, request: dict[str, Any]) -> dict[str, Any]:
        if operation != "audio_transcription":
            raise ValueError(f"{CLIENT_TYPE} 只支持 audio_transcription，收到: {operation}")

        audio_base64 = str(request.get("audio_base64") or "").strip()
        if not audio_base64:
            raise ValueError("MaiBot 没有向 Provider 传入 audio_base64")
        try:
            audio_bytes = base64.b64decode(audio_base64, validate=True)
        except Exception as exc:
            raise ValueError("audio_base64 不是有效的 Base64 音频") from exc
        if not audio_bytes:
            raise ValueError("收到的音频为空")

        api_key = _extract_api_key(request, self._settings)
        if not api_key:
            raise RuntimeError(
                "未找到 Gemini API Key。请在 model_config.toml 的对应 api_provider.api_key、"
                "插件 config.toml 的 [gemini].api_key，或 GEMINI_API_KEY 环境变量中配置。"
            )

        model = _extract_model(request, self._settings)
        if model == "gemini-3.5-transcribe-live":
            raise ValueError("MaiBot 的 voice 是录音文件 ASR，请使用 gemini-3.5-transcribe，而不是 -live 模型")

        gemini_cfg = _as_dict(self._settings.get("gemini"))
        timeout = float(gemini_cfg.get("ffmpeg_timeout", 45.0) or 45.0)
        delete_uploaded_file = bool(gemini_cfg.get("delete_uploaded_file", True))
        fallback_on_empty = bool(gemini_cfg.get("fallback_on_empty", True))
        fallback_model = str(gemini_cfg.get("fallback_model", "gemini-3.5-flash-lite") or "").strip()
        fallback_prompt = str(
            gemini_cfg.get(
                "fallback_prompt",
                "Generate an accurate transcript of the speech. Return only the transcript, with no commentary.",
            )
            or ""
        ).strip()
        transcription_config = _build_transcription_config(self._settings, request)

        mime_type, suffix = _detect_audio(audio_bytes)
        with tempfile.TemporaryDirectory(prefix="maibot-gemini35-asr-") as tmpdir:
            tmp = Path(tmpdir)
            source_path = tmp / f"input{suffix or '.bin'}"
            source_path.write_bytes(audio_bytes)

            upload_path = source_path
            upload_mime = mime_type
            if upload_mime is None:
                wav_path = tmp / "converted.wav"
                await asyncio.to_thread(_convert_to_wav, source_path, wav_path, timeout)
                upload_path = wav_path
                upload_mime = "audio/wav"

            text, dedicated_diagnostic = await asyncio.to_thread(
                _run_transcription,
                api_key=api_key,
                model=model,
                audio_path=upload_path,
                mime_type=upload_mime,
                transcription_config=transcription_config,
                delete_uploaded_file=delete_uploaded_file,
                fallback_on_empty=fallback_on_empty,
                fallback_model=fallback_model,
                fallback_prompt=fallback_prompt,
            )

        if dedicated_diagnostic is not None:
            self.ctx.logger.warning(
                "gemini-3.5-transcribe 静默返回空结果，已使用 %s fallback 成功；Google 响应摘要: %s",
                fallback_model,
                dedicated_diagnostic,
            )
        return {"content": text}


def create_plugin() -> Gemini35TranscribePlugin:
    return Gemini35TranscribePlugin()
