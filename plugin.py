from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path
from typing import Any

from maibot_sdk import LLMProvider, MaiBotPlugin

from .audio import NormalizedAudio, detect_audio, is_clearly_silent, normalize_audio
from .gemini_client import GeminiClientPool
from .gemini_transport import extract_interaction_text, interaction_diagnostic, transcribe


CLIENT_TYPE = "gemini35.transcribe"
DEFAULT_MODEL = "gemini-3.5-transcribe"
DEFAULT_INLINE_MAX_BYTES = 256 * 1024


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


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    if value is None:
        return default
    return bool(value)


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
    diarization = _as_bool(extra.get("speaker_diarization", gemini_cfg.get("speaker_diarization", False)), False)
    word_timestamps = _as_bool(extra.get("word_timestamps", gemini_cfg.get("word_timestamps", False)), False)

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
    return config


def _transport_settings(settings: dict[str, Any], request: dict[str, Any]) -> tuple[str, int, bool]:
    gemini_cfg = _as_dict(settings.get("gemini"))
    extra = _as_dict(request.get("extra_params"))
    strategy = str(
        extra.get(
            "audio_transport",
            extra.get("transport", gemini_cfg.get("audio_transport", gemini_cfg.get("transport", "auto"))),
        )
        or "auto"
    ).strip().lower()
    try:
        inline_max_bytes = int(
            extra.get("inline_max_bytes", gemini_cfg.get("inline_max_bytes", DEFAULT_INLINE_MAX_BYTES))
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("gemini.inline_max_bytes 必须是整数") from exc
    local_speech_gate = _as_bool(extra.get("local_speech_gate", gemini_cfg.get("local_speech_gate", True)), True)
    return strategy, inline_max_bytes, local_speech_gate


class Gemini35TranscribePlugin(MaiBotPlugin):
    def __init__(self) -> None:
        super().__init__()
        self._settings: dict[str, Any] = {}
        self._client_pool = GeminiClientPool()

    async def on_load(self) -> None:
        self._settings = self.get_plugin_config_data()
        self.ctx.logger.info("Gemini 3.5 Transcribe Provider 已加载 (%s)", CLIENT_TYPE)

    async def on_unload(self) -> None:
        await self._client_pool.close()

    async def on_config_update(self, scope: str, config_data: dict[str, object], version: str) -> None:
        del version
        if scope == "self":
            self._settings = dict(config_data)
            await self._client_pool.invalidate()

    @LLMProvider(
        CLIENT_TYPE,
        name="Gemini 3.5 Transcribe",
        description="通过 Interactions API 的 inline/files 音频输入调用 gemini-3.5-transcribe",
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

        gemini_cfg = _as_dict(self._settings.get("gemini"))
        try:
            timeout = float(gemini_cfg.get("ffmpeg_timeout", 45.0) or 45.0)
        except (TypeError, ValueError) as exc:
            raise ValueError("gemini.ffmpeg_timeout 必须是数字") from exc
        strategy, inline_max_bytes, local_speech_gate = _transport_settings(self._settings, request)

        detected = detect_audio(audio_bytes)
        if detected.mime_type:
            audio = NormalizedAudio(
                data=audio_bytes,
                mime_type=detected.mime_type,
                suffix=detected.suffix,
                original_base64=audio_base64,
            )
        else:
            # Decoder/transcoder work is local and blocking; API calls below use
            # the SDK's native async client and never occupy this worker thread.
            audio = await asyncio.to_thread(
                normalize_audio,
                audio_bytes,
                timeout=timeout,
                original_base64=audio_base64,
            )

        if local_speech_gate and is_clearly_silent(audio.data, audio.mime_type) is True:
            self.ctx.logger.debug("本地 speech gate 过滤了明显静音音频")
            return {
                "content": "",
                "raw_data": {"status": "no_speech", "reason": "local_speech_gate"},
            }

        api_key = _extract_api_key(request, self._settings)
        if not api_key:
            raise RuntimeError(
                "未找到 Gemini API Key。请在 model_config.toml 的对应 api_provider.api_key、"
                "插件 config.toml 的 [gemini].api_key，或 GEMINI_API_KEY 环境变量中配置。"
            )

        model = _extract_model(request, self._settings)
        if model == "gemini-3.5-transcribe-live":
            raise ValueError("MaiBot 的 voice 是录音文件 ASR，请使用 gemini-3.5-transcribe，而不是 -live 模型")

        delete_uploaded_file = _as_bool(gemini_cfg.get("delete_uploaded_file", True), True)
        fallback_on_empty = _as_bool(gemini_cfg.get("fallback_on_empty", True), True)
        fallback_model = str(gemini_cfg.get("fallback_model", "gemini-3.5-flash-lite") or "").strip()
        fallback_prompt = str(
            gemini_cfg.get(
                "fallback_prompt",
                "Generate an accurate transcript of the speech. Return only the transcript, with no commentary.",
            )
            or ""
        ).strip()
        transcription_config = _build_transcription_config(self._settings, request)
        provider = _as_dict(request.get("api_provider"))
        base_url = str(provider.get("base_url") or "").strip()

        lease = await self._client_pool.acquire(api_key, base_url=base_url)
        try:
            text, dedicated_diagnostic = await transcribe(
                client=lease.client,
                audio=audio,
                base64_data=audio.original_base64,
                strategy=strategy,
                inline_max_bytes=inline_max_bytes,
                model=model,
                transcription_config=transcription_config,
                delete_uploaded_file=delete_uploaded_file,
                fallback_on_empty=fallback_on_empty,
                fallback_model=fallback_model,
                fallback_prompt=fallback_prompt,
            )
        finally:
            await lease.release()

        if dedicated_diagnostic is not None:
            self.ctx.logger.warning(
                "gemini-3.5-transcribe 静默返回空结果，已使用 %s fallback 成功；Google 响应摘要: %s",
                fallback_model,
                dedicated_diagnostic,
            )
        return {"content": text}


def create_plugin() -> Gemini35TranscribePlugin:
    return Gemini35TranscribePlugin()


# Compatibility wrapper for integrations/tests that imported the former helper.
def _detect_audio(data: bytes) -> tuple[str | None, str | None]:
    detected = detect_audio(data)
    return detected.mime_type, detected.suffix


def _convert_to_wav(src: Path, dst: Path, timeout: float) -> None:
    """Compatibility wrapper for callers of the former path-based helper."""
    from .audio import _convert_with_ffmpeg

    dst.write_bytes(_convert_with_ffmpeg(src.read_bytes(), src.suffix, timeout))


_extract_interaction_text = extract_interaction_text
_interaction_diagnostic = interaction_diagnostic
