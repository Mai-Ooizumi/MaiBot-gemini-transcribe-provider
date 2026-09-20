from __future__ import annotations

import base64
import inspect
import io
from typing import Any

from audio import NormalizedAudio


TRANSPORT_STRATEGIES = {"auto", "inline", "files"}


class _InlineUnsupportedError(RuntimeError):
    """Marks only the dedicated inline interaction as eligible for Files fallback."""


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def extract_interaction_text(interaction: Any) -> str:
    """Read text from current and older google-genai Interactions shapes."""
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

    for step in getattr(interaction, "steps", []) or []:
        for content in getattr(step, "content", []) or []:
            if getattr(content, "type", None) == "text":
                text = str(getattr(content, "text", "") or "").strip()
                if text:
                    chunks.append(text)
    return "\n".join(chunks).strip()


def interaction_diagnostic(interaction: Any) -> str:
    status = str(getattr(interaction, "status", "") or "unknown")
    interaction_id = str(getattr(interaction, "id", "") or "")
    output_types = [
        str(getattr(output, "type", type(output).__name__))
        for output in getattr(interaction, "outputs", []) or []
    ]
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


def _error_text(error: BaseException) -> str:
    values = [str(error), str(getattr(error, "message", "") or "")]
    cause = getattr(error, "__cause__", None)
    if cause is not None:
        values.append(str(cause))
    return " ".join(values).lower()


def is_inline_unsupported_error(error: BaseException) -> bool:
    """Classify only explicit inline/request-format failures as file fallbacks."""
    code = getattr(error, "code", None) or getattr(error, "status_code", 0)
    try:
        code = int(code or 0)
    except (TypeError, ValueError):
        code = 0
    if code not in {400, 413, 415, 422}:
        return False
    message = _error_text(error)
    if code == 413 and any(marker in message for marker in ("too large", "payload", "size", "limit", "exceed")):
        return True
    markers = (
        "unsupported",
        "not supported",
        "invalid argument",
        "invalid request",
        "invalid format",
        "malformed",
        "mime type",
        "mime_type",
        "inline data",
        "inline audio",
        "audio format",
        "audio data",
        "base64",
    )
    return any(marker in message for marker in markers)


def _inline_part(data: bytes, mime_type: str) -> Any:
    # `types.Part.from_bytes` is the supported GenerateContent representation;
    # the dict is a compatibility fallback for SDK test doubles/older builds.
    try:
        from google.genai import types

        return types.Part.from_bytes(data=data, mime_type=mime_type)
    except (ImportError, AttributeError):
        return {
            "inline_data": {
                "data": base64.b64encode(data).decode("ascii"),
                "mime_type": mime_type,
            }
        }


async def _fallback_transcription(
    *,
    client: Any,
    audio: NormalizedAudio,
    uploaded: Any,
    fallback_model: str,
    prompt: str,
) -> str:
    fallback_input = uploaded if uploaded is not None else _inline_part(audio.data, audio.mime_type)
    response = await _maybe_await(
        client.models.generate_content(
            model=fallback_model,
            contents=[prompt, fallback_input],
        )
    )
    return str(getattr(response, "text", "") or "").strip()


async def _transcribe_interaction(
    *,
    client: Any,
    audio: NormalizedAudio,
    input_content: dict[str, Any],
    uploaded: Any,
    model: str,
    transcription_config: dict[str, Any],
    fallback_on_empty: bool,
    fallback_model: str,
    fallback_prompt: str,
    allow_inline_fallback: bool = False,
) -> tuple[str, str | None]:
    kwargs: dict[str, Any] = {
        "model": model,
        "input": [input_content],
    }
    if transcription_config:
        kwargs["generation_config"] = {"transcription_config": transcription_config}
    try:
        interaction = await _maybe_await(client.interactions.create(**kwargs))
    except Exception as exc:
        if allow_inline_fallback and is_inline_unsupported_error(exc):
            raise _InlineUnsupportedError(str(exc)) from exc
        raise
    text = extract_interaction_text(interaction)
    if text:
        return text, None

    diagnostic = interaction_diagnostic(interaction)
    # A normal HTTP response with empty text deliberately does not trigger a
    # Files retry.  The existing Flash-Lite fallback remains the next step.
    if fallback_on_empty and fallback_model:
        fallback_text = await _fallback_transcription(
            client=client,
            audio=audio,
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


async def _transcribe_inline(
    *,
    client: Any,
    audio: NormalizedAudio,
    base64_data: str | None,
    model: str,
    transcription_config: dict[str, Any],
    fallback_on_empty: bool,
    fallback_model: str,
    fallback_prompt: str,
) -> tuple[str, str | None]:
    encoded = base64_data or base64.b64encode(audio.data).decode("ascii")
    return await _transcribe_interaction(
        client=client,
        audio=audio,
        input_content={
            "type": "audio",
            "data": encoded,
            "mime_type": audio.mime_type,
        },
        uploaded=None,
        model=model,
        transcription_config=transcription_config,
        fallback_on_empty=fallback_on_empty,
        fallback_model=fallback_model,
        fallback_prompt=fallback_prompt,
        allow_inline_fallback=True,
    )


async def _transcribe_files(
    *,
    client: Any,
    audio: NormalizedAudio,
    model: str,
    transcription_config: dict[str, Any],
    delete_uploaded_file: bool,
    fallback_on_empty: bool,
    fallback_model: str,
    fallback_prompt: str,
) -> tuple[str, str | None]:
    upload_stream = io.BytesIO(audio.data)
    upload_stream.name = f"input{audio.suffix or '.bin'}"
    uploaded = None
    try:
        uploaded = await _maybe_await(
            client.files.upload(
                file=upload_stream,
                config={"mime_type": audio.mime_type},
            )
        )
        uploaded_mime = str(getattr(uploaded, "mime_type", "") or audio.mime_type)
        uploaded_uri = str(getattr(uploaded, "uri", "") or "")
        if not uploaded_uri:
            raise RuntimeError("Google Files API 上传成功但没有返回 file URI")
        return await _transcribe_interaction(
            client=client,
            audio=audio,
            input_content={
                "type": "audio",
                "uri": uploaded_uri,
                "mime_type": uploaded_mime,
            },
            uploaded=uploaded,
            model=model,
            transcription_config=transcription_config,
            fallback_on_empty=fallback_on_empty,
            fallback_model=fallback_model,
            fallback_prompt=fallback_prompt,
        )
    finally:
        if delete_uploaded_file and uploaded is not None:
            name = str(getattr(uploaded, "name", "") or "")
            if name:
                try:
                    await _maybe_await(client.files.delete(name=name))
                except Exception:
                    # Cleanup failure must not discard a successful transcription.
                    pass


async def transcribe(
    *,
    client: Any,
    audio: NormalizedAudio,
    base64_data: str | None,
    strategy: str,
    inline_max_bytes: int,
    model: str,
    transcription_config: dict[str, Any],
    delete_uploaded_file: bool,
    fallback_on_empty: bool,
    fallback_model: str,
    fallback_prompt: str,
) -> tuple[str, str | None]:
    strategy = strategy.strip().lower()
    if strategy not in TRANSPORT_STRATEGIES:
        raise ValueError("gemini.audio_transport 只能是 'auto'、'inline' 或 'files'")
    if inline_max_bytes <= 0:
        raise ValueError("gemini.inline_max_bytes 必须大于 0")

    use_inline = strategy == "inline" or (strategy == "auto" and len(audio.data) <= inline_max_bytes)
    if use_inline:
        try:
            return await _transcribe_inline(
                client=client,
                audio=audio,
                base64_data=base64_data,
                model=model,
                transcription_config=transcription_config,
                fallback_on_empty=fallback_on_empty,
                fallback_model=fallback_model,
                fallback_prompt=fallback_prompt,
            )
        except _InlineUnsupportedError:
            return await _transcribe_files(
                client=client,
                audio=audio,
                model=model,
                transcription_config=transcription_config,
                delete_uploaded_file=delete_uploaded_file,
                fallback_on_empty=fallback_on_empty,
                fallback_model=fallback_model,
                fallback_prompt=fallback_prompt,
            )

    return await _transcribe_files(
        client=client,
        audio=audio,
        model=model,
        transcription_config=transcription_config,
        delete_uploaded_file=delete_uploaded_file,
        fallback_on_empty=fallback_on_empty,
        fallback_model=fallback_model,
        fallback_prompt=fallback_prompt,
    )
