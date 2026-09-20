from __future__ import annotations

import asyncio
import base64
import io
import random
import sys
import types
import unittest
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


if "maibot_sdk" not in sys.modules:
    sdk = types.ModuleType("maibot_sdk")

    class FakePlugin:
        def __init__(self) -> None:
            self.ctx = types.SimpleNamespace(logger=types.SimpleNamespace(
                info=lambda *args, **kwargs: None,
                warning=lambda *args, **kwargs: None,
                debug=lambda *args, **kwargs: None,
            ))

        def get_plugin_config_data(self):
            return {}

    def provider_decorator(*args, **kwargs):
        del args, kwargs
        return lambda function: function

    sdk.MaiBotPlugin = FakePlugin
    sdk.LLMProvider = provider_decorator
    sys.modules["maibot_sdk"] = sdk


import audio
import gemini_client
import plugin


def make_wav(amplitude: int, *, frames: int = 3200, sample_rate: int = 16000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        positive = int(amplitude).to_bytes(2, "little", signed=True)
        negative = (-int(amplitude)).to_bytes(2, "little", signed=True)
        wav_file.writeframes((positive + negative) * (frames // 2) + positive * (frames % 2))
    return output.getvalue()


class FakeError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class FakeInteraction:
    def __init__(self, text: str = "") -> None:
        self.output_text = text
        self.status = "completed"
        self.id = "interaction-test"
        self.outputs = []


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeUploaded:
    name = "files/test-audio"
    uri = "https://example.test/files/test-audio"
    mime_type = "audio/ogg"


class FakeService:
    def __init__(self) -> None:
        self.interaction_results: list[object] = []
        self.interaction_calls: list[dict] = []
        self.upload_calls: list[dict] = []
        self.delete_calls: list[str] = []
        self.generate_calls: list[dict] = []
        self.fallback_text = "fallback text"

    async def interaction_create(self, **kwargs):
        self.interaction_calls.append(kwargs)
        outcome = self.interaction_results.pop(0) if self.interaction_results else FakeInteraction("transcribed")
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def files_upload(self, *, file, config=None):
        self.upload_calls.append({"file": file, "config": config})
        return FakeUploaded()

    async def files_delete(self, *, name):
        self.delete_calls.append(name)

    async def generate_content(self, **kwargs):
        self.generate_calls.append(kwargs)
        return FakeResponse(self.fallback_text)


class FakeAsyncClient:
    def __init__(self, service: FakeService) -> None:
        self.service = service
        self.interactions = types.SimpleNamespace(create=service.interaction_create)
        self.files = types.SimpleNamespace(upload=service.files_upload, delete=service.files_delete)
        self.models = types.SimpleNamespace(generate_content=service.generate_content)
        self.aclose_calls = 0

    async def aclose(self):
        self.aclose_calls += 1


class FakeSyncClient:
    def __init__(self, async_client: FakeAsyncClient) -> None:
        self.aio = async_client
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class FakeFactory:
    def __init__(self) -> None:
        self.services: list[FakeService] = []
        self.sync_clients: list[FakeSyncClient] = []

    def __call__(self, key: gemini_client.ClientKey):
        del key
        service = FakeService()
        async_client = FakeAsyncClient(service)
        sync_client = FakeSyncClient(async_client)
        self.services.append(service)
        self.sync_clients.append(sync_client)
        return sync_client, async_client


def request_for(data: bytes, *, key: str = "test-key", extra: dict | None = None) -> dict:
    return {
        "audio_base64": base64.b64encode(data).decode("ascii"),
        "api_provider": {"api_key": key},
        "model_info": {"model_identifier": "gemini-3.5-transcribe"},
        "extra_params": extra or {},
    }


class PluginTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plugin = plugin.Gemini35TranscribePlugin()
        self.factory = FakeFactory()
        self.plugin._client_pool = gemini_client.GeminiClientPool(factory=self.factory)
        self.plugin._settings = {
            "gemini": {
                "api_key": "",
                "audio_transport": "auto",
                "inline_max_bytes": 262144,
                "local_speech_gate": True,
                "fallback_on_empty": True,
                "delete_uploaded_file": True,
            }
        }

    def run_request(self, data: bytes, **kwargs):
        return asyncio.run(self.plugin.handle_llm_provider("audio_transcription", request_for(data, **kwargs)))

    def test_small_audio_uses_inline_without_files(self) -> None:
        data = b"OggS" + b"speech" * 20
        result = self.run_request(data)
        self.assertEqual(result["content"], "transcribed")
        service = self.factory.services[0]
        self.assertEqual(service.upload_calls, [])
        content = service.interaction_calls[0]["input"][0]
        self.assertIn("data", content)
        self.assertNotIn("uri", content)
        self.assertEqual(content["data"], base64.b64encode(data).decode("ascii"))

    def test_large_audio_uses_files_and_deletes(self) -> None:
        data = b"OggS" + b"x" * 1000
        result = self.run_request(data, extra={"inline_max_bytes": 64})
        self.assertEqual(result["content"], "transcribed")
        service = self.factory.services[0]
        self.assertEqual(len(service.upload_calls), 1)
        self.assertEqual(service.delete_calls, ["files/test-audio"])
        self.assertIn("uri", service.interaction_calls[0]["input"][0])

    def test_inline_unsupported_falls_back_to_files(self) -> None:
        service_result = FakeError(400, "inline audio data is not supported")
        self.plugin._settings["gemini"]["audio_transport"] = "inline"
        result_future = None

        async def run():
            lease = await self.plugin._client_pool.acquire("test-key")
            service = self.factory.services[0]
            service.interaction_results = [service_result, FakeInteraction("from files")]
            await lease.release()
            return await self.plugin.handle_llm_provider(
                "audio_transcription", request_for(b"OggS" + b"x", extra={"audio_transport": "inline"})
            )

        result_future = asyncio.run(run())
        self.assertEqual(result_future["content"], "from files")
        service = self.factory.services[0]
        self.assertEqual(len(service.upload_calls), 1)
        self.assertEqual(len(service.interaction_calls), 2)

    def test_dedicated_empty_uses_flash_lite_without_files_retry(self) -> None:
        async def run():
            lease = await self.plugin._client_pool.acquire("test-key")
            service = self.factory.services[0]
            service.interaction_results = [FakeInteraction("")]
            await lease.release()
            return await self.plugin.handle_llm_provider(
                "audio_transcription", request_for(b"OggS" + b"x", extra={"audio_transport": "inline"})
            )

        result = asyncio.run(run())
        self.assertEqual(result["content"], "fallback text")
        service = self.factory.services[0]
        self.assertEqual(service.upload_calls, [])
        self.assertEqual(len(service.generate_calls), 1)

    def test_same_key_reuses_client(self) -> None:
        async def run():
            first = await self.plugin.handle_llm_provider(
                "audio_transcription", request_for(b"OggS" + b"one")
            )
            second = await self.plugin.handle_llm_provider(
                "audio_transcription", request_for(b"OggS" + b"two")
            )
            return first, second

        first, second = asyncio.run(run())
        self.assertEqual(first["content"], "transcribed")
        self.assertEqual(second["content"], "transcribed")
        self.assertEqual(len(self.factory.services), 1)

    def test_different_keys_do_not_share_client(self) -> None:
        async def run():
            first = await self.plugin.handle_llm_provider(
                "audio_transcription", request_for(b"OggS" + b"one", key="key-a")
            )
            second = await self.plugin.handle_llm_provider(
                "audio_transcription", request_for(b"OggS" + b"two", key="key-b")
            )
            return first, second

        first, second = asyncio.run(run())
        self.assertEqual(first["content"], "transcribed")
        self.assertEqual(second["content"], "transcribed")
        self.assertEqual(len(self.factory.services), 2)

    def test_unload_closes_client(self) -> None:
        async def run():
            await self.plugin.handle_llm_provider("audio_transcription", request_for(b"OggS" + b"x"))
            await self.plugin.on_unload()

        asyncio.run(run())
        self.assertEqual(self.factory.sync_clients[0].aio.aclose_calls, 1)
        self.assertEqual(self.factory.sync_clients[0].close_calls, 1)

    def test_config_update_retires_existing_client(self) -> None:
        async def run():
            await self.plugin.handle_llm_provider("audio_transcription", request_for(b"OggS" + b"x"))
            await self.plugin.on_config_update("self", {"gemini": {"audio_transport": "files"}}, "2")
            await self.plugin.handle_llm_provider(
                "audio_transcription", request_for(b"OggS" + b"y", extra={"audio_transport": "inline"})
            )

        asyncio.run(run())
        self.assertEqual(len(self.factory.services), 2)
        self.assertEqual(self.factory.sync_clients[0].aio.aclose_calls, 1)
        self.assertEqual(self.factory.sync_clients[0].close_calls, 1)

    def test_unload_waits_for_in_flight_lease(self) -> None:
        async def run():
            lease = await self.plugin._client_pool.acquire("test-key")
            close_task = asyncio.create_task(self.plugin.on_unload())
            await asyncio.sleep(0)
            self.assertEqual(self.factory.sync_clients[0].close_calls, 0)
            await lease.release()
            await close_task

        asyncio.run(run())
        self.assertEqual(self.factory.sync_clients[0].aio.aclose_calls, 1)
        self.assertEqual(self.factory.sync_clients[0].close_calls, 1)

    def test_silk_prefers_dedicated_decoder(self) -> None:
        wav = make_wav(1000)
        calls = {"ffmpeg": 0}
        old_loader = audio._load_silk_decoder
        old_ffmpeg = audio._convert_with_ffmpeg

        class Decoder:
            @staticmethod
            def silk_to_wav(data, output, rate=24000):
                del data, output, rate
                return wav

        audio._load_silk_decoder = lambda: Decoder
        audio._convert_with_ffmpeg = lambda *args: calls.__setitem__("ffmpeg", calls["ffmpeg"] + 1) or wav
        try:
            normalized = audio.normalize_audio(b"#!SILK_V3" + b"payload", timeout=1)
        finally:
            audio._load_silk_decoder = old_loader
            audio._convert_with_ffmpeg = old_ffmpeg
        self.assertEqual(normalized.mime_type, "audio/wav")
        self.assertEqual(calls["ffmpeg"], 0)

    def test_silk_decoder_failure_falls_back_to_ffmpeg(self) -> None:
        wav = make_wav(1000)
        calls = {"ffmpeg": 0}
        old_loader = audio._load_silk_decoder
        old_ffmpeg = audio._convert_with_ffmpeg

        class Decoder:
            @staticmethod
            def silk_to_wav(*args, **kwargs):
                raise RuntimeError("bad sample")

        audio._load_silk_decoder = lambda: Decoder
        audio._convert_with_ffmpeg = lambda *args: calls.__setitem__("ffmpeg", calls["ffmpeg"] + 1) or wav
        try:
            normalized = audio.normalize_audio(b"#!SILK_V3" + b"payload", timeout=1)
        finally:
            audio._load_silk_decoder = old_loader
            audio._convert_with_ffmpeg = old_ffmpeg
        self.assertEqual(normalized.mime_type, "audio/wav")
        self.assertEqual(calls["ffmpeg"], 1)

    def test_clear_silence_skips_gemini(self) -> None:
        result = self.run_request(make_wav(0), key="")
        self.assertEqual(result["content"], "")
        self.assertEqual(len(self.factory.services), 0)
        self.assertEqual(result["raw_data"]["status"], "no_speech")

    def test_normal_voice_is_not_filtered(self) -> None:
        result = self.run_request(make_wav(4000))
        self.assertEqual(result["content"], "transcribed")
        self.assertEqual(len(self.factory.services), 1)

    def test_loud_noise_is_not_treated_as_silence(self) -> None:
        rng = random.Random(7)
        output = io.BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(
                b"".join(rng.randint(-20000, 20000).to_bytes(2, "little", signed=True) for _ in range(3200))
            )
        self.assertFalse(audio.is_clearly_silent(output.getvalue(), "audio/wav"))

    def test_defaults_and_old_config_are_compatible(self) -> None:
        self.assertEqual(
            plugin._transport_settings({}, {}),
            ("auto", plugin.DEFAULT_INLINE_MAX_BYTES, True),
        )
        self.assertEqual(
            plugin._transport_settings({"gemini": {"api_key": "old"}}, {}),
            ("auto", plugin.DEFAULT_INLINE_MAX_BYTES, True),
        )
        self.assertEqual(
            plugin._transport_settings(
                {"gemini": {"audio_transport": "files", "inline_max_bytes": 99, "local_speech_gate": False}},
                {},
            ),
            ("files", 99, False),
        )


if __name__ == "__main__":
    unittest.main()
