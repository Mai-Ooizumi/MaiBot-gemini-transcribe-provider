from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable

from google import genai


@dataclass(frozen=True, slots=True)
class ClientKey:
    # The key is never logged or exposed; it only partitions clients so one
    # provider/request cannot accidentally use another provider's credentials.
    api_key: str
    base_url: str = ""


@dataclass(slots=True)
class _ClientEntry:
    key: ClientKey
    sync_client: Any
    async_client: Any
    references: int = 0
    retired: bool = False
    closed: bool = False


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class ClientLease:
    def __init__(self, pool: "GeminiClientPool", entry: _ClientEntry) -> None:
        self._pool = pool
        self._entry = entry
        self._released = False

    @property
    def client(self) -> Any:
        return self._entry.async_client

    async def release(self) -> None:
        if not self._released:
            self._released = True
            await self._pool._release(self._entry)

    async def __aenter__(self) -> Any:
        return self.client

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.release()


class GeminiClientPool:
    """Lifecycle-scoped async Gemini clients without serializing requests."""

    def __init__(
        self,
        factory: Callable[[ClientKey], tuple[Any, Any]] | None = None,
    ) -> None:
        self._factory = factory or self._create_client
        self._entries: dict[ClientKey, _ClientEntry] = {}
        self._lock = asyncio.Lock()
        self._active_references = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._closing = False

    @staticmethod
    def _create_client(key: ClientKey) -> tuple[Any, Any]:
        # Use the official async facade.  A sync parent is retained because the
        # SDK documents that AsyncClient.aclose() does not close it.
        sync_client = genai.Client(api_key=key.api_key)
        return sync_client, sync_client.aio

    async def acquire(self, api_key: str, *, base_url: str = "") -> ClientLease:
        key = ClientKey(api_key=api_key, base_url=base_url)
        async with self._lock:
            if self._closing:
                raise RuntimeError("Gemini client pool 正在关闭")
            entry = self._entries.get(key)
            if entry is None or entry.retired:
                sync_client, async_client = self._factory(key)
                entry = _ClientEntry(key=key, sync_client=sync_client, async_client=async_client)
                self._entries[key] = entry
            entry.references += 1
            self._active_references += 1
            self._idle.clear()
            return ClientLease(self, entry)

    async def _release(self, entry: _ClientEntry) -> None:
        should_close = False
        async with self._lock:
            if entry.references > 0:
                entry.references -= 1
                self._active_references = max(0, self._active_references - 1)
            if entry.references == 0 and entry.retired and not self._closing:
                if self._entries.get(entry.key) is entry:
                    del self._entries[entry.key]
                should_close = True
            if self._active_references == 0:
                self._idle.set()
        if should_close:
            await self._close_entry(entry)

    async def invalidate(self) -> None:
        """Retire clients after a config reload without interrupting in-flight calls."""
        to_close: list[_ClientEntry] = []
        async with self._lock:
            for entry in self._entries.values():
                entry.retired = True
                if entry.references == 0:
                    to_close.append(entry)
            for entry in to_close:
                if self._entries.get(entry.key) is entry:
                    del self._entries[entry.key]
        await asyncio.gather(*(self._close_entry(entry) for entry in to_close))

    async def close(self) -> None:
        async with self._lock:
            if self._closing:
                wait_for_idle = self._active_references > 0
                to_close: list[_ClientEntry] = []
            else:
                self._closing = True
                for entry in self._entries.values():
                    entry.retired = True
                wait_for_idle = self._active_references > 0
                if wait_for_idle:
                    to_close = []
                else:
                    to_close = list(self._entries.values())
                    self._entries.clear()
        if wait_for_idle:
            await self._idle.wait()
            async with self._lock:
                to_close = list(self._entries.values())
                self._entries.clear()
        await asyncio.gather(*(self._close_entry(entry) for entry in to_close))

    async def _close_entry(self, entry: _ClientEntry) -> None:
        if entry.closed:
            return
        entry.closed = True
        try:
            await _maybe_await(getattr(entry.async_client, "aclose", lambda: None)())
        finally:
            close_sync = getattr(entry.sync_client, "close", None)
            if close_sync is not None:
                try:
                    await _maybe_await(close_sync())
                except Exception:
                    # Async close is the authoritative cleanup path.  A sync
                    # facade may already share a closed transport.
                    pass
