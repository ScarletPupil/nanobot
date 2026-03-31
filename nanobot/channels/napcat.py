"""NapCat OneBot11 channel using forward WebSocket client mode."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import tempfile
from datetime import datetime
from collections import OrderedDict
from itertools import count
from uuid import uuid4
from typing import Any, Literal

from loguru import logger
from pydantic import Field

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import Base


class NapCatConfig(Base):
    """NapCat channel configuration (WS client)."""

    enabled: bool = False
    ws_url: str = ""
    host: str = "127.0.0.1"
    port: int = 3001
    path: str = "/onebot/v11/ws"
    access_token: str = ""
    allow_from: list[str] = Field(default_factory=list)
    group_policy: Literal["open", "mention"] = "mention"
    send_timeout_s: float = 12.0
    reconnect_delay_s: float = 5.0
    connect_timeout_s: float = 15.0
    render_markdown_as_image: bool = True


class NapCatChannel(BaseChannel):
    """NapCat channel using OneBot11 over a client WebSocket connection."""

    name = "napcat"
    display_name = "NapCat"
    _MD_HINT_RE = re.compile(
        r"```|^#{1,6}\s+|^\s*[-*+]\s+|^\s*\d+\.\s+|\[[^\]]+\]\([^\)]+\)|\*\*.+?\*\*|~~.+?~~|\|.+\|",
        re.MULTILINE,
    )
    _MARKDOWN_IMAGE_WIDTH = 760
    _GITHUB_MARKDOWN_CSS = """
body {
    margin: 0;
    background: #ffffff;
    color: #24292f;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
}
.markdown-body {
    box-sizing: border-box;
    width: 100%;
    max-width: 760px;
    margin: 0 auto;
    padding: 24px 28px;
    line-height: 1.65;
    font-size: 16px;
    overflow-wrap: break-word;
}
.markdown-body h1, .markdown-body h2, .markdown-body h3, .markdown-body h4 {
    margin-top: 24px;
    margin-bottom: 14px;
    line-height: 1.3;
    border-bottom: 1px solid #d0d7de;
    padding-bottom: .3em;
}
.markdown-body p, .markdown-body ul, .markdown-body ol, .markdown-body table, .markdown-body pre, .markdown-body blockquote {
    margin-top: 0;
    margin-bottom: 16px;
}
.markdown-body blockquote {
    margin-left: 0;
    padding: 0 1em;
    color: #57606a;
    border-left: .25em solid #d0d7de;
}
.markdown-body code {
    padding: .2em .4em;
    font-size: 85%;
    border-radius: 6px;
    background: rgba(175,184,193,0.2);
    font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, monospace;
}
.markdown-body pre {
    padding: 14px;
    border-radius: 8px;
    overflow-x: auto;
    background: #f6f8fa;
    border: 1px solid #d0d7de;
}
.markdown-body pre code {
    padding: 0;
    background: transparent;
}
.markdown-body table {
    display: block;
    border-collapse: collapse;
    width: max-content;
    max-width: 100%;
    overflow-x: auto;
    font-size: 14px;
    background: #ffffff;
}
.markdown-body th, .markdown-body td {
    border: 1px solid #d0d7de;
    padding: 8px 12px;
    white-space: pre-wrap;
    vertical-align: top;
}
.markdown-body th {
    background: #f6f8fa;
    font-weight: 600;
}
.markdown-body tr:nth-child(even) td {
    background: #fcfcfd;
}
.markdown-body img {
    max-width: 100%;
}
"""

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return NapCatConfig().model_dump(by_alias=True)

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = NapCatConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: NapCatConfig = config
        self._ws: Any | None = None
        self._connected = False
        self._pending: dict[str, asyncio.Future] = {}
        self._echo = count(1)
        self._send_lock = asyncio.Lock()
        self._chat_type_cache: dict[str, str] = {}
        self._processed_message_ids: OrderedDict[str, None] = OrderedDict()

    async def start(self) -> None:
        """Start channel and maintain a reconnecting WS client session."""
        import websockets

        self._running = True
        ws_url = self._build_ws_url()

        while self._running:
            try:
                headers = self._build_auth_headers()
                async with self._connect_client(websockets, ws_url, headers) as ws:
                    self._ws = ws
                    self._connected = True
                    logger.info("Connected to NapCat WS server: {}", ws_url)

                    async for raw in ws:
                        if isinstance(raw, bytes):
                            raw = raw.decode("utf-8", errors="ignore")
                        await self._handle_ws_payload(raw)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("NapCat connection error: {}", e)
            finally:
                self._connected = False
                self._ws = None

            if self._running:
                await asyncio.sleep(self.config.reconnect_delay_s)

    async def stop(self) -> None:
        """Stop channel and release WS resources."""
        self._running = False
        self._connected = False
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()

    async def send(self, msg: OutboundMessage) -> None:
        """Send message via OneBot action over the active WebSocket connection."""
        # Optional control flow: revoke a previously sent/received message.
        revoke_id = self._extract_revoke_message_id(msg)
        if revoke_id is not None:
            try:
                await self._send_action("delete_msg", {"message_id": revoke_id})
            except Exception as e:
                logger.error("Error deleting NapCat message {}: {}", revoke_id, e)
            return

        content = msg.content or ""
        media = list(msg.media or [])

        if self.config.render_markdown_as_image and content.strip() and self._should_render_markdown_as_image(content):
            output_path = self._build_markdown_image_output_path()
            rendered = await self._render_markdown_to_image(content, output_path)
            if rendered:
                media.insert(0, rendered)
                content = ""

        message_type = self._resolve_message_type(msg)
        params: dict[str, Any] = {
            "message_type": message_type,
            "message": self._build_message_payload(msg, content_override=content, media_override=media),
            "auto_escape": True,
        }
        if message_type == "group":
            params["group_id"] = str(msg.chat_id)
        else:
            params["user_id"] = str(msg.chat_id)

        try:
            await self._send_action("send_msg", params)
        except Exception as e:
            logger.error("Error sending NapCat message: {}", e)

    @staticmethod
    def _extract_revoke_message_id(msg: OutboundMessage) -> int | str | None:
        meta = msg.metadata or {}
        candidate = meta.get("delete_message_id")
        if candidate is None:
            candidate = meta.get("revoke_message_id")
        if candidate is None:
            return None
        if isinstance(candidate, (int, str)) and str(candidate).strip():
            return candidate
        return None

    @staticmethod
    def _media_segment_for_path(path: str) -> dict[str, Any] | None:
        file_ref = NapCatChannel._normalize_media_ref(path)
        if not file_ref:
            return None
        ext = os.path.splitext(path)[1].lower()
        if ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
            return {"type": "image", "data": {"file": file_ref}}
        if ext in {".mp3", ".wav", ".ogg", ".opus", ".m4a", ".amr"}:
            return {"type": "record", "data": {"file": file_ref}}
        if ext in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
            return {"type": "video", "data": {"file": file_ref}}
        return {"type": "file", "data": {"file": file_ref}}

    @staticmethod
    def _normalize_media_ref(path: str) -> str | None:
        """Normalize outbound media reference for remote NapCat compatibility.

        If *path* is a local file, convert it to base64:// payload so the remote
        NapCat process can consume it without shared filesystem access.
        """
        p = path.strip()
        if not p:
            return None
        if p.startswith(("http://", "https://", "base64://", "file://")):
            return p
        if not os.path.isfile(p):
            return f"file://{p}"
        try:
            with open(p, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("ascii")
            return f"base64://{encoded}"
        except Exception as e:
            logger.warning("Failed to read media file for base64 payload {}: {}", p, e)
            return f"file://{p}"

    def _build_message_payload(
        self,
        msg: OutboundMessage,
        *,
        content_override: str | None = None,
        media_override: list[str] | None = None,
    ) -> str | list[dict[str, Any]]:
        """Build OneBot11 message payload with reply/media support."""
        meta = msg.metadata or {}
        message_type = self._resolve_message_type(msg)
        # Private chat default: do not auto-reply by inbound message_id.
        reply_id = msg.reply_to or meta.get("reply_to_message_id")
        if reply_id is None and message_type == "group":
            reply_id = meta.get("message_id")

        segments: list[dict[str, Any]] = []
        if reply_id is not None and str(reply_id).strip():
            segments.append({"type": "reply", "data": {"id": str(reply_id)}})

        text = (content_override if content_override is not None else msg.content or "").strip()
        if text:
            segments.append({"type": "text", "data": {"text": text}})

        for media_path in (media_override if media_override is not None else (msg.media or [])):
            if not isinstance(media_path, str) or not media_path.strip():
                continue
            seg = self._media_segment_for_path(media_path.strip())
            if seg:
                segments.append(seg)

        if not segments:
            return ""
        if len(segments) == 1 and segments[0].get("type") == "text":
            return text
        return segments

    @classmethod
    def _should_render_markdown_as_image(cls, content: str) -> bool:
        # Render only when content clearly contains markdown syntax and is not too short.
        return len(content.strip()) >= 40 and bool(cls._MD_HINT_RE.search(content))

    @staticmethod
    def _build_markdown_image_output_path() -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        rand = uuid4().hex[:8]
        return os.path.join(tempfile.gettempdir(), f"nanobot_md_{stamp}_{rand}.png")

    @classmethod
    def _build_markdown_html(cls, body_html: str) -> str:
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<style>{cls._GITHUB_MARKDOWN_CSS}</style>"
            "</head><body>"
            f"<article class='markdown-body'>{body_html}</article>"
            "</body></html>"
        )

    @staticmethod
    def _normalize_markdown_text(md_text: str) -> str:
        """Normalize common LLM markdown quirks for better HTML rendering."""
        text = md_text.replace("\r\n", "\n").strip()
        if not text:
            return text

        # Convert full-width separators often produced in CJK markdown answers.
        text = text.replace("｜", "|")

        normalized_lines: list[str] = []
        for line in text.split("\n"):
            stripped = line.strip()
            # Unescape pipes for table rows and trim outer spaces for parser stability.
            if stripped.count("|") >= 2:
                normalized_lines.append(stripped.replace("\\|", "|"))
            else:
                normalized_lines.append(line)
        return "\n".join(normalized_lines)

    async def _render_markdown_to_image(self, md_text: str, output_path: str) -> str | None:
        """Render markdown to image via HTML + Playwright screenshot."""
        try:
            from markdown import markdown
            from playwright.async_api import async_playwright
        except Exception as e:
            logger.warning("Markdown/Playwright not available, skip markdown image rendering: {}", e)
            return None

        text = self._normalize_markdown_text(md_text)
        if not text:
            return None

        try:
            html_body = markdown(
                text,
                extensions=["extra", "fenced_code", "tables", "nl2br", "sane_lists"],
            )
            page_html = self._build_markdown_html(html_body)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                try:
                    page = await browser.new_page(
                        viewport={"width": self._MARKDOWN_IMAGE_WIDTH + 64, "height": 120},
                    )
                    await page.set_content(page_html, wait_until="networkidle")
                    await page.screenshot(path=output_path, full_page=True, type="png")
                finally:
                    await browser.close()

            return output_path if os.path.exists(output_path) else None
        except Exception as e:
            logger.error("Failed to render markdown screenshot with Playwright: {}", e)
            return None

    def _resolve_message_type(self, msg: OutboundMessage) -> str:
        meta = msg.metadata or {}
        mt = str(meta.get("message_type") or meta.get("chat_type") or "").lower()
        if mt in {"group", "private"}:
            return mt
        return self._chat_type_cache.get(str(msg.chat_id), "private")

    async def _send_action(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        ws = self._ws
        if ws is None or not self._connected:
            raise RuntimeError("NapCat is not connected")

        echo = f"nb-{next(self._echo)}"
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._pending[echo] = fut
        payload = {"action": action, "params": params, "echo": echo}

        async with self._send_lock:
            await ws.send(json.dumps(payload, ensure_ascii=False))

        try:
            data = await asyncio.wait_for(fut, timeout=self.config.send_timeout_s)
        finally:
            self._pending.pop(echo, None)

        if str(data.get("status", "")).lower() == "failed" or int(data.get("retcode", 0) or 0) != 0:
            raise RuntimeError(
                "NapCat action failed: "
                f"status={data.get('status')} retcode={data.get('retcode')} "
                f"message={data.get('message')} wording={data.get('wording')}"
            )
        return data

    def _build_ws_url(self) -> str:
        if self.config.ws_url.strip():
            return self.config.ws_url.strip()
        path = self.config.path or "/"
        if not path.startswith("/"):
            path = "/" + path
        return f"ws://{self.config.host}:{self.config.port}{path}"
      
    def _build_auth_headers(self) -> dict[str, str] | None:
        token = self.config.access_token.strip()
        if not token:
            return None
        return {"Authorization": f"Bearer {token}"}

    def _connect_client(self, websockets_mod: Any, ws_url: str, headers: dict[str, str] | None):
        """Create WS client connection context across websockets versions."""
        kwargs = {
            "open_timeout": self.config.connect_timeout_s,
        }
        if headers:
            try:
                return websockets_mod.connect(ws_url, additional_headers=headers, **kwargs)
            except TypeError:
                return websockets_mod.connect(ws_url, extra_headers=headers, **kwargs)
        return websockets_mod.connect(ws_url, **kwargs)

    async def _handle_ws_payload(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("NapCat: invalid JSON frame ignored")
            return

        if not isinstance(data, dict):
            return

        # Action responses carry echo/status/retcode; complete the waiting future.
        echo = data.get("echo")
        if echo is not None and ("retcode" in data or "status" in data):
            fut = self._pending.get(str(echo))
            if fut and not fut.done():
                fut.set_result(data)
            return

        if data.get("post_type") == "message":
            await self._handle_message_event(data)

    async def _handle_message_event(self, event: dict[str, Any]) -> None:
        message_id = str(event.get("message_id", ""))
        if message_id:
            if message_id in self._processed_message_ids:
                return
            self._processed_message_ids[message_id] = None
            while len(self._processed_message_ids) > 1000:
                self._processed_message_ids.popitem(last=False)

        message_type = str(event.get("message_type") or "private")
        user_id = str(event.get("user_id") or "")
        group_id = str(event.get("group_id") or "")
        self_id = str(event.get("self_id") or "")

        if message_type == "group" and not self._is_group_message_for_bot(event, self_id):
            return

        chat_id = group_id if message_type == "group" else user_id
        if not chat_id or not user_id:
            return

        self._chat_type_cache[chat_id] = "group" if message_type == "group" else "private"
        content = self._extract_text(event, self_id)
        if not content:
            return

        await self._handle_message(
            sender_id=user_id,
            chat_id=chat_id,
            content=content,
            metadata={
                "message_id": event.get("message_id"),
                "post_type": event.get("post_type"),
                "message_type": message_type,
                "sub_type": event.get("sub_type"),
                "group_id": group_id or None,
                "user_id": user_id,
                "time": event.get("time"),
            },
        )

    def _is_group_message_for_bot(self, event: dict[str, Any], self_id: str) -> bool:
        if self.config.group_policy == "open":
            return True
        if not self_id:
            return False

        msg = event.get("message")
        if isinstance(msg, list):
            for seg in msg:
                if not isinstance(seg, dict):
                    continue
                if seg.get("type") != "at":
                    continue
                qq = (seg.get("data") or {}).get("qq")
                if str(qq) == self_id:
                    return True

        raw = str(event.get("raw_message") or "")
        return f"[CQ:at,qq={self_id}]" in raw or f"[CQ:at,qq={self_id}," in raw

    @staticmethod
    def _extract_text(event: dict[str, Any], self_id: str) -> str:
        raw = event.get("raw_message")
        if isinstance(raw, str) and raw.strip():
            text = raw.strip()
            if self_id:
                text = re.sub(rf"\[CQ:at,qq={re.escape(self_id)}(?:,[^\]]*)?\]", "", text).strip()
            return text

        msg = event.get("message")
        if isinstance(msg, str):
            return msg.strip()
        if not isinstance(msg, list):
            return ""

        parts: list[str] = []
        for seg in msg:
            if not isinstance(seg, dict):
                continue
            seg_type = seg.get("type")
            data = seg.get("data") or {}
            if seg_type == "text":
                t = data.get("text")
                if isinstance(t, str) and t:
                    parts.append(t)
            elif seg_type == "at":
                qq = str(data.get("qq") or "")
                if qq and qq != self_id:
                    parts.append(f"@{qq}")
            elif seg_type in {"image", "record", "video", "file"}:
                parts.append(f"[{seg_type}]")
        return " ".join(p for p in parts if p).strip()