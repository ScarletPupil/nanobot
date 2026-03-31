"""Tests for NapCat forward-WS channel."""

from __future__ import annotations

import asyncio
from unittest.mock import ANY, AsyncMock, patch

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.napcat import NapCatChannel, NapCatConfig


def _make_channel(*, group_policy: str = "mention") -> tuple[NapCatChannel, MessageBus]:
    bus = MessageBus()
    ch = NapCatChannel(
        NapCatConfig(
            enabled=True,
            allow_from=["*"],
            group_policy=group_policy,
        ),
        bus,
    )
    return ch, bus


def test_default_config_contains_client_fields() -> None:
    cfg = NapCatChannel.default_config()
    assert cfg["enabled"] is False
    assert cfg["wsUrl"] == ""
    assert cfg["host"] == "127.0.0.1"
    assert cfg["port"] == 3001
    assert cfg["path"] == "/onebot/v11/ws"


def test_markdown_image_output_path_contains_timestamp_and_random_suffix() -> None:
    p1 = NapCatChannel._build_markdown_image_output_path()
    p2 = NapCatChannel._build_markdown_image_output_path()
    assert p1 != p2
    assert p1.endswith(".png")
    assert p2.endswith(".png")
    assert "nanobot_md_" in p1


@pytest.mark.asyncio
async def test_private_message_event_publishes_to_bus() -> None:
    ch, bus = _make_channel()
    event = {
        "post_type": "message",
        "message_type": "private",
        "user_id": "12345",
        "message_id": 1,
        "raw_message": "hello napcat",
        "self_id": 10001,
    }

    await ch._handle_message_event(event)
    inbound = await asyncio.wait_for(bus.consume_inbound(), timeout=0.2)

    assert inbound.channel == "napcat"
    assert inbound.sender_id == "12345"
    assert inbound.chat_id == "12345"
    assert inbound.content == "hello napcat"
    assert inbound.metadata["message_type"] == "private"


@pytest.mark.asyncio
async def test_group_message_requires_mention_when_policy_is_mention() -> None:
    ch, bus = _make_channel(group_policy="mention")

    ignored = {
        "post_type": "message",
        "message_type": "group",
        "user_id": "222",
        "group_id": "333",
        "message_id": 2,
        "raw_message": "no mention",
        "self_id": 10001,
    }
    await ch._handle_message_event(ignored)
    assert bus.inbound_size == 0

    accepted = {
        "post_type": "message",
        "message_type": "group",
        "user_id": "222",
        "group_id": "333",
        "message_id": 3,
        "raw_message": "[CQ:at,qq=10001] hi bot",
        "self_id": 10001,
    }
    await ch._handle_message_event(accepted)
    inbound = await asyncio.wait_for(bus.consume_inbound(), timeout=0.2)

    assert inbound.chat_id == "333"
    assert inbound.sender_id == "222"
    assert inbound.content == "hi bot"


@pytest.mark.asyncio
async def test_send_uses_send_msg_with_group_params() -> None:
    ch, _ = _make_channel(group_policy="open")
    ch._chat_type_cache["999"] = "group"

    with patch.object(ch, "_send_action", new=AsyncMock(return_value={"status": "ok", "retcode": 0})) as mocked:
        await ch.send(OutboundMessage(channel="napcat", chat_id="999", content="hello group"))

    mocked.assert_awaited_once()
    action, params = mocked.await_args.args
    assert action == "send_msg"
    assert params["message_type"] == "group"
    assert params["group_id"] == "999"
    assert params["message"] == "hello group"


@pytest.mark.asyncio
async def test_send_builds_reply_and_media_segments() -> None:
    ch, _ = _make_channel(group_policy="open")

    with patch.object(ch, "_send_action", new=AsyncMock(return_value={"status": "ok", "retcode": 0})) as mocked:
        await ch.send(
            OutboundMessage(
                channel="napcat",
                chat_id="10001",
                content="hello",
                reply_to="888",
                media=["/tmp/a.png", "/tmp/b.pdf"],
            )
        )

    action, params = mocked.await_args.args
    assert action == "send_msg"
    assert params["message_type"] == "private"
    assert params["user_id"] == "10001"
    msg_payload = params["message"]
    assert isinstance(msg_payload, list)
    assert msg_payload[0] == {"type": "reply", "data": {"id": "888"}}
    assert msg_payload[1] == {"type": "text", "data": {"text": "hello"}}
    assert msg_payload[2]["type"] == "image"
    assert msg_payload[3]["type"] == "file"


def test_private_message_does_not_auto_reply_by_message_id() -> None:
    ch, _ = _make_channel(group_policy="open")
    payload = ch._build_message_payload(
        OutboundMessage(
            channel="napcat",
            chat_id="10001",
            content="hello",
            metadata={"message_id": 12345},
        )
    )
    assert payload == "hello"


@pytest.mark.asyncio
async def test_send_can_revoke_message() -> None:
    ch, _ = _make_channel(group_policy="open")

    with patch.object(ch, "_send_action", new=AsyncMock(return_value={"status": "ok", "retcode": 0})) as mocked:
        await ch.send(
            OutboundMessage(
                channel="napcat",
                chat_id="10001",
                content="",
                metadata={"delete_message_id": 12345},
            )
        )

    mocked.assert_awaited_once_with("delete_msg", {"message_id": 12345})


@pytest.mark.asyncio
async def test_send_renders_markdown_to_image_when_enabled() -> None:
    ch, _ = _make_channel(group_policy="open")

    with (
        patch.object(ch, "_render_markdown_to_image", new=AsyncMock(return_value="/tmp/md_render.png")) as render_mock,
        patch.object(ch, "_send_action", new=AsyncMock(return_value={"status": "ok", "retcode": 0})) as mocked,
    ):
        await ch.send(
            OutboundMessage(
                channel="napcat",
                chat_id="10001",
                content="# Title\n\n```python\nprint('x')\n```",
            )
        )

    render_mock.assert_awaited_once_with("# Title\n\n```python\nprint('x')\n```", ANY)
    action, params = mocked.await_args.args
    assert action == "send_msg"
    payload = params["message"]
    assert isinstance(payload, list)
    assert payload[0]["type"] == "image"


@pytest.mark.asyncio
async def test_handle_ws_payload_resolves_pending_echo() -> None:
    ch, _ = _make_channel()
    fut = asyncio.get_running_loop().create_future()
    ch._pending["nb-1"] = fut

    await ch._handle_ws_payload('{"status":"ok","retcode":0,"echo":"nb-1","data":{"message_id":7}}')

    assert fut.done()
    assert fut.result()["data"]["message_id"] == 7
