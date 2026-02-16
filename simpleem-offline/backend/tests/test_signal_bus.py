"""Tests for the SignalBus event bus."""

import asyncio
import pytest
from backend.signal_bus import SignalBus, SignalEvent, SignalType


@pytest.fixture
def bus():
    """Return a fresh SignalBus for each test."""
    return SignalBus()


@pytest.mark.asyncio
async def test_publish_and_subscribe(bus):
    """Subscribe to VISUAL, publish a VISUAL event, verify the handler receives it."""
    received = []

    async def handler(event: SignalEvent):
        received.append(event)

    bus.subscribe(SignalType.VISUAL, handler)

    event = SignalEvent(
        signal_type=SignalType.VISUAL,
        participant_id="p1",
        timestamp=1.0,
        data={"smile": 0.8},
    )
    await bus.publish(event)

    assert len(received) == 1
    assert received[0] is event
    assert received[0].signal_type == SignalType.VISUAL
    assert received[0].participant_id == "p1"
    assert received[0].timestamp == 1.0
    assert received[0].data == {"smile": 0.8}


@pytest.mark.asyncio
async def test_multiple_subscribers(bus):
    """Two handlers for AUDIO both receive the same event."""
    received_a = []
    received_b = []

    async def handler_a(event: SignalEvent):
        received_a.append(event)

    async def handler_b(event: SignalEvent):
        received_b.append(event)

    bus.subscribe(SignalType.AUDIO, handler_a)
    bus.subscribe(SignalType.AUDIO, handler_b)

    event = SignalEvent(
        signal_type=SignalType.AUDIO,
        participant_id="p2",
        timestamp=2.0,
        data={"volume": 0.6},
    )
    await bus.publish(event)

    assert len(received_a) == 1
    assert len(received_b) == 1
    assert received_a[0] is event
    assert received_b[0] is event


@pytest.mark.asyncio
async def test_subscribe_all(bus):
    """Handler subscribed to all types receives VISUAL, AUDIO, and TEXT events."""
    received = []

    async def handler(event: SignalEvent):
        received.append(event)

    bus.subscribe_all(handler)

    for st in (SignalType.VISUAL, SignalType.AUDIO, SignalType.TEXT):
        await bus.publish(
            SignalEvent(signal_type=st, participant_id="p3", timestamp=3.0)
        )

    assert len(received) == 3
    types = [e.signal_type for e in received]
    assert SignalType.VISUAL in types
    assert SignalType.AUDIO in types
    assert SignalType.TEXT in types


@pytest.mark.asyncio
async def test_unsubscribe(bus):
    """After calling unsubscribe(), the handler no longer receives events."""
    received = []

    async def handler(event: SignalEvent):
        received.append(event)

    unsubscribe = bus.subscribe(SignalType.TEXT, handler)

    # First publish — handler should fire
    await bus.publish(
        SignalEvent(signal_type=SignalType.TEXT, participant_id="p4", timestamp=4.0)
    )
    assert len(received) == 1

    # Unsubscribe and publish again — handler must NOT fire
    unsubscribe()
    await bus.publish(
        SignalEvent(signal_type=SignalType.TEXT, participant_id="p4", timestamp=5.0)
    )
    assert len(received) == 1  # still 1, not 2
