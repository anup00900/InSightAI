"""
SignalBus — central async event bus that decouples signal producers
(Visual / Audio / Text pipelines) from consumers (EngagementEngine,
WebSocket, DB).

All three pipelines publish normalised SignalEvent objects here; consumers
subscribe to the signal types they care about.
"""

from __future__ import annotations

import asyncio
import enum
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Dict, List


# ---------------------------------------------------------------------------
# Enums & data
# ---------------------------------------------------------------------------

class SignalType(enum.Enum):
    """The four kinds of signal the system handles."""
    VISUAL = "visual"
    AUDIO = "audio"
    TEXT = "text"
    ENGAGEMENT = "engagement"


@dataclass
class SignalEvent:
    """A single normalised signal emitted by any pipeline."""
    signal_type: SignalType
    participant_id: str
    timestamp: float
    data: dict = field(default_factory=dict)


# Type alias for an async handler function.
AsyncHandler = Callable[[SignalEvent], Awaitable[None]]


# ---------------------------------------------------------------------------
# SignalBus
# ---------------------------------------------------------------------------

class SignalBus:
    """Async publish/subscribe event bus for SignalEvents.

    Usage
    -----
    >>> bus = SignalBus()
    >>> unsub = bus.subscribe(SignalType.VISUAL, my_handler)
    >>> await bus.publish(SignalEvent(...))
    >>> unsub()   # stop receiving events
    """

    def __init__(self) -> None:
        # Per-type subscriber lists.
        self._subscribers: Dict[SignalType, List[AsyncHandler]] = {
            st: [] for st in SignalType
        }

    # ------------------------------------------------------------------
    # Subscribe
    # ------------------------------------------------------------------

    def subscribe(
        self,
        signal_type: SignalType,
        handler: AsyncHandler,
    ) -> Callable[[], None]:
        """Register *handler* to receive events of *signal_type*.

        Returns a callable that, when invoked, removes the subscription.
        """
        self._subscribers[signal_type].append(handler)

        def _unsubscribe() -> None:
            try:
                self._subscribers[signal_type].remove(handler)
            except ValueError:
                pass  # already removed

        return _unsubscribe

    def subscribe_all(
        self,
        handler: AsyncHandler,
    ) -> Callable[[], None]:
        """Register *handler* to receive events of **every** signal type.

        Returns a callable that removes the subscription from all types.
        """
        unsubs = [self.subscribe(st, handler) for st in SignalType]

        def _unsubscribe_all() -> None:
            for unsub in unsubs:
                unsub()

        return _unsubscribe_all

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    async def publish(self, event: SignalEvent) -> None:
        """Dispatch *event* to every handler subscribed to its signal type.

        Handlers are awaited concurrently via ``asyncio.gather``.
        """
        handlers = list(self._subscribers[event.signal_type])
        if handlers:
            await asyncio.gather(*(h(event) for h in handlers))
