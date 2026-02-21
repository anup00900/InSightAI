"""Lightweight wrapper for Core42 API calls with timeout, retry, and validation."""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

API_TIMEOUT = 30  # seconds
MAX_RETRIES = 1
RETRY_DELAY = 2.0  # seconds


async def safe_api_call(
    coro_factory,
    *,
    timeout: float = API_TIMEOUT,
    retries: int = MAX_RETRIES,
    required_keys: list[str] | None = None,
    fallback: dict | list | None = None,
    label: str = "api_call",
) -> dict | list:
    """Execute an async API call with timeout, retry, and response validation.

    Args:
        coro_factory: A zero-argument callable that returns a new coroutine each call.
        timeout: Max seconds to wait per attempt.
        retries: Number of retries on transient errors (0 = no retry).
        required_keys: If set, validate response dict contains these keys.
        fallback: Value to return on total failure. If None, re-raises the exception.
        label: Human-readable label for log messages.
    """
    last_error = None
    for attempt in range(1 + retries):
        try:
            result = await asyncio.wait_for(coro_factory(), timeout=timeout)

            # Validate required keys if specified
            if required_keys and isinstance(result, dict):
                missing = [k for k in required_keys if k not in result]
                if missing:
                    logger.warning(f"[{label}] Response missing keys: {missing}")
                    if fallback is not None:
                        return fallback

            return result

        except asyncio.TimeoutError:
            last_error = TimeoutError(f"{label} timed out after {timeout}s")
            logger.warning(f"[{label}] Timeout (attempt {attempt + 1}/{1 + retries})")

        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            is_transient = any(
                code in error_str
                for code in ["429", "500", "502", "503", "504", "rate", "connection", "timeout"]
            )

            if not is_transient or attempt >= retries:
                logger.error(f"[{label}] Failed: {e}")
                if fallback is not None:
                    return fallback
                raise

            logger.warning(f"[{label}] Transient error (attempt {attempt + 1}): {e}")

        # Wait before retry
        if attempt < retries:
            await asyncio.sleep(RETRY_DELAY * (attempt + 1))

    # All attempts exhausted
    if fallback is not None:
        logger.error(f"[{label}] All attempts failed, using fallback")
        return fallback
    raise last_error or RuntimeError(f"{label} failed with no error captured")


async def cascade_api_call(
    coro_factories: list,
    *,
    timeout: float = API_TIMEOUT,
    required_keys: list[str] | None = None,
    fallback: dict | list | None = None,
    label: str = "cascade_api_call",
) -> dict | list:
    """Try a list of coroutine factories in order. First success wins.

    Used for summary/finalization where accuracy matters more than latency.
    Each factory is tried once; on failure the next is attempted.
    """
    last_error = None
    for i, factory in enumerate(coro_factories):
        model_label = f"{label}[{i}]"
        try:
            result = await asyncio.wait_for(factory(), timeout=timeout)
            if required_keys and isinstance(result, dict):
                missing = [k for k in required_keys if k not in result]
                if missing:
                    logger.warning(f"[{model_label}] Response missing keys: {missing}")
                    continue
            logger.info(f"[{model_label}] Succeeded")
            return result
        except Exception as e:
            last_error = e
            logger.warning(f"[{model_label}] Failed: {e}, trying next...")

    if fallback is not None:
        logger.error(f"[{label}] All cascade models failed, using fallback")
        return fallback
    raise last_error or RuntimeError(f"{label} cascade exhausted")
