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
    call_factory,
    *,
    models: list[str],
    timeout: float = API_TIMEOUT,
    retries: int = MAX_RETRIES,
    required_keys: list[str] | None = None,
    fallback: dict | list | None = None,
    label: str = "cascade_call",
) -> dict | list:
    """Try multiple models in sequence until one succeeds.

    Args:
        call_factory: A callable that takes a model name and returns a coroutine factory.
                      Usage: call_factory(model) returns a zero-arg callable for safe_api_call.
        models: Ordered list of model names to try.
        timeout, retries, required_keys: Passed through to safe_api_call.
        fallback: Final fallback if ALL models fail.
        label: Human-readable label for logs.
    """
    for i, model in enumerate(models):
        try:
            result = await safe_api_call(
                call_factory(model),
                timeout=timeout,
                retries=retries,
                required_keys=required_keys,
                fallback=None,  # Don't use fallback yet — try next model
                label=f"{label}_{model}",
            )
            if result is not None:
                if i > 0:
                    logger.info(f"[{label}] Succeeded with fallback model {model} (attempt {i + 1})")
                return result
        except Exception as e:
            logger.warning(f"[{label}] Model {model} failed: {e}")
            if i < len(models) - 1:
                logger.info(f"[{label}] Trying next model: {models[i + 1]}")
            continue

    logger.error(f"[{label}] All {len(models)} models failed, using fallback")
    if fallback is not None:
        return fallback
    raise RuntimeError(f"{label}: all models exhausted")
