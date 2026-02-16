"""Multi-strategy video downloader with cookie support.

Tries strategies in order: yt-dlp (with optional cookies) -> direct HTTP.
Each strategy returns a DownloadResult. Stops on first success.
"""

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse, unquote, parse_qs

import httpx

logger = logging.getLogger(__name__)

COOKIE_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads", "cookies")


@dataclass
class DownloadResult:
    success: bool
    file_path: str
    strategy: str
    error: str = ""


def extract_name_from_url(url: str) -> str:
    """Extract a human-readable name from a URL."""
    if not url:
        return "Imported Recording"

    parsed = urlparse(url)

    # SharePoint: try 'id' query parameter
    qs = parse_qs(parsed.query)
    if "id" in qs:
        path = unquote(qs["id"][0])
        parts = path.rstrip("/").split("/")
        for part in reversed(parts):
            if part and "." in part:
                return part

    # Generic: try path segments
    path = unquote(parsed.path)
    parts = path.rstrip("/").split("/")
    for part in reversed(parts):
        if part and "." in part:
            return part
    for part in reversed(parts):
        if part and part not in ("stream.aspx", "_layouts", "15", "embed.aspx", "watch"):
            return part[:60]

    # YouTube: use video ID
    if "v" in qs:
        return f"YouTube_{qs['v'][0]}"

    return "Imported Recording"


class MediaIngestion:
    """Multi-strategy video downloader."""

    def __init__(self, cookie_path: Optional[str] = None):
        self.cookie_path = cookie_path  # Path to Netscape cookies.txt

    async def download(self, url: str, output_dir: str) -> DownloadResult:
        """Try each download strategy in order. Return first success or combined error."""
        errors = []

        # Strategy 1: yt-dlp (handles YouTube, Vimeo, many streaming sites)
        result = await self._try_ytdlp(url, output_dir)
        if result.success:
            return result
        errors.append(f"yt-dlp: {result.error}")

        # Strategy 2: Direct HTTP download
        result = await self._try_http(url, output_dir)
        if result.success:
            return result
        errors.append(f"http: {result.error}")

        # All strategies failed
        combined_error = " | ".join(errors)
        return DownloadResult(
            success=False,
            file_path="",
            strategy="all",
            error=combined_error,
        )

    async def _try_ytdlp(self, url: str, output_dir: str) -> DownloadResult:
        """Download using yt-dlp with optional cookie support."""
        file_id = uuid.uuid4().hex[:12]
        output_template = os.path.join(output_dir, f"dl_{file_id}.%(ext)s")

        cmd = [
            "yt-dlp",
            "-f", "best[ext=mp4]/best",
            "-o", output_template,
            "--no-playlist",
            "--socket-timeout", "30",
            "--retries", "3",
        ]

        # Add cookie support if available
        if self.cookie_path and os.path.isfile(self.cookie_path):
            cmd.extend(["--cookies", self.cookie_path])
            logger.info(f"Using cookies from {self.cookie_path}")

        cmd.append(url)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)

            if proc.returncode == 0:
                # Find the downloaded file (yt-dlp may change extension)
                for fname in os.listdir(output_dir):
                    if fname.startswith(f"dl_{file_id}"):
                        file_path = os.path.join(output_dir, fname)
                        if os.path.getsize(file_path) > 1000:
                            logger.info(f"yt-dlp OK: {file_path} ({os.path.getsize(file_path)} bytes)")
                            return DownloadResult(True, file_path, "yt-dlp")

            error_text = stderr.decode()[:300] if stderr else "unknown error"
            return DownloadResult(False, "", "yt-dlp", error_text)

        except FileNotFoundError:
            return DownloadResult(False, "", "yt-dlp", "yt-dlp not installed")
        except asyncio.TimeoutError:
            return DownloadResult(False, "", "yt-dlp", "timed out after 180s")
        except Exception as e:
            return DownloadResult(False, "", "yt-dlp", str(e)[:200])

    async def _try_http(self, url: str, output_dir: str) -> DownloadResult:
        """Direct HTTP download with retry and redirect following."""
        # Determine file extension from URL
        ext = "mp4"
        path = unquote(urlparse(url).path)
        for e in (".mp4", ".mov", ".avi", ".mkv", ".webm", ".mp3", ".wav", ".m4a"):
            if path.lower().endswith(e):
                ext = e.lstrip(".")
                break

        file_id = uuid.uuid4().hex[:12]
        file_path = os.path.join(output_dir, f"dl_{file_id}.{ext}")

        for attempt in range(3):  # Retry up to 3 times
            try:
                async with httpx.AsyncClient(
                    follow_redirects=True,
                    timeout=httpx.Timeout(60.0, connect=15.0),
                    max_redirects=5,
                ) as client:
                    async with client.stream("GET", url) as resp:
                        if resp.status_code != 200:
                            return DownloadResult(
                                False, "", "http",
                                f"HTTP {resp.status_code}",
                            )

                        # Check content-type if available
                        content_type = resp.headers.get("content-type", "")
                        if content_type and "text/html" in content_type:
                            return DownloadResult(
                                False, "", "http",
                                "URL returned HTML page, not a video file. Authentication may be required.",
                            )

                        total = 0
                        with open(file_path, "wb") as f:
                            async for chunk in resp.aiter_bytes(chunk_size=256 * 1024):
                                f.write(chunk)
                                total += len(chunk)

                        if total > 1000:
                            logger.info(f"HTTP download OK: {file_path} ({total} bytes)")
                            return DownloadResult(True, file_path, "http")
                        else:
                            return DownloadResult(
                                False, "", "http",
                                f"Download too small: {total} bytes",
                            )

            except httpx.TimeoutException:
                if attempt < 2:
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                return DownloadResult(False, "", "http", "timed out")
            except Exception as e:
                return DownloadResult(False, "", "http", str(e)[:200])

        return DownloadResult(False, "", "http", "all retries exhausted")
