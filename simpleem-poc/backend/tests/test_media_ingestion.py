import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from backend.media_ingestion import (
    MediaIngestion,
    DownloadResult,
    extract_name_from_url,
)


class TestExtractNameFromUrl:
    def test_youtube_url(self):
        name = extract_name_from_url("https://www.youtube.com/watch?v=abc123")
        assert name  # Should extract something meaningful
        assert len(name) > 0

    def test_direct_mp4_url(self):
        name = extract_name_from_url("https://example.com/videos/meeting.mp4")
        assert name == "meeting.mp4"

    def test_sharepoint_url(self):
        name = extract_name_from_url(
            "https://company.sharepoint.com/_layouts/15/stream.aspx?id=%2Fsites%2Fteam%2FRecording.mp4"
        )
        assert "Recording.mp4" in name

    def test_empty_url(self):
        name = extract_name_from_url("")
        assert name == "Imported Recording"


class TestDownloadResult:
    def test_success_result(self):
        result = DownloadResult(
            success=True,
            file_path="/tmp/video.mp4",
            strategy="yt-dlp",
        )
        assert result.success
        assert result.file_path == "/tmp/video.mp4"

    def test_failure_result(self):
        result = DownloadResult(
            success=False,
            file_path="",
            strategy="http",
            error="404 Not Found",
        )
        assert not result.success
        assert result.error == "404 Not Found"


class TestMediaIngestion:
    @pytest.mark.asyncio
    async def test_download_tries_strategies_in_order(self):
        ingestion = MediaIngestion(cookie_path=None)

        # Both strategies fail → final result is failure with combined error
        with patch.object(ingestion, '_try_ytdlp', new_callable=AsyncMock) as mock_yt, \
             patch.object(ingestion, '_try_http', new_callable=AsyncMock) as mock_http:
            mock_yt.return_value = DownloadResult(False, "", "yt-dlp", "not installed")
            mock_http.return_value = DownloadResult(False, "", "http", "404")

            result = await ingestion.download("https://example.com/vid.mp4", "/tmp")
            assert not result.success
            assert "yt-dlp" in result.error
            assert "404" in result.error
            mock_yt.assert_called_once()
            mock_http.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_stops_on_first_success(self):
        ingestion = MediaIngestion(cookie_path=None)

        with patch.object(ingestion, '_try_ytdlp', new_callable=AsyncMock) as mock_yt, \
             patch.object(ingestion, '_try_http', new_callable=AsyncMock) as mock_http:
            mock_yt.return_value = DownloadResult(True, "/tmp/vid.mp4", "yt-dlp")

            result = await ingestion.download("https://youtube.com/watch?v=abc", "/tmp")
            assert result.success
            assert result.file_path == "/tmp/vid.mp4"
            mock_yt.assert_called_once()
            mock_http.assert_not_called()  # Didn't need to try HTTP
