"""True real-time analysis session.

Pure live: when video plays, analysis happens simultaneously.
No pre-analysis. Frames analyzed as they occur. Audio transcribed in real-time chunks.
Engagement chart flows EVERY SECOND via heartbeat signals.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid

import aiosqlite
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from .models import DB_PATH
from .frame_extractor import extract_single_frame, extract_full_audio, extract_audio_chunk, extract_audio_chunk_from_video
from .video_processor import get_video_duration
from .core42_client import (
    analyze_frame_all_signals,
    analyze_voice_signal,
    analyze_words_signal,
    analyze_personality,
    build_correlations,
    transcribe_audio,
    cleanup_transcription,
    attribute_speaker_from_context,
    build_speaker_buckets,
    generate_coaching,
    generate_summary_and_flags,
    extract_names_from_frame,
    _next_vision_model,
)
from .pipelines.audio import AudioAnalyzer
from .pipelines.diarization import diarize_audio, assign_speakers_to_transcript, DiarizationSegment
from .signal_bus import SignalBus, SignalEvent, SignalType
from .engagement_engine import EngagementEngine, EngagementAlertDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "processed")


class RealtimeSession:
    """Pure real-time analysis — everything happens live as video plays."""

    FRAME_INTERVAL = 1.0
    MAX_CONCURRENT = 8
    MAX_TRACKED_TASKS = 15       # Backpressure: skip frame tick if exceeded
    AUDIO_CHUNK_SECONDS = 1.5  # Smaller chunks for faster transcript delivery
    AUDIO_MIN_CHUNK = 0.8
    AUDIO_WORKER_MIN = 0.5
    MAX_PARALLEL_TRANSCRIBE = 2
    PERSONALITY_INTERVAL = 30.0
    CORRELATION_INTERVAL = 45.0
    HEARTBEAT_INTERVAL = 2.0  # Reduced from 1.0 to halve GC pressure
    NAME_RESCAN_INTERVAL = 5.0  # Scan every 5s — catches late joiners quickly
    NAME_SCAN_PLAYBACK_INTERVAL = 30.0  # Extract names from current frame every 30s during playback
    NAME_SCAN_STABLE_COUNT = 3  # Lock after N consecutive scans with same participant count

    # State caps — evict oldest when exceeded
    MAX_SIGNAL_SNAPSHOTS = 120
    MAX_TRANSCRIPT_SEGMENTS = 300
    MAX_VOICE_SIGNALS = 120
    MAX_WORD_SIGNALS = 120
    MAX_AUDIO_FEATURES = 120

    # Frame pre-buffer
    FRAME_BUFFER_AHEAD = 3  # Pre-extract this many frames ahead

    WHISPER_HALLUCINATIONS = {
        "you", "you.", "you!", "You", "You.", "bye", "bye.", "bye-bye", "bye-bye.",
        "thanks", "thanks.", "thank you", "thank you.", "Thank you.", "Thank you",
        "thanks for watching", "thanks for watching.", "Thanks for watching.",
        "subscribe", "like and subscribe", "Bye.", "Bye", "Bye-bye.",
        "uh", "um", "hmm", "huh", "oh", "ah",
        # Common non-English Whisper hallucinations
        "뉴스", "자막", "구독", "감사합니다", "시청해주셔서", "MBC 뉴스",
        "字幕", "谢谢", "订阅", "感谢观看", "請訂閱",
        "ご視聴ありがとうございました", "チャンネル登録",
    }

    def __init__(self, ws: WebSocket, video_id: str, video_path: str):
        self.ws = ws
        self.video_id = video_id
        self.video_path = video_path
        self.output_dir = os.path.join(PROCESSED_DIR, video_id, "realtime")
        os.makedirs(self.output_dir, exist_ok=True)

        # Playback state
        self.is_playing = False
        self.last_analyzed_ts = -999.0
        self._last_heartbeat_ts = -999.0

        # Concurrency — tracked task registry with backpressure
        self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT)
        self._tracked_tasks: set[asyncio.Task] = set()

        # Audio streaming state
        self._audio_path: str | None = None
        self._audio_ready = False
        self._audio_extracted = asyncio.Event()
        self._audio_transcribed_up_to = 0.0
        self._transcribe_lock = asyncio.Lock()
        self._active_transcribe_count = 0
        self._transcribe_queue: float = 0.0
        self._transcript_segments: list[dict] = []
        self._next_segment_idx = 0
        self._audio_task: asyncio.Task | None = None
        self._transcribe_tasks: list[asyncio.Task] = []

        # Accumulated analysis data (capped)
        self.signal_snapshots: list[dict] = []
        self.voice_signals: list[dict] = []
        self.word_signals: list[dict] = []
        self.personality_data: dict | None = None
        self.correlations: list[dict] = []
        self.flags: list[dict] = []
        self.participant_labels: set[str] = set()

        # Name mapping: "Person 1" -> "Anup Roy"
        self._name_map: dict[str, str] = {}
        self._confirmed_names: dict[str, str] = {}  # Persistent: once confirmed, never reverts
        self._all_known_full_names: set[str] = set()  # Every full name ever discovered (never shrinks)
        self._initials_map: dict[str, str] = {}
        self._last_name_rescan_ts: float = 0.0
        self._participants_locked: bool = False  # DETERMINISTIC: lock after stable name count
        self._last_name_scan_playback_ts: float = -999.0  # Last playback timestamp where name scan ran
        self._name_scan_stable_streak: int = 0  # Consecutive scans with same unique first-name count
        self._last_name_scan_count: int = 0  # Unique first-name count from last scan

        # Last known signal per participant (for heartbeat continuation)
        self._last_known_signal: dict | None = None

        # Speaking timeline: [{timestamp, speaker}] — from visual is_speaking detection
        self._speaking_timeline: list[dict] = []
        self.MAX_SPEAKING_TIMELINE = 200

        # Speaking duration tracking (backend is source of truth for speaking_pct)
        self._speaking_durations: dict[str, float] = {}  # canonical_name -> seconds
        self._total_speaking_time: float = 0.0
        self._last_speaking_ts: float = 0.0

        # Visual speaking counts — for live chart display only (noisy but immediate)
        self._visual_speaking_counts: dict[str, int] = {}
        self._visual_speaking_total: int = 0

        # Speaker diarization state (pyannote)
        self._diarization_segments: list = []
        self._diarization_ready = asyncio.Event()
        self._diarization_speaker_map: dict[str, str] = {}  # "SPEAKER_00" -> real name
        self._last_diarization_remap_ts: float = 0.0
        self._diarization_speaking_set = False  # True once diarization computes speaking durations

        # Audio analysis + engagement engine
        self.signal_bus = SignalBus()
        self.engagement_engine = EngagementEngine()
        self.alert_detector = EngagementAlertDetector(
            threshold=float(os.getenv("ENGAGEMENT_ALERT_THRESHOLD", "35")),
            min_duration=float(os.getenv("ENGAGEMENT_ALERT_DURATION", "60")),
        )
        self._last_alert_ts: float = 0.0  # Cooldown for alerts
        self.audio_analyzer = AudioAnalyzer()
        self.audio_features: list[dict] = []

        # Periodic analysis timing
        self._last_personality_ts = 0.0
        self._last_correlation_ts = 0.0

        # Lifecycle flags
        self._cancelled = False
        self._finalizing = False

        # First frame detection task
        self._detect_task: asyncio.Task | None = None

        # Frame pre-buffer: {timestamp: frame_path}
        self._frame_buffer: dict[float, str] = {}
        self._frame_buffer_task: asyncio.Task | None = None

    # Timestamps to scan for participant gallery — covers full meeting duration
    # Dense at start (gallery appears early), then periodic throughout for late joiners
    NAME_SCAN_TIMESTAMPS = [3.0, 5.0, 10.0, 15.0, 25.0, 40.0, 60.0, 90.0, 120.0, 180.0, 240.0, 360.0, 480.0, 600.0, 900.0, 1200.0, 1800.0]

    def _create_tracked_task(self, coro) -> asyncio.Task:
        """Create a task tracked in _tracked_tasks with auto-removal on completion."""
        task = asyncio.create_task(coro)
        self._tracked_tasks.add(task)
        task.add_done_callback(self._tracked_tasks.discard)
        return task

    async def initialize(self):
        """Called on WebSocket connect. Start audio extraction only.
        Name detection runs in parallel with frame analysis during playback — zero pre-fetching."""
        logger.info(f"[{self.video_id}] Session initialized")
        await self._send_status("Loading analysis...")
        # Start audio extraction + pre-transcription only — no name pre-fetching
        self._audio_task = self._create_tracked_task(self._extract_audio_background())
        self._create_tracked_task(self._pre_transcribe_first_chunk())
        # Names detected via extract_names_from_frame during playback (parallel, grow-only)
        await self._send_status("Ready — press play")

    async def _detect_participants_and_names(self):
        """Analyze frame at 5s (skip splash screen) to detect participants AND extract names.
        Only extracts names here — does NOT send signals (those come on play/tick)."""
        try:
            # Use 5s instead of 0s — frame 0 is often a splash/title screen with no participants
            detect_ts = 5.0
            frame_path = await extract_single_frame(self.video_path, detect_ts, self.output_dir)

            # Only run name extraction at init (NOT signals — those come during playback)
            name_map = await extract_names_from_frame(frame_path)

            # Handle name extraction result
            if isinstance(name_map, dict) and name_map:
                self._name_map = {k: self._clean_name(v) for k, v in name_map.items() if self._clean_name(v)}
                # Store initial names — these can be corrected by later scans
                self._confirmed_names.update(self._name_map)
                # Accumulate multi-word names for surname resolution
                for v in self._name_map.values():
                    if ' ' in v:
                        self._all_known_full_names.add(v)
                self._build_initials_map()
                self.participant_labels = set(self._name_map.values())
                logger.info(f"[{self.video_id}] Extracted {len(self._name_map)} names at {detect_ts}s: {self._name_map}")
                # Send first-name-only to frontend for display
                display_map = {k: self._first_name(v) for k, v in self._name_map.items()}
                await self._send_json({"type": "name_map", "data": display_map})
            elif isinstance(name_map, Exception):
                logger.warning(f"[{self.video_id}] Name extraction at {detect_ts}s failed: {name_map}")

            await self._send_status("Ready — press play")

            try:
                os.remove(frame_path)
            except OSError:
                pass

        except Exception as e:
            logger.error(f"[{self.video_id}] Detection failed: {e}")
            await self._send_status("Ready — press play")

    async def _progressive_name_scan(self):
        """Background progressive scan: try multiple frames to find ALL participant names.

        Scans at fixed timestamps PLUS dynamically computed ones based on video duration.
        This catches late joiners who appear mid-meeting, not just at the start.
        """
        try:
            # Wait briefly for initial detection to complete first
            await asyncio.sleep(3.0)

            logger.info(f"[{self.video_id}] Starting progressive name scan (current: {len(self._name_map)} names)")

            # Get video duration to compute dynamic scan points
            duration = 0.0
            try:
                duration = await get_video_duration(self.video_path)
            except Exception:
                pass

            # Build scan timestamps: fixed early ones + every 60s throughout the video
            scan_timestamps = list(self.NAME_SCAN_TIMESTAMPS)
            if duration > 0:
                # Add a scan every 60s throughout the entire video
                t = 60.0
                while t < duration - 5.0:
                    if t not in scan_timestamps:
                        scan_timestamps.append(t)
                    t += 60.0
                scan_timestamps.sort()
                # Remove timestamps beyond video duration
                scan_timestamps = [t for t in scan_timestamps if t < duration - 2.0]

            for ts in scan_timestamps:
                if self._cancelled:
                    break

                # Skip timestamps we already scanned in initial detection
                if ts == 5.0 and self._name_map:
                    continue

                try:
                    frame_path = await extract_single_frame(self.video_path, ts, self.output_dir)
                    name_map = await extract_names_from_frame(frame_path)

                    try:
                        os.remove(frame_path)
                    except OSError:
                        pass

                    if not isinstance(name_map, dict) or not name_map:
                        continue

                    logger.info(f"[{self.video_id}] Name scan at {ts}s: found {len(name_map)} names")
                    await self._merge_names(name_map)

                except Exception as e:
                    logger.warning(f"[{self.video_id}] Name scan at {ts}s failed: {e}")
                    continue

            # LOCK — participant list is now FINAL. No new participants after this.
            # Late joiners are already covered because scan runs every 60s through the whole video.
            self._participants_locked = True
            unique_first = set(self._first_name(v) for v in self._name_map.values())
            logger.info(f"[{self.video_id}] Progressive scan complete. LOCKED {len(unique_first)} participants: {unique_first}")

        except Exception as e:
            logger.error(f"[{self.video_id}] Progressive name scan error: {e}")
            self._participants_locked = True

    async def _incremental_name_scan(self, timestamp: float):
        """Extract names from the CURRENT playback frame. Works for both uploaded videos and live calls.
        Runs periodically during playback (every NAME_SCAN_PLAYBACK_INTERVAL seconds).
        Never locks during playback — late joiners can appear at any time.
        Lock only happens at finalize (video end)."""
        try:
            frame_path = await extract_single_frame(self.video_path, timestamp, self.output_dir)
            name_map = await extract_names_from_frame(frame_path)

            try:
                os.remove(frame_path)
            except OSError:
                pass

            if not isinstance(name_map, dict) or not name_map:
                return

            unique_before = len(set(self._first_name(v) for v in self._name_map.values()))
            logger.info(f"[{self.video_id}] Live name scan at {timestamp:.0f}s: found {len(name_map)} names")
            await self._merge_names(name_map)
            unique_after = len(set(self._first_name(v) for v in self._name_map.values()))

            if unique_after > unique_before:
                logger.info(f"[{self.video_id}] New participant detected at {timestamp:.0f}s! "
                            f"{unique_before} -> {unique_after}")

        except Exception as e:
            logger.warning(f"[{self.video_id}] Live name scan at {timestamp:.0f}s failed: {e}")

    def _resolve_to_known_full_name(self, clean: str) -> str:
        """If 'clean' is a standalone surname or partial name that matches a
        previously-discovered full name, return the full name. Otherwise return clean as-is.

        This prevents gallery reshuffles from losing name info — once we've seen
        "Sarang Zendehrooh", a later OCR read of just "Zendehrooh" resolves back to it.
        """
        if not self._all_known_full_names or ' ' in clean:
            # Already multi-word or no history — skip
            return clean
        clean_lower = clean.lower()
        for full in self._all_known_full_names:
            if ' ' not in full:
                continue  # Only match against multi-word full names
            full_words = [w.lower() for w in full.split() if len(w) >= 3]
            if clean_lower in full_words:
                return full
            # Shared prefix check (OCR truncation: "Zendher" → "Zendehrooh")
            for fw in full_words:
                if len(clean_lower) >= 5 and len(fw) >= 5:
                    prefix_len = 0
                    for i in range(min(len(clean_lower), len(fw))):
                        if clean_lower[i] == fw[i]:
                            prefix_len += 1
                        else:
                            break
                    if prefix_len >= 4:
                        return full
            # Edit distance for OCR typos
            for fw in full_words:
                if len(clean_lower) >= 6 and len(fw) >= 6 and abs(len(clean_lower) - len(fw)) <= 2:
                    if self._edit_distance(clean_lower, fw) <= 2:
                        return full
        return clean

    @staticmethod
    def _better_name(a: str, b: str) -> str:
        """Pick the higher-quality name variant between two same-person names.

        Prefers: more real words (>= 3 chars), then longer total length.
        Never picks a name with single/two-char initials over a clean name.
        """
        a_words = [w for w in a.split() if len(w) >= 3]
        b_words = [w for w in b.split() if len(w) >= 3]
        a_initials = sum(1 for w in a.split() if len(w) <= 2)
        b_initials = sum(1 for w in b.split() if len(w) <= 2)

        # Penalize names with initials — prefer clean names
        if a_initials > 0 and b_initials == 0:
            return b
        if b_initials > 0 and a_initials == 0:
            return a

        # More real words wins
        if len(a_words) > len(b_words):
            return a
        if len(b_words) > len(a_words):
            return b

        # Same word count — prefer longer total (more complete OCR read)
        return a if len(a) >= len(b) else b

    async def _merge_names(self, name_map: dict[str, str]):
        """Merge newly discovered names into the participant registry.

        ACCUMULATE-ONLY design:
        - _all_known_full_names is the source of truth (set of all unique full names)
        - _name_map tracks position → name for frontend display
        - Position keys are volatile (gallery reshuffles), so we NEVER overwrite
          a position that already has a confirmed name with a DIFFERENT person.
        - Only same-person updates (longer name variant) are allowed.
        - After lock, no new participants can be added.
        """
        changed = False

        for position_key, raw_name in name_map.items():
            clean = self._clean_name(raw_name)
            if not clean:
                continue

            # Resolve standalone surnames to previously-known full names
            clean = self._resolve_to_known_full_name(clean)

            current = self._name_map.get(position_key)

            # Same name at same position — no change
            if current == clean:
                continue

            # Check if this name matches the CURRENT value at this position
            # (same person, possibly better OCR read → update to longer name)
            if current and self._names_are_same_person(clean, current):
                winner = self._better_name(clean, current)
                if winner != current:
                    self._name_map[position_key] = winner
                    self._confirmed_names[position_key] = winner
                    changed = True
                if ' ' in winner:
                    self._all_known_full_names.add(winner)
                continue

            # Different person at this position (gallery reshuffle).
            # DON'T overwrite — the original person is still valid.
            # Just accumulate the new name if it's genuinely new.
            if current:
                # Check if this name already exists at another position
                already_known = any(
                    self._names_are_same_person(clean, v)
                    for k, v in self._name_map.items() if k != position_key
                )
                if already_known:
                    # Name exists elsewhere → update existing entry with longer variant
                    for k, v in list(self._name_map.items()):
                        if k != position_key and self._names_are_same_person(clean, v):
                            winner = self._better_name(clean, v)
                            if winner != v:
                                self._name_map[k] = winner
                                self._confirmed_names[k] = winner
                                changed = True
                            if ' ' in winner:
                                self._all_known_full_names.add(winner)
                            break
                    continue

                # Genuinely new name — REJECT if locked, ADD if unlocked
                if self._participants_locked:
                    continue  # Silently ignore OCR noise after lock
                new_key = f"Person {len(self._name_map) + 1}"
                while new_key in self._name_map:
                    new_key = f"Person {int(new_key.split()[-1]) + 1}"
                self._name_map[new_key] = clean
                self._confirmed_names[new_key] = clean
                if ' ' in clean:
                    self._all_known_full_names.add(clean)
                changed = True
                continue

            # Empty position — first time seeing this key
            already_known = any(
                self._names_are_same_person(clean, v)
                for v in self._name_map.values()
            )
            if already_known:
                for k, v in list(self._name_map.items()):
                    if self._names_are_same_person(clean, v):
                        winner = self._better_name(clean, v)
                        self._name_map[position_key] = winner
                        if winner != v:
                            self._name_map[k] = winner
                            self._confirmed_names[k] = winner
                        if ' ' in winner:
                            self._all_known_full_names.add(winner)
                        changed = True
                        break
                continue

            if self._participants_locked:
                continue  # Reject after lock

            # Truly new participant at new position
            self._name_map[position_key] = clean
            self._confirmed_names[position_key] = clean
            if ' ' in clean:
                self._all_known_full_names.add(clean)
            changed = True

        if not changed:
            return

        self._build_initials_map()
        self.participant_labels = set(self._name_map.values())
        unique_first = set(self._first_name(v) for v in self._name_map.values())
        logger.info(f"[{self.video_id}] Name map updated ({len(unique_first)} unique first names): {self._name_map}")
        # Send first-name-only to frontend for display
        display_map = {k: self._first_name(v) for k, v in self._name_map.items()}
        await self._send_json({"type": "name_map", "data": display_map})

        # Trigger diarization remap when participant list changes
        # so new participants get included in speaking distribution
        if self._diarization_segments:
            await self._map_diarization_to_names()
            # Recompute speaking distribution with updated name map
            if self._diarization_speaking_set:
                await self._recompute_speaking_distribution()

    async def _pre_transcribe_first_chunk(self):
        """Pre-transcribe the first 2s of audio during initialization.
        This way transcript appears instantly when user presses play."""
        try:
            await asyncio.sleep(0.5)  # Brief wait for ffmpeg to be ready
            chunk_path = await extract_audio_chunk_from_video(
                self.video_path, 0.0, 2.0, self.output_dir
            )
            transcript_data = await transcribe_audio(chunk_path)
            segments = transcript_data.get("segments", [])
            for seg in segments:
                text = seg.get("text", "").strip()
                if not text or text in self.WHISPER_HALLUCINATIONS:
                    continue
                if len(text.split()) <= 2 and text.lower().rstrip('.!?,') in (
                    "you", "bye", "bye-bye", "thank you", "thanks",
                ):
                    continue
                # Reject non-English text (CJK, Korean, Arabic, etc.)
                non_latin = sum(1 for c in text if ord(c) > 0x024F and not c.isspace() and c not in ".,!?;:'-\"()")
                if non_latin > len(text) * 0.3:
                    continue
                seg["start"] = seg.get("start", 0)
                seg["end"] = seg.get("end", 0)
                self._transcript_segments.append(seg)

            self._audio_transcribed_up_to = 2.0
            logger.info(f"[{self.video_id}] Pre-transcribed first 2s: {len(segments)} segments")
            try:
                os.remove(chunk_path)
            except OSError:
                pass
        except Exception as e:
            logger.warning(f"[{self.video_id}] Pre-transcription failed: {e}")

    async def _extract_audio_background(self):
        """Extract audio track from video."""
        try:
            logger.info(f"[{self.video_id}] Extracting audio track...")
            self._audio_path = await extract_full_audio(self.video_path, self.output_dir)
            if self._audio_path:
                self._audio_ready = True
                self._audio_extracted.set()
                logger.info(f"[{self.video_id}] Audio extraction complete")
                # Start speaker diarization in background
                self._create_tracked_task(self._run_diarization())
            else:
                self._audio_extracted.set()
                logger.warning(f"[{self.video_id}] No audio track found")
                await self._send_json({"type": "no_audio"})
        except Exception as e:
            logger.error(f"[{self.video_id}] Audio extraction failed: {e}")
            self._audio_extracted.set()

    async def _run_diarization(self):
        """Run pyannote speaker diarization on the full audio track.
        After diarization, compute speaking distribution directly from segments
        (most accurate source — replaces unreliable visual is_speaking detection).
        """
        try:
            if not self._audio_path:
                self._diarization_ready.set()
                return
            logger.info(f"[{self.video_id}] Running speaker diarization...")
            await self._send_status("Identifying speakers...")
            segments = await diarize_audio(self._audio_path)
            if not segments:
                logger.warning(f"[{self.video_id}] Diarization returned no segments")
                self._diarization_ready.set()
                return
            self._diarization_segments = segments
            unique_speakers = sorted(set(s.speaker for s in segments))
            logger.info(f"[{self.video_id}] Diarization complete: {len(segments)} segments, {len(unique_speakers)} speakers")
            await self._map_diarization_to_names()

            # ── Compute speaking distribution from diarization (SOURCE OF TRUTH) ──
            # This replaces the noisy visual is_speaking-based accumulation
            dia_durations: dict[str, float] = {}
            for seg in segments:
                dur = max(0, seg.end - seg.start)
                dia_durations[seg.speaker] = dia_durations.get(seg.speaker, 0) + dur

            # Convert to first-name-based durations using the speaker map
            new_speaking_durations: dict[str, float] = {}
            new_total = 0.0
            for dia_speaker, dur in dia_durations.items():
                real_name = self._diarization_speaker_map.get(dia_speaker, "")
                if real_name and not re.match(r'^Person \d+$', real_name):
                    display = self._first_name(real_name)
                    new_speaking_durations[display] = new_speaking_durations.get(display, 0) + dur
                    new_total += dur
                else:
                    # Unmapped speaker — use generic label
                    generic = f"Speaker {dia_speaker.replace('SPEAKER_', '')}"
                    new_speaking_durations[generic] = new_speaking_durations.get(generic, 0) + dur
                    new_total += dur

            # Overwrite speaking durations with diarization data
            if new_total > 0:
                self._speaking_durations = new_speaking_durations
                self._total_speaking_time = new_total
                self._diarization_speaking_set = True  # Lock — visual/transcript won't overwrite
                logger.info(f"[{self.video_id}] Speaking distribution from diarization: {new_speaking_durations}")

                # Build and send speaking distribution to frontend immediately
                speaking_dist: dict[str, float] = {}
                for name, dur in self._speaking_durations.items():
                    speaking_dist[name] = round((dur / self._total_speaking_time) * 100, 1)

                # Update participant speaking_pct via a signals message
                if not self._cancelled and not self._finalizing:
                    await self._send_json({
                        "type": "signals",
                        "data": {
                            "timestamp": 0,
                            "participants": [],
                            "speaking_distribution": speaking_dist,
                            "is_heartbeat": True,
                        },
                    })

            self._diarization_ready.set()
        except Exception as e:
            logger.error(f"[{self.video_id}] Diarization failed: {e}")
            self._diarization_ready.set()

    async def _map_diarization_to_names(self):
        """Map diarization speaker labels (SPEAKER_00, etc.) to real participant names.

        Strategy:
        1. If visual speaking_timeline has data → use voting to map
           - Only assign when a name has > 50% of votes for that diarization speaker
           - Prevents incorrect mappings from noisy visual detection
        2. Unmapped speakers stay as "Speaker N" — NO arbitrary rank-based assignment
        3. Generic "Speaker N" labels also mapped for transcript attribution
        """
        if not self._diarization_segments:
            return

        # Strategy 1: Visual speaking timeline voting (if available)
        if self._speaking_timeline:
            speaker_ranges: dict[str, list[tuple[float, float]]] = {}
            for seg in self._diarization_segments:
                speaker_ranges.setdefault(seg.speaker, []).append((seg.start, seg.end))

            # Prevent one-to-many: track which names have been assigned
            assigned_names: set[str] = set()

            for dia_speaker, ranges in speaker_ranges.items():
                name_votes: dict[str, float] = {}
                total_votes = 0
                for start, end in ranges:
                    for vt in self._speaking_timeline:
                        ts = vt.get("timestamp", 0)
                        if start <= ts <= end:
                            name = vt.get("speaker", "")
                            if name and not re.match(r'^Person \d+$', name):
                                name_votes[name] = name_votes.get(name, 0) + 1
                                total_votes += 1
                if name_votes and total_votes >= 2:
                    best_name = max(name_votes, key=name_votes.get)
                    best_count = name_votes[best_name]
                    # Only assign if clear majority (> 50% of votes)
                    # AND this name hasn't been assigned to another speaker already
                    if best_count > total_votes * 0.5 and best_name not in assigned_names:
                        self._diarization_speaker_map[dia_speaker] = best_name
                        assigned_names.add(best_name)

        # NO rank-based fallback — unmapped speakers stay as "Speaker N"

        # Create generic "Speaker N" → real name mapping for transcript attribution
        unique_speakers = sorted(set(s.speaker for s in self._diarization_segments))
        for i, sp in enumerate(unique_speakers):
            generic_label = f"Speaker {i + 1}"
            if sp in self._diarization_speaker_map:
                self._diarization_speaker_map[generic_label] = self._diarization_speaker_map[sp]

        logger.info(f"[{self.video_id}] Diarization speaker map: {self._diarization_speaker_map}")

    async def _recompute_speaking_distribution(self):
        """Recompute speaking durations from diarization using the latest speaker map.

        Called when names change (late joiner detected) so their speaking time
        gets attributed correctly instead of staying as "Speaker N".
        """
        if not self._diarization_segments:
            return

        dia_durations: dict[str, float] = {}
        for seg in self._diarization_segments:
            dur = max(0, seg.end - seg.start)
            dia_durations[seg.speaker] = dia_durations.get(seg.speaker, 0) + dur

        new_speaking_durations: dict[str, float] = {}
        new_total = 0.0
        for dia_speaker, dur in dia_durations.items():
            real_name = self._diarization_speaker_map.get(dia_speaker, "")
            if real_name and not re.match(r'^Person \d+$', real_name):
                display = self._first_name(real_name)
                new_speaking_durations[display] = new_speaking_durations.get(display, 0) + dur
                new_total += dur
            else:
                generic = f"Speaker {dia_speaker.replace('SPEAKER_', '')}"
                new_speaking_durations[generic] = new_speaking_durations.get(generic, 0) + dur
                new_total += dur

        if new_total > 0:
            self._speaking_durations = new_speaking_durations
            self._total_speaking_time = new_total
            logger.info(f"[{self.video_id}] Recomputed speaking distribution: {new_speaking_durations}")

            # Send updated distribution to frontend
            speaking_dist: dict[str, float] = {}
            for name, dur in self._speaking_durations.items():
                speaking_dist[name] = round((dur / self._total_speaking_time) * 100, 1)

            if not self._cancelled and not self._finalizing:
                await self._send_json({
                    "type": "signals",
                    "data": {
                        "timestamp": 0,
                        "participants": [],
                        "speaking_distribution": speaking_dist,
                        "is_heartbeat": True,
                    },
                })

    # ─── Playback handlers ────────────────────────────────────────

    async def handle_play(self):
        """Video started playing. Detection already started in initialize()."""
        self.is_playing = True
        logger.info(f"[{self.video_id}] PLAY")

        # If initialize() somehow didn't start tasks, do it now as fallback
        if self._audio_task is None:
            self._audio_task = self._create_tracked_task(self._extract_audio_background())
            self._detect_task = self._create_tracked_task(self._detect_participants_and_names())

        await self._send_status("Live analysis active")

    async def handle_pause(self):
        self.is_playing = False
        logger.info(f"[{self.video_id}] PAUSE")
        await self._send_status("Paused — press play to resume")

    async def handle_seek(self, timestamp: float):
        """Reset throttle so analysis resumes at new position. Invalidates frame buffer."""
        self.last_analyzed_ts = timestamp - self.FRAME_INTERVAL - 1
        self._last_heartbeat_ts = timestamp - 1
        self._next_segment_idx = 0
        for i, seg in enumerate(self._transcript_segments):
            if seg.get("start", 0) > timestamp:
                self._next_segment_idx = i
                break
        else:
            self._next_segment_idx = len(self._transcript_segments)
        # Invalidate frame buffer — clean up old pre-extracted frames
        for path in self._frame_buffer.values():
            try:
                os.remove(path)
            except OSError:
                pass
        self._frame_buffer.clear()
        # Pre-fill buffer from new position
        self._create_tracked_task(self._prefill_frame_buffer(timestamp))
        logger.info(f"[{self.video_id}] SEEK to {timestamp:.1f}s")
        await self._send_status(f"Seeked to {timestamp:.1f}s")

    async def handle_tick(self, timestamp: float):
        """Called on every video timeupdate (~250ms intervals).

        1. Backpressure check — skip if too many tasks in flight
        2. Send heartbeat signal every 1s (keeps engagement chart flowing)
        3. Deliver pending transcript segments
        4. Transcribe new audio chunks
        5. Launch live frame analysis when due
        6. Trigger periodic personality/correlation analysis
        7. Maintain frame pre-buffer
        """
        if not self.is_playing:
            return

        # 1. BACKPRESSURE — if too many tasks, skip this tick (show last-known data)
        if len(self._tracked_tasks) > self.MAX_TRACKED_TASKS:
            # Still send heartbeat so chart doesn't freeze
            if timestamp - self._last_heartbeat_ts >= self.HEARTBEAT_INTERVAL:
                self._last_heartbeat_ts = timestamp
                await self._send_heartbeat(timestamp)
            return

        # 2. HEARTBEAT — send engagement update every 1s to keep chart flowing
        if timestamp - self._last_heartbeat_ts >= self.HEARTBEAT_INTERVAL:
            self._last_heartbeat_ts = timestamp
            await self._send_heartbeat(timestamp)

        # 3. Deliver pending transcript segments
        await self._send_pending_transcript(timestamp)

        # 4. Transcription — parallel workers, never drops audio
        if timestamp - self._audio_transcribed_up_to >= self.AUDIO_MIN_CHUNK:
            self._transcribe_queue = timestamp
            if self._active_transcribe_count < self.MAX_PARALLEL_TRANSCRIBE:
                task = self._create_tracked_task(self._transcription_worker())
                self._transcribe_tasks.append(task)
                task.add_done_callback(lambda t: self._transcribe_tasks.remove(t) if t in self._transcribe_tasks else None)

        # 5. Live frame analysis — launch when due
        if timestamp - self.last_analyzed_ts >= self.FRAME_INTERVAL:
            if self._semaphore._value > 0:  # noqa: SLF001
                self.last_analyzed_ts = timestamp
                self._create_tracked_task(self._analyze_frame_live(timestamp))

        # 6. Name detection is piggybacked on frame analysis (step 5) — no separate scan needed

        # 7. Periodic deeper analysis
        if (timestamp - self._last_personality_ts >= self.PERSONALITY_INTERVAL
                and len(self.signal_snapshots) >= 3):
            self._last_personality_ts = timestamp
            self._create_tracked_task(self._analyze_personality_periodic())

        if (timestamp - self._last_correlation_ts >= self.CORRELATION_INTERVAL
                and len(self.signal_snapshots) >= 4):
            self._last_correlation_ts = timestamp
            self._create_tracked_task(self._build_correlations_periodic())

        # Re-map diarization speakers as more visual data arrives (every 15s)
        if (self._diarization_segments
                and self._speaking_timeline
                and timestamp - self._last_diarization_remap_ts >= 15.0):
            self._last_diarization_remap_ts = timestamp
            await self._map_diarization_to_names()
            if self._diarization_speaking_set:
                await self._recompute_speaking_distribution()

        # 7. Maintain frame pre-buffer
        self._create_tracked_task(self._prefill_frame_buffer(timestamp))

    async def _send_heartbeat(self, timestamp: float):
        """Send a signal carrying forward last known engagement with natural variation.
        This makes the engagement chart flow continuously every second.
        The is_heartbeat flag tells the frontend to use this for the chart
        but NOT for gesture/reaction/body-language signal panels.
        """
        if not self._last_known_signal or self._cancelled or self._finalizing:
            return

        # Carry forward exact last-known engagement values (no artificial noise)
        heartbeat = {
            "timestamp": timestamp,
            "participants": self._last_known_signal.get("participants", []),
            "is_heartbeat": True,
        }
        # Include speaking distribution so frontend always gets latest percentages
        if self._total_speaking_time > 0:
            speaking_dist: dict[str, float] = {}
            for name, dur in self._speaking_durations.items():
                speaking_dist[name] = round((dur / self._total_speaking_time) * 100, 1)
            heartbeat["speaking_distribution"] = speaking_dist
        await self._send_json({"type": "signals", "data": heartbeat})

    # ─── Frame analysis (real-time) ───────────────────────────────

    async def _prefill_frame_buffer(self, current_ts: float):
        """Pre-extract frames ahead of playback position."""
        for offset in range(1, self.FRAME_BUFFER_AHEAD + 1):
            target_ts = round(current_ts + offset * self.FRAME_INTERVAL, 1)
            if target_ts in self._frame_buffer:
                continue
            try:
                path = await extract_single_frame(self.video_path, target_ts, self.output_dir)
                self._frame_buffer[target_ts] = path
            except Exception:
                pass  # Non-critical — will extract on demand

        # Evict stale entries (behind playback)
        stale = [ts for ts in self._frame_buffer if ts < current_ts - 2.0]
        for ts in stale:
            try:
                os.remove(self._frame_buffer.pop(ts))
            except (OSError, KeyError):
                pass

    async def _analyze_frame_live(self, timestamp: float):
        """Extract and analyze a single frame. Sends results as soon as API returns."""
        frame_path = None
        async with self._semaphore:
            if self._cancelled:
                return
            try:
                logger.info(f"[{self.video_id}] Analyzing frame at {timestamp:.1f}s...")

                # Check frame buffer first — skip ffmpeg if pre-extracted
                rounded_ts = round(timestamp, 1)
                if rounded_ts in self._frame_buffer:
                    frame_path = self._frame_buffer.pop(rounded_ts)
                else:
                    frame_path = await extract_single_frame(
                        self.video_path, timestamp, self.output_dir
                    )

                vision_model = _next_vision_model()

                # Build the known names list (first names only) to pass into signal analysis
                # This tells the vision model to use first names instead of "Person N"
                known_names_list = sorted(set(self._first_name(v) for v in self._name_map.values())) if self._name_map else None

                # Parallel: frame analysis + name extraction (separate API calls)
                tasks = [analyze_frame_all_signals(
                    frame_path, timestamp,
                    model_override=vision_model,
                    known_names=known_names_list,
                )]

                # Run name extraction alongside frame analysis every NAME_RESCAN_INTERVAL
                do_name_rescan = (timestamp - self._last_name_rescan_ts >= self.NAME_RESCAN_INTERVAL)
                if do_name_rescan:
                    tasks.append(extract_names_from_frame(frame_path, model=vision_model))
                    self._last_name_rescan_ts = timestamp

                results = await asyncio.gather(*tasks, return_exceptions=True)
                result = results[0] if not isinstance(results[0], Exception) else None

                # Merge names from the parallel extraction call
                if do_name_rescan and len(results) > 1 and isinstance(results[1], dict) and results[1]:
                    await self._merge_names(results[1])

                if result is None:
                    logger.error(f"[{self.video_id}] Frame analysis failed: {results[0]}")
                    return

                result["timestamp"] = timestamp

                # ── The LOCKED participant list comes from _name_map ONLY ──
                # Use FIRST NAMES as canonical identifiers for display.
                # _name_map is deterministic after lock; _all_known_full_names has garbled intermediates.
                all_full_names = set(self._name_map.values())
                canonical_names = sorted(set(self._first_name(v) for v in all_full_names if v)) if all_full_names else []
                raw_participants = result.get("participants", [])

                # Match each vision result to a canonical first name
                matched: dict[str, dict] = {}  # first_name -> vision data
                for p in raw_participants:
                    raw = p.get("label", "").strip()
                    # Convert to first name, then match against canonical first names
                    raw_first = self._first_name(raw) if raw else raw
                    resolved = self._to_canonical(raw_first, canonical_names)

                    # If resolved to "Person N", try direct name_map lookup
                    if re.match(r'^Person \d+$', resolved):
                        mapped = self._name_map.get(resolved) or self._name_map.get(raw)
                        if mapped:
                            resolved = self._first_name(mapped)
                        elif canonical_names:
                            # Have canonical names but can't resolve this label —
                            # skip it to prevent OCR noise from inflating count.
                            # Real participants are detected via name_map scans.
                            continue

                    # Dedup: if this canonical name was already matched, merge
                    if resolved in matched:
                        ex = matched[resolved]
                        if p.get("emotions", {}).get("engagement", 0) > ex.get("emotions", {}).get("engagement", 0):
                            ex["emotions"] = p.get("emotions", ex.get("emotions"))
                            ex["body_language"] = p.get("body_language", ex.get("body_language"))
                        ex["gestures"] = list(set(ex.get("gestures", []) + p.get("gestures", [])))[:4]
                        ex["reactions"] = list(set(ex.get("reactions", []) + p.get("reactions", [])))[:4]
                        if p.get("is_speaking"):
                            ex["is_speaking"] = True
                    else:
                        p["label"] = resolved
                        matched[resolved] = p

                # Build final list: ONLY participants detected in THIS frame
                # When canonical names ARE available, only send resolved names.
                # When canonical names are NOT yet available, use "Person N" labels
                # (never send raw vision model labels — they create duplicates).
                final_participants: list[dict] = []
                if matched:
                    final_participants = list(matched.values())
                elif not canonical_names:
                    # No name scan yet — use generic "Person N" labels
                    for idx, p in enumerate(raw_participants, 1):
                        p["label"] = f"Person {idx}"
                    final_participants = raw_participants

                # Convert ALL participant labels to first-name-only for display
                # (full names kept in _name_map for DB persistence)
                for p in final_participants:
                    p["label"] = self._first_name(p["label"])

                # Dedup after first-name conversion (two full names → same first name)
                deduped: dict[str, dict] = {}
                for p in final_participants:
                    lbl = p["label"]
                    if lbl in deduped:
                        # Merge: keep higher engagement
                        if p.get("emotions", {}).get("engagement", 0) > deduped[lbl].get("emotions", {}).get("engagement", 0):
                            deduped[lbl]["emotions"] = p.get("emotions", deduped[lbl].get("emotions"))
                        if p.get("is_speaking"):
                            deduped[lbl]["is_speaking"] = True
                    else:
                        deduped[lbl] = p
                final_participants = list(deduped.values())

                # Inject camera-off participants: anyone in _name_map but NOT in frame analysis
                # They're in the meeting — just not visually analyzable.
                if canonical_names:
                    seen = set(p["label"] for p in final_participants)
                    for cn in canonical_names:
                        if cn not in seen:
                            final_participants.append({
                                "label": cn,
                                "is_speaking": False,
                                "emotions": {"primary": "passive", "confidence": 0.3, "engagement": 30},
                                "body_language": {"posture": "upright", "openness": "mixed", "leaning": "neutral", "mirroring": False, "description": "camera off"},
                                "gestures": ["none visible"],
                                "reactions": ["none visible"],
                            })

                result["participants"] = final_participants

                # Speaker detection — enforce exactly ONE speaker
                speakers = [p for p in final_participants if p.get("is_speaking")]
                if len(speakers) > 1:
                    speakers.sort(key=lambda x: x.get("emotions", {}).get("engagement", 0), reverse=True)
                    for extra in speakers[1:]:
                        extra["is_speaking"] = False

                for p in final_participants:
                    if p.get("is_speaking"):
                        eng = p.get("emotions", {}).get("engagement", 50)
                        if eng < 65:
                            p["emotions"]["engagement"] = max(65, eng)
                        self._speaking_timeline.append({
                            "timestamp": timestamp,
                            "speaker": p["label"],
                        })
                        if len(self._speaking_timeline) > self.MAX_SPEAKING_TIMELINE:
                            self._speaking_timeline = self._speaking_timeline[-self.MAX_SPEAKING_TIMELINE:]

                        # Track visual speaking for LIVE distribution display
                        # (noisy but only signal available during playback)
                        self._visual_speaking_counts[p["label"]] = self._visual_speaking_counts.get(p["label"], 0) + 1
                        self._visual_speaking_total += 1

                self._last_speaking_ts = timestamp

                # Build speaking distribution for live display
                # Priority: transcript-based (accurate) > visual detection (noisy but immediate)
                speaking_dist: dict[str, float] = {}
                if self._total_speaking_time > 0:
                    # Use transcript-based distribution when available
                    for name, dur in self._speaking_durations.items():
                        speaking_dist[name] = round((dur / self._total_speaking_time) * 100, 1)
                elif self._visual_speaking_total > 0:
                    # Fallback: visual is_speaking counts for live chart
                    for name, cnt in self._visual_speaking_counts.items():
                        speaking_dist[name] = round((cnt / self._visual_speaking_total) * 100, 1)

                # Inject speaking_pct into each participant
                for p in final_participants:
                    speaker_name = p["label"]
                    p["speaking_pct"] = speaking_dist.get(speaker_name, 0.0)

                result["speaking_distribution"] = speaking_dist

                # State cap
                self.signal_snapshots.append(result)
                if len(self.signal_snapshots) > self.MAX_SIGNAL_SNAPSHOTS:
                    self.signal_snapshots = self.signal_snapshots[-self.MAX_SIGNAL_SNAPSHOTS:]

                self._last_known_signal = result

                if not self._cancelled and not self._finalizing:
                    await self._send_json({"type": "signals", "data": result})

                    for p in final_participants:
                        label = p.get("label", "")
                        visual_signal = SignalEvent(
                            signal_type=SignalType.VISUAL,
                            participant_id=label,
                            timestamp=timestamp,
                            data={"engagement": p.get("emotions", {}).get("engagement", 50)},
                        )
                        audio_signal = None
                        if self.audio_features:
                            latest_audio = self.audio_features[-1]
                            audio_signal = SignalEvent(
                                signal_type=SignalType.AUDIO,
                                participant_id=label,
                                timestamp=timestamp,
                                data={"energy": latest_audio.get("energy", 50)},
                            )

                        self.engagement_engine.compute(
                            participant_id=label,
                            timestamp=timestamp,
                            visual_signal=visual_signal,
                            audio_signal=audio_signal,
                        )

                logger.info(f"[{self.video_id}] Frame {timestamp:.1f}s: "
                            f"{len(final_participants)}/{len(canonical_names) or '?'} participants "
                            f"({', '.join(p['label'] for p in final_participants)}), "
                            f"speaker={'→'.join(p['label'] for p in final_participants if p.get('is_speaking')) or 'none'}")

            except Exception as e:
                logger.error(f"[{self.video_id}] Frame analysis at {timestamp:.1f}s failed: {e}")
            finally:
                if frame_path:
                    try:
                        os.remove(frame_path)
                    except OSError:
                        pass

    # ─── Audio transcription (real-time streaming) ────────────────

    async def _transcription_worker(self, force: bool = False):
        """Parallel transcription worker. Claims a range under lock, transcribes, releases.

        Multiple workers can run concurrently on non-overlapping ranges.
        """
        self._active_transcribe_count += 1
        min_chunk = 0.2 if force else self.AUDIO_WORKER_MIN
        try:
            while not self._cancelled:
                # Claim a range under lock
                async with self._transcribe_lock:
                    target = self._transcribe_queue
                    start = self._audio_transcribed_up_to

                    if target - start < min_chunk:
                        break  # Nothing worth transcribing

                    end = min(start + self.AUDIO_CHUNK_SECONDS, target)
                    self._audio_transcribed_up_to = end  # Advance so next worker takes next range

                logger.info(f"[{self.video_id}] Transcribing [{start:.1f}-{end:.1f}]...")

                try:
                    if self._audio_ready and self._audio_path:
                        chunk_path = await extract_audio_chunk(
                            self._audio_path, start, end, self.output_dir
                        )
                    else:
                        chunk_path = await extract_audio_chunk_from_video(
                            self.video_path, start, end, self.output_dir
                        )

                    # Run transcription and audio analysis in parallel
                    transcript_coro = transcribe_audio(chunk_path)
                    audio_coro = self._analyze_audio_features(chunk_path, start, end)
                    transcript_data, _ = await asyncio.gather(
                        transcript_coro, audio_coro, return_exceptions=True
                    )

                    if isinstance(transcript_data, Exception):
                        logger.error(f"[{self.video_id}] Transcription failed [{start:.1f}-{end:.1f}]: {transcript_data}")
                        continue

                    segments = transcript_data.get("segments", [])
                    accepted = 0

                    # Collect all text for batch cleanup
                    raw_texts = []
                    valid_segments = []
                    for seg in segments:
                        text = seg.get("text", "").strip()
                        if not text:
                            continue
                        if text in self.WHISPER_HALLUCINATIONS:
                            continue
                        if len(text.split()) <= 2 and text.lower().rstrip('.!?,') in (
                            "you", "bye", "bye-bye", "thank you", "thanks", "hmm",
                            "uh", "um", "oh", "ah", "huh", "subscribe",
                        ):
                            continue
                        # Reject non-English text (CJK, Korean, Arabic, etc.)
                        non_latin_count = sum(1 for c in text if ord(c) > 0x024F and not c.isspace() and c not in ".,!?;:'-\"()")
                        if non_latin_count > len(text) * 0.3:
                            logger.info(f"[{self.video_id}] Rejected non-English transcript: '{text[:40]}'")
                            continue
                        raw_texts.append(text)
                        valid_segments.append(seg)

                    # Multi-model cleanup: verify transcription with text model
                    if raw_texts:
                        combined_raw = " ".join(raw_texts)
                        cleaned = await cleanup_transcription(combined_raw, (start + end) / 2)
                        # If cleanup returned something meaningful, use it
                        if cleaned and len(cleaned) > 3 and cleaned != combined_raw:
                            logger.info(f"[{self.video_id}] Transcription cleaned: '{combined_raw[:60]}' → '{cleaned[:60]}'")

                    for seg in valid_segments:
                        seg_ts = seg.get("start", 0) + start
                        seg["start"] = seg_ts
                        seg["end"] = seg.get("end", 0) + start

                        # PRIORITY 1: Audio diarization (most reliable)
                        if self._diarization_segments:
                            assigned = assign_speakers_to_transcript([seg], self._diarization_segments)
                            raw_speaker = assigned[0].get("speaker", "") if assigned else ""
                            if raw_speaker:
                                real_name = self._diarization_speaker_map.get(raw_speaker, "")
                                if real_name:
                                    seg["speaker"] = self._first_name(real_name)

                        # PRIORITY 2: Visual speaker detection (fallback)
                        # speaking_timeline already contains first names (converted in _analyze_frame_live),
                        # so attribute_speaker_from_context returns first names directly.
                        first_names = sorted(set(self._first_name(v) for v in self._name_map.values())) if self._name_map else []
                        if not seg.get("speaker") and self._speaking_timeline:
                            speaker = await attribute_speaker_from_context(
                                seg.get("text", ""),
                                self._speaking_timeline,
                                first_names,
                                seg_ts,
                            )
                            if speaker:
                                seg["speaker"] = self._first_name(speaker)

                        # Track speaking duration from transcript attribution
                        # ONLY before diarization sets it (diarization is more accurate)
                        if not self._diarization_speaking_set:
                            transcript_speaker = seg.get("speaker", "")
                            if transcript_speaker and not re.match(r'^Person \d+$', transcript_speaker):
                                seg_duration = max(0.5, seg.get("end", 0) - seg.get("start", 0))
                                first = self._first_name(transcript_speaker)
                                if first and not re.match(r'^Person \d+$', first):
                                    self._speaking_durations[first] = self._speaking_durations.get(first, 0) + seg_duration
                                    self._total_speaking_time += seg_duration

                        self._transcript_segments.append(seg)
                        accepted += 1

                    logger.info(f"[{self.video_id}] Transcribed [{start:.1f}-{end:.1f}]: {accepted}/{len(segments)} segments (filtered+cleaned)")

                    # Immediately deliver
                    await self._send_pending_transcript(float("inf"))

                    # Cap transcript segments
                    if len(self._transcript_segments) > self.MAX_TRANSCRIPT_SEGMENTS:
                        self._transcript_segments = self._transcript_segments[-self.MAX_TRANSCRIPT_SEGMENTS:]
                        # Adjust index to prevent stale pointer
                        self._next_segment_idx = min(self._next_segment_idx, len(self._transcript_segments))

                    # Voice/word signal analysis
                    chunk_text = " ".join(s.get("text", "") for s in segments)
                    if chunk_text.strip():
                        task = self._create_tracked_task(self._analyze_audio_signals(chunk_text, (start + end) / 2))
                        task.add_done_callback(self._log_task_exception)

                    try:
                        os.remove(chunk_path)
                    except OSError:
                        pass

                except Exception as e:
                    logger.error(f"[{self.video_id}] Transcription [{start:.1f}-{end:.1f}] failed: {e}")

                # Check if more audio was queued
                if self._transcribe_queue <= self._audio_transcribed_up_to:
                    break

        finally:
            self._active_transcribe_count -= 1

    async def _analyze_audio_features(self, chunk_path: str, start: float, end: float):
        """Analyze audio features from a transcription chunk.
        Runs librosa off the event loop to prevent blocking."""
        try:
            loop = asyncio.get_event_loop()
            features = await loop.run_in_executor(None, self.audio_analyzer.analyze_file, chunk_path)
            audio_data = {
                "timestamp": (start + end) / 2,
                "energy": features.engagement_score,
                "pitch_mean": features.pitch_mean,
                "pitch_std": features.pitch_std,
                "volume_energy": features.volume_energy,
                "speaking_rate": features.speaking_rate,
                "pause_count": features.pause_count,
            }
            self.audio_features.append(audio_data)
            if len(self.audio_features) > self.MAX_AUDIO_FEATURES:
                self.audio_features = self.audio_features[-self.MAX_AUDIO_FEATURES:]
            if not self._cancelled and not self._finalizing:
                await self._send_json({"type": "audio_features", "data": audio_data})
        except Exception as e:
            logger.warning(f"[{self.video_id}] Audio analysis failed: {e}")

    async def _send_pending_transcript(self, current_ts: float):
        """Send transcript segments that have been reached by playback."""
        while self._next_segment_idx < len(self._transcript_segments):
            seg = self._transcript_segments[self._next_segment_idx]
            if seg.get("start", 0) > current_ts:
                break
            await self._send_json({
                "type": "transcript",
                "data": {
                    "start_time": seg.get("start", 0),
                    "end_time": seg.get("end", 0),
                    "text": seg.get("text", ""),
                    "speaker": seg.get("speaker", ""),
                },
            })
            self._next_segment_idx += 1

    async def _analyze_audio_signals(self, chunk_text: str, timestamp: float):
        """Analyze voice and words for an audio chunk."""
        try:
            voice_task = analyze_voice_signal(chunk_text)
            words_task = analyze_words_signal(chunk_text, timestamp)
            voice_result, words_result = await asyncio.gather(
                voice_task, words_task, return_exceptions=True
            )

            if not isinstance(voice_result, Exception):
                voice_result["timestamp"] = timestamp
                self.voice_signals.append(voice_result)
                if len(self.voice_signals) > self.MAX_VOICE_SIGNALS:
                    self.voice_signals = self.voice_signals[-self.MAX_VOICE_SIGNALS:]
                await self._send_json({"type": "voice", "data": voice_result})

            if not isinstance(words_result, Exception):
                words_result["timestamp"] = timestamp
                self.word_signals.append(words_result)
                if len(self.word_signals) > self.MAX_WORD_SIGNALS:
                    self.word_signals = self.word_signals[-self.MAX_WORD_SIGNALS:]
                await self._send_json({"type": "words", "data": words_result})

        except Exception as e:
            logger.error(f"[{self.video_id}] Audio signal analysis failed: {e}")

    # ─── Periodic deeper analysis ─────────────────────────────────

    async def _analyze_personality_periodic(self):
        """Analyze personality from accumulated signals."""
        try:
            signals_summary = json.dumps(self.signal_snapshots[-20:])
            transcript_text = "\n".join(
                s.get("text", "") for s in self._transcript_segments
            )
            result = await analyze_personality(signals_summary, transcript_text)
            self.personality_data = result
            await self._send_json({"type": "personality", "data": result})
        except Exception as e:
            logger.error(f"[{self.video_id}] Personality analysis failed: {e}")

    async def _build_correlations_periodic(self):
        """Build cross-signal correlations."""
        try:
            all_signals = json.dumps({
                "visual": self.signal_snapshots[-15:],
                "voice": self.voice_signals[-5:],
                "words": self.word_signals[-5:],
            })
            transcript_text = "\n".join(
                s.get("text", "") for s in self._transcript_segments
            )
            result = await build_correlations(all_signals, transcript_text)
            new_corrs = result.get("correlations", [])
            self.correlations.extend(new_corrs)
            for corr in new_corrs:
                await self._send_json({"type": "correlation", "data": corr})
        except Exception as e:
            logger.error(f"[{self.video_id}] Correlation analysis failed: {e}")

    # ─── Finalization ─────────────────────────────────────────────

    async def _deferred_speaker_attribution(self):
        """Re-attribute unattributed transcript segments using the COMPLETE speaking_timeline.

        Uses pre-computed speaker buckets (10s windows) with dominance requirement:
        a speaker must have 2+ detections AND 2x the runner-up in each bucket.
        This filters out scattered false positives from noisy visual is_speaking.
        Also recomputes speaking_durations from the corrected attributions.
        """
        if not self._speaking_timeline or not self._transcript_segments:
            return

        first_names = sorted(set(self._first_name(v) for v in self._name_map.values())) if self._name_map else []

        # Build speaker buckets from the complete timeline
        buckets = await build_speaker_buckets(self._speaking_timeline, bucket_size=10.0)
        logger.info(f"[{self.video_id}] Speaker buckets (10s windows): {buckets}")

        attributed_count = 0
        # Reset speaking durations — will recompute from corrected attributions
        self._speaking_durations.clear()
        self._total_speaking_time = 0.0

        for seg in self._transcript_segments:
            seg_ts = seg.get("start", 0)

            # Re-attribute ALL segments using buckets (even previously attributed ones)
            # This ensures consistency — live attribution was unreliable due to lag
            speaker = await attribute_speaker_from_context(
                seg.get("text", ""),
                self._speaking_timeline,
                first_names,
                seg_ts,
                speaker_buckets=buckets,
            )
            if speaker:
                old = seg.get("speaker", "")
                seg["speaker"] = self._first_name(speaker)
                if not old:
                    attributed_count += 1
            else:
                seg["speaker"] = ""  # Clear any noisy live attribution

            # Accumulate speaking durations from ALL attributed segments
            transcript_speaker = seg.get("speaker", "")
            if transcript_speaker and not re.match(r'^Person \d+$', transcript_speaker):
                seg_duration = max(0.5, seg.get("end", 0) - seg.get("start", 0))
                first = self._first_name(transcript_speaker)
                if first and not re.match(r'^Person \d+$', first):
                    self._speaking_durations[first] = self._speaking_durations.get(first, 0) + seg_duration
                    self._total_speaking_time += seg_duration

        unattributed = sum(1 for s in self._transcript_segments if not s.get("speaker"))
        logger.info(
            f"[{self.video_id}] Deferred attribution: {attributed_count} newly attributed, "
            f"{unattributed} still unattributed, timeline={len(self._speaking_timeline)} entries, "
            f"speaking_durations={dict(self._speaking_durations)}"
        )

    async def finalize(self):
        """On video_ended: wait for pending, generate summary + coaching, persist."""
        self.is_playing = False
        self._finalizing = True  # Prevent pending tasks from sending signals to chart
        logger.info(f"[{self.video_id}] Video ended — finalizing...")
        await self._send_json({"type": "video_ended_ack"})  # Tell frontend to freeze chart
        await self._send_status("Completing analysis...")

        # Lock participants if not already locked (video ended before stability)
        if not self._participants_locked:
            self._participants_locked = True
            unique_first = set(self._first_name(v) for v in self._name_map.values())
            logger.info(f"[{self.video_id}] Finalize: LOCKED {len(unique_first)} participants: {unique_first}")

        # Wait for all in-flight analysis tasks
        if self._tracked_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*list(self._tracked_tasks), return_exceptions=True),
                    timeout=60.0,
                )
            except asyncio.TimeoutError:
                logger.warning(f"[{self.video_id}] Tracked tasks timed out after 60s")

        await self._send_status("Generating summary and coaching...")

        # Transcribe ALL remaining audio — force=True processes even tiny chunks
        duration = await get_video_duration(self.video_path)
        if self._audio_ready and self._audio_transcribed_up_to < duration:
            self._transcribe_queue = duration
            # Wait for any running workers to finish
            if self._transcribe_tasks:
                await asyncio.gather(*self._transcribe_tasks, return_exceptions=True)
            # Force-transcribe any remaining audio (even sub-second)
            if self._audio_transcribed_up_to < duration:
                await self._transcription_worker(force=True)

        await self._send_pending_transcript(float("inf"))

        # Deferred speaker attribution: re-attribute unattributed transcript segments
        # using the COMPLETE speaking_timeline (no lag issues at finalize)
        await self._deferred_speaker_attribution()

        # Personality if never ran
        if self.personality_data is None and len(self.signal_snapshots) >= 2:
            await self._analyze_personality_periodic()

        # Correlations if never ran
        if not self.correlations and len(self.signal_snapshots) >= 2:
            await self._build_correlations_periodic()

        try:
            transcript_text = "\n".join(
                f"[{s.get('start', 0):.1f}s]: {s.get('text', '')}"
                for s in self._transcript_segments
            )
            emotions_summary = json.dumps(
                [
                    {
                        "timestamp": s.get("timestamp", 0),
                        "participants": s.get("participants", []),
                    }
                    for s in self.signal_snapshots[:30]
                ],
            )
            # Full names for DB persistence
            participant_names = sorted(self.participant_labels) or ["Person 1"]
            if self._name_map:
                participant_names = sorted(set(
                    self._name_map.get(name, name) for name in participant_names
                ))

            # First names for display (WebSocket messages)
            display_names = sorted(set(self._first_name(n) for n in participant_names))

            analysis = await generate_summary_and_flags(
                transcript_text[:8000],
                emotions_summary[:4000],
                display_names,
            )
            logger.info(f"[{self.video_id}] LLM analysis keys: {list(analysis.keys())}, summary_len={len(analysis.get('summary',''))}, p_scores={len(analysis.get('participant_scores',[]))}")

            # Send summary
            await self._send_json({
                "type": "summary",
                "data": {
                    "summary": analysis.get("summary", ""),
                    "key_topics": analysis.get("key_topics", []),
                    "overall_sentiment": analysis.get("overall_sentiment", "neutral"),
                },
            })

            # Generate coaching per participant (use first names for display)
            coaching_per_participant: dict[str, list[dict]] = {}
            for name in display_names:
                p_emotions = json.dumps(
                    [s for s in self.signal_snapshots if name in str(s)][:15],
                )
                coaching_items = await generate_coaching(
                    transcript_text[:4000], p_emotions[:2000], name
                )
                coaching_per_participant[name] = coaching_items
                await self._send_json({
                    "type": "coaching",
                    "data": {"participant": name, "items": coaching_items},
                })

            # Send flags
            for flag in analysis.get("flags", []):
                self.flags.append(flag)
                await self._send_json({"type": "flag", "data": flag})

            # Persist to DB
            await self._persist_to_db(analysis, participant_names, coaching_per_participant)

            logger.info(f"[{self.video_id}] Finalization complete")
            await self._send_status("Analysis complete")
            await self._send_json({"type": "complete"})

        except Exception as e:
            logger.error(f"[{self.video_id}] Finalization error: {e}")
            await self._send_json({"type": "error", "data": {"message": f"Analysis error: {str(e)[:100]}"}})
            await self._send_status(f"Error finalizing: {str(e)[:100]}")
            await self._send_json({"type": "complete"})

    async def cleanup(self):
        """Cancel everything on disconnect. Clean up frame buffer."""
        self._cancelled = True
        for task in list(self._tracked_tasks):
            if not task.done():
                task.cancel()
        self._tracked_tasks.clear()
        # Clean up pre-extracted frames
        for path in self._frame_buffer.values():
            try:
                os.remove(path)
            except OSError:
                pass
        self._frame_buffer.clear()

    # ─── DB Persistence ───────────────────────────────────────────

    async def _persist_to_db(self, analysis: dict, participant_names: list[str],
                             coaching_per_participant: dict[str, list[dict]] | None = None):
        """Persist all results to the database."""
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                duration = await get_video_duration(self.video_path)

                p_scores = analysis.get("participant_scores", [])
                avg_engagement = 0.0
                if p_scores:
                    avg_engagement = sum(p.get("engagement", 70) for p in p_scores) / len(p_scores)

                await db.execute(
                    "UPDATE videos SET duration = ?, overall_score = ?, status = ? WHERE id = ?",
                    (duration, avg_engagement, "complete", self.video_id),
                )

                # Compute actual speaking percentages from backend tracking
                actual_speaking_pct: dict[str, float] = {}
                if self._total_speaking_time > 0:
                    for spk_name, dur in self._speaking_durations.items():
                        actual_speaking_pct[spk_name] = round((dur / self._total_speaking_time) * 100, 1)

                # Build LLM scores lookup for enrichment (keyed by LLM-reported name)
                llm_scores: dict[str, dict] = {}
                for ps in p_scores:
                    llm_scores[ps.get("name", "")] = ps

                def _find_llm_scores(full_name: str) -> dict:
                    """Match participant full name to LLM scores (which use first names)."""
                    # Exact match
                    if full_name in llm_scores:
                        return llm_scores[full_name]
                    # First-name match
                    first = self._first_name(full_name)
                    if first in llm_scores:
                        return llm_scores[first]
                    # Case-insensitive first-name match
                    for k, v in llm_scores.items():
                        if k.lower() == first.lower() or k.lower() == full_name.lower():
                            return v
                    # Word overlap match
                    name_words = set(w.lower() for w in full_name.split() if len(w) >= 3)
                    for k, v in llm_scores.items():
                        k_words = set(w.lower() for w in k.split() if len(w) >= 3)
                        if name_words & k_words:
                            return v
                    return {}

                # Clean old data from previous analysis runs for this video
                await db.execute("DELETE FROM participants WHERE video_id = ?", (self.video_id,))
                await db.execute("DELETE FROM emotions WHERE video_id = ?", (self.video_id,))
                await db.execute("DELETE FROM transcript_segments WHERE video_id = ?", (self.video_id,))
                await db.execute("DELETE FROM flags WHERE video_id = ?", (self.video_id,))

                # Persist ALL known participants (not just LLM-reported ones)
                participant_id_map = {}
                for name in participant_names:
                    pid = str(uuid.uuid4())
                    participant_id_map[name] = pid
                    ps = _find_llm_scores(name)
                    # Use actual tracked speaking_pct, fall back to LLM, then even split
                    speaking_pct = actual_speaking_pct.get(name)
                    if speaking_pct is None:
                        # Fuzzy match speaking durations by word overlap
                        for spk, pct in actual_speaking_pct.items():
                            if spk.lower() == name.lower():
                                speaking_pct = pct
                                break
                            spk_words = set(w.lower() for w in spk.split() if len(w) >= 3)
                            name_words = set(w.lower() for w in name.split() if len(w) >= 3)
                            if spk_words & name_words:
                                speaking_pct = pct
                                break
                    if speaking_pct is None:
                        speaking_pct = ps.get("speaking_pct", round(100.0 / max(len(participant_names), 1), 1))

                    # Evidence-based scoring: scale down scores for silent participants.
                    # If someone barely spoke, we have no evidence for high engagement.
                    raw_eng = ps.get("engagement", 70)
                    raw_sent = max(0, min(100, (ps.get("sentiment", 0.3) + 1) * 50))
                    raw_clar = ps.get("clarity", 70)
                    raw_rap = ps.get("rapport", 70)
                    raw_ener = ps.get("energy", 70)

                    if speaking_pct < 1.0:
                        # Silent participant — minimal scores, no evidence
                        raw_eng = min(raw_eng, 15)
                        raw_sent = 50  # neutral — no data
                        raw_clar = min(raw_clar, 15)
                        raw_rap = min(raw_rap, 15)
                        raw_ener = min(raw_ener, 15)
                    elif speaking_pct < 5.0:
                        # Very low participation — cap scores
                        cap = 30 + speaking_pct * 4  # 30-50 range for 1-5%
                        raw_eng = min(raw_eng, cap)
                        raw_clar = min(raw_clar, cap)
                        raw_rap = min(raw_rap, cap)
                        raw_ener = min(raw_ener, cap)

                    await db.execute(
                        """INSERT INTO participants (id, video_id, name, engagement_score,
                           sentiment_score, speaking_pct, clarity_score, rapport_score, energy_score)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            pid, self.video_id, name,
                            raw_eng,
                            raw_sent,
                            speaking_pct,
                            raw_clar,
                            raw_rap,
                            raw_ener,
                        ),
                    )

                for snap in self.signal_snapshots:
                    ts = snap.get("timestamp", 0)
                    for p in snap.get("participants", []):
                        label = p.get("label", "Person 1")
                        pid = participant_id_map.get(label)
                        if not pid and participant_id_map:
                            pid = list(participant_id_map.values())[0]
                        elif not pid:
                            # Generate a real UUID for unmatched labels instead of "unknown"
                            pid = str(uuid.uuid4())
                            participant_id_map[label] = pid
                        emotions = p.get("emotions", {})
                        bl = p.get("body_language", {})
                        await db.execute(
                            """INSERT INTO signal_snapshots
                               (video_id, participant_id, timestamp,
                                emotion, emotion_confidence, engagement,
                                posture, openness, leaning, mirroring, body_language_desc,
                                gestures, reactions)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                self.video_id, pid, ts,
                                emotions.get("primary", "neutral"),
                                emotions.get("confidence", 0.5),
                                emotions.get("engagement", 50),
                                bl.get("posture", "upright"),
                                bl.get("openness", "mixed"),
                                bl.get("leaning", "neutral"),
                                bl.get("mirroring", False),
                                bl.get("description", ""),
                                json.dumps(p.get("gestures", [])),
                                json.dumps(p.get("reactions", [])),
                            ),
                        )
                        # Derive emotions from signal_snapshots (single source of truth)
                        await db.execute(
                            """INSERT INTO emotions (video_id, participant_id, timestamp,
                               emotion, confidence, engagement) VALUES (?, ?, ?, ?, ?, ?)""",
                            (
                                self.video_id, pid, ts,
                                emotions.get("primary", "neutral"),
                                emotions.get("confidence", 0.5),
                                emotions.get("engagement", 50),
                            ),
                        )

                for seg in self._transcript_segments:
                    await db.execute(
                        """INSERT INTO transcript_segments
                           (video_id, start_time, end_time, text, speaker)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            self.video_id,
                            seg.get("start", 0),
                            seg.get("end", 0),
                            seg.get("text", ""),
                            seg.get("speaker", ""),
                        ),
                    )

                for vs in self.voice_signals:
                    await db.execute(
                        """INSERT INTO voice_signals
                           (video_id, timestamp, tone, pace, energy, dynamics)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            self.video_id,
                            vs.get("timestamp", 0),
                            vs.get("tone", "neutral"),
                            vs.get("pace", "moderate"),
                            vs.get("energy", 50),
                            vs.get("dynamics", ""),
                        ),
                    )

                for ws_data in self.word_signals:
                    await db.execute(
                        """INSERT INTO word_signals
                           (video_id, timestamp, sentiment, sentiment_score, keywords, key_phrases)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            self.video_id,
                            ws_data.get("timestamp", 0),
                            ws_data.get("sentiment", "neutral"),
                            ws_data.get("sentiment_score", 0.5),
                            json.dumps(ws_data.get("keywords", [])),
                            json.dumps(ws_data.get("key_phrases", [])),
                        ),
                    )

                if self.personality_data:
                    for p in self.personality_data.get("participants", []):
                        label = p.get("label", "Person 1")
                        pid = participant_id_map.get(label)
                        if not pid and participant_id_map:
                            pid = list(participant_id_map.values())[0]
                        elif not pid:
                            pid = str(uuid.uuid4())
                            participant_id_map[label] = pid
                        await db.execute(
                            """INSERT INTO personality_signals
                               (video_id, participant_id, traits, communication_style, confidence_level)
                               VALUES (?, ?, ?, ?, ?)""",
                            (
                                self.video_id, pid,
                                json.dumps(p.get("traits", [])),
                                p.get("communication_style", ""),
                                p.get("confidence_level", ""),
                            ),
                        )

                for corr in self.correlations:
                    await db.execute(
                        """INSERT INTO correlations
                           (video_id, signal_pair, pattern, confidence, implication)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            self.video_id,
                            json.dumps(corr.get("signal_pair", [])),
                            corr.get("pattern", ""),
                            corr.get("confidence", 0.0),
                            corr.get("implication", ""),
                        ),
                    )

                for flag in self.flags:
                    await db.execute(
                        """INSERT INTO flags (video_id, timestamp, type, description, severity)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            self.video_id,
                            flag.get("timestamp", 0),
                            flag.get("type", "neutral"),
                            flag.get("description", ""),
                            flag.get("severity", "medium"),
                        ),
                    )

                summary_text = analysis.get("summary", "")
                if summary_text:
                    await db.execute(
                        """INSERT OR REPLACE INTO summaries
                           (video_id, summary, key_topics, overall_sentiment)
                           VALUES (?, ?, ?, ?)""",
                        (
                            self.video_id,
                            summary_text,
                            json.dumps(analysis.get("key_topics", [])),
                            analysis.get("overall_sentiment", "neutral"),
                        ),
                    )

                for name, pid in participant_id_map.items():
                    coaching_items = (coaching_per_participant or {}).get(name, [])
                    for item in coaching_items:
                        await db.execute(
                            """INSERT INTO coaching
                               (video_id, participant_id, recommendation, category, priority)
                               VALUES (?, ?, ?, ?, ?)""",
                            (
                                self.video_id, pid,
                                item.get("recommendation", ""),
                                item.get("category", "general"),
                                item.get("priority", 3),
                            ),
                        )

                await db.commit()
                logger.info(f"[{self.video_id}] Results persisted to DB")

        except Exception as e:
            logger.error(f"[{self.video_id}] DB persistence error: {e}")
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    "UPDATE videos SET status = ? WHERE id = ?",
                    (f"error: {str(e)[:200]}", self.video_id),
                )
                await db.commit()

    # ─── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _clean_name(name: str) -> str:
        """Normalize name format: 'Chong, Sam' -> 'Sam Chong', 'Giorgio Manenti (E)' -> 'Giorgio Manenti'.

        Handles truncated names: 'Christ... Schmell' -> 'Schmell', '... Kaushik' -> 'Kaushik',
        'Sa... Zendehrooh' -> 'Zendehrooh'. Truncated fragments are REMOVED entirely —
        only full words survive so they can fuzzy-match to the real name later.

        ALWAYS returns Title Case to prevent "john smith" vs "John Smith" variants.
        """
        # Remove trailing parenthetical like "(E)"
        name = re.sub(r'\s*\([^)]*\)\s*$', '', name).strip()
        # Fix "Last, First" format (common in Teams)
        if ',' in name:
            parts = [p.strip() for p in name.split(',', 1)]
            if len(parts) == 2 and parts[0] and parts[1]:
                # If first part looks like a last name (single word), swap
                if len(parts[0].split()) == 1:
                    name = f"{parts[1]} {parts[0]}"
                else:
                    # Multi-word before comma: keep as-is but remove comma
                    name = f"{parts[0]} {parts[1]}"
        # Remove ALL ellipsis (trailing, leading, middle) and truncated word fragments
        # "Christ... Schmell" → split on "..." → ["Christ", " Schmell"]
        # "... Kaushik" → ["", " Kaushik"] → "Kaushik"
        # "Sa... Zendehrooh" → ["Sa", " Zendehrooh"]
        # Keep only parts that are AFTER the last "..." (they are the complete words)
        if '...' in name:
            # Split on ellipsis and take only complete fragments
            # The word BEFORE ... is always truncated, so drop it
            fragments = name.split('...')
            # Keep the last fragment (text after last ...)
            last_frag = fragments[-1].strip()
            # Also check if there's a clean first part (word before first ...)
            first_frag = fragments[0].strip()
            clean_parts = []
            # First fragment: only keep if it looks complete (no truncation indicator)
            # A truncated word like "Christ" or "Sa" before ... is incomplete, skip it
            # But "Albert Maestro Pablo" before "..." IS complete multi-word, keep all words
            if first_frag:
                first_words = first_frag.split()
                # Keep all words except the last one (which is the truncated one)
                if len(first_words) > 1:
                    clean_parts.extend(first_words[:-1])
                # If single word before ..., it's the truncated one — skip
            if last_frag:
                clean_parts.extend(last_frag.split())
            name = ' '.join(clean_parts)
        else:
            # Remove simple trailing ellipsis (Unicode or ascii)
            name = re.sub(r'[…\.\.\.]$', '', name).strip()
        # Remove any remaining dots that look like truncation
        name = re.sub(r'\.{2,}', '', name).strip()
        # Normalize to Title Case — prevents "john smith" vs "John Smith" variants
        name = name.title()

        # REJECT non-name strings that OCR picks up from Teams UI
        # "+2", "+3" — gallery overflow indicators
        # Single chars, numbers, "Person" without number
        if not name:
            return ""
        # Reject pure numbers or +N overflow indicators
        if re.match(r'^[+\-]?\d+$', name.strip()):
            return ""
        # Reject single character names
        if len(name.strip()) < 2:
            return ""
        # Reject "Person" by itself (no number)
        if name.strip().lower() == "person":
            return ""
        # Reject strings that are just punctuation/symbols
        alpha_chars = sum(1 for c in name if c.isalpha())
        if alpha_chars < 2:
            return ""

        return name

    def _build_initials_map(self):
        """Build reverse map: initials -> full name (e.g., 'RM' -> 'Robin Mills').

        Generates multiple initials variants to handle Teams-style display initials.
        """
        self._initials_map = {}
        for name in self._name_map.values():
            clean = self._clean_name(name)
            words = clean.split()
            if len(words) >= 2:
                # All first letters: "Albert Maestro Pablo Vicente" -> "AMPV"
                all_initials = ''.join(w[0].upper() for w in words if w)
                self._initials_map[all_initials] = clean
                # First + Last: "AV"
                fl = words[0][0].upper() + words[-1][0].upper()
                self._initials_map[fl] = clean
                # Last + First: "VA"
                lf = words[-1][0].upper() + words[0][0].upper()
                self._initials_map[lf] = clean
                # Any pair of consecutive word initials + reversed
                for i in range(len(words) - 1):
                    pair = words[i][0].upper() + words[i+1][0].upper()
                    rev = words[i+1][0].upper() + words[i][0].upper()
                    if pair not in self._initials_map:
                        self._initials_map[pair] = clean
                    if rev not in self._initials_map:
                        self._initials_map[rev] = clean
            elif len(words) == 1 and len(clean) >= 2:
                self._initials_map[clean[:2].upper()] = clean

    def _fuzzy_match_name(self, raw_label: str, known_names: set[str]) -> str | None:
        """Match a name variant to a canonical known name.

        STRICT matching — only returns matches with high confidence:
        - Full/case-insensitive exact match
        - Initials map lookup
        - Single word exact match to first or last name (must be unique, >= 4 chars)
        - Multi-word exact word overlap (>= 4 chars per word)
        """
        if not known_names or not raw_label:
            return None

        # Clean the raw label
        cleaned = self._clean_name(raw_label)
        if not cleaned:
            return None
        raw_lower = cleaned.lower().strip()

        # 1. Exact case-insensitive match
        for name in known_names:
            if name.lower() == raw_lower:
                return name

        # 2. Check initials map
        if raw_label.strip() in self._initials_map:
            return self._initials_map[raw_label.strip()]
        if raw_label.strip().upper() in self._initials_map:
            return self._initials_map[raw_label.strip().upper()]

        raw_parts = [w for w in raw_lower.replace(".", "").replace(",", " ").split() if len(w) >= 4]

        # 3. Single word (>= 4 chars) → match as first or last name (must be UNIQUE match)
        if len(raw_parts) == 1:
            word = raw_parts[0]
            matches = [n for n in known_names
                       if word in [w.lower() for w in n.split() if len(w) >= 4]]
            if len(matches) == 1:
                return matches[0]

        # 4. Multi-word: exact word overlap (words >= 4 chars)
        if len(raw_parts) >= 2:
            raw_set = set(raw_parts)
            best_match = None
            best_score = 0
            for name in known_names:
                name_words = set(w.lower() for w in name.split() if len(w) >= 4)
                score = len(name_words & raw_set)
                if score > best_score:
                    best_score = score
                    best_match = name
            if best_score >= 1:
                return best_match

        return None

    def _resolve_label(self, label: str) -> str:
        """Resolve a participant label to a real name using all available maps.

        Confirmed names are preferred. Falls through to aggressive fuzzy matching.
        """
        if not self._name_map:
            return label

        # Check confirmed names first (persistent cache)
        if label in self._confirmed_names:
            return self._confirmed_names[label]

        # Direct name_map lookup: "Person 1" -> "Robin Mills"
        if label in self._name_map:
            return self._name_map[label]

        known_names = set(self._name_map.values())

        # Check if label IS a known real name (already resolved)
        if label in known_names:
            return label

        # Extract "Person N" number and map it
        m = re.match(r'Person\s+(\d+)', label)
        if m:
            key = f"Person {m.group(1)}"
            if key in self._name_map:
                return self._name_map[key]

        # Initials lookup: "RM" -> "Robin Mills"
        if label in self._initials_map:
            return self._initials_map[label]

        # Aggressive fuzzy match — handles all name variants
        fuzzy = self._fuzzy_match_name(label, known_names)
        if fuzzy:
            return fuzzy

        return label

    def _to_canonical(self, raw: str, canonical_names: list[str]) -> str:
        """Map a name variant to the closest canonical name.

        STRICT matching rules:
        1. Exact match (case-insensitive) → done
        2. Exact word overlap (word >= 4 chars) → use that canonical
        3. Fall back to _resolve_label for Person N mapping

        No prefix matching, no substring matching — these cause false positives.
        """
        if not raw:
            return raw
        cleaned = self._clean_name(raw)
        if not cleaned:
            return raw

        # Exact match
        for cn in canonical_names:
            if cn.lower() == cleaned.lower():
                return cn

        # Exact word overlap (word must be >= 4 chars — no short words like "Sam")
        raw_words = set(w.lower() for w in cleaned.split() if len(w) >= 4)
        for cn in canonical_names:
            cn_words = set(w.lower() for w in cn.split() if len(w) >= 4)
            if raw_words and cn_words and (raw_words & cn_words):
                return cn

        # First letter + last word: "A. Roy" → "Anup Roy"
        parts = cleaned.replace(".", "").split()
        if len(parts) >= 2 and len(parts[0]) <= 2:
            initial = parts[0][0].lower()
            last_word = parts[-1].lower()
            for cn in canonical_names:
                cn_parts = cn.split()
                if len(cn_parts) >= 2 and cn_parts[0][0].lower() == initial and cn_parts[-1].lower() == last_word:
                    return cn

        # Initials
        if cleaned in self._initials_map:
            return self._initials_map[cleaned]
        if cleaned.upper() in self._initials_map:
            return self._initials_map[cleaned.upper()]

        # Fuzzy match — aggressive matching for remaining variants
        fuzzy = self._fuzzy_match_name(cleaned, canonical_names)
        if fuzzy:
            return fuzzy

        # Surname fallback: check if raw label matches any word in FULL names
        # stored in _name_map.  E.g. OCR reads "Schmell" → matches
        # "Christoph Schmell" in _name_map → return "Christoph" (the first name).
        if self._name_map:
            cl_words = set(w.lower() for w in cleaned.split() if len(w) >= 3)
            for full_name in set(self._name_map.values()):
                fn_words = set(w.lower() for w in full_name.split() if len(w) >= 3)
                if cl_words and fn_words and (cl_words & fn_words):
                    first = self._first_name(full_name)
                    if first in canonical_names:
                        return first
        # Person N fallback via _resolve_label
        resolved = self._resolve_label(raw)
        # If we have canonical names and resolution returned the raw label unchanged,
        # reject it to prevent name variants from leaking through
        if canonical_names and resolved == raw:
            return f"Person {len(canonical_names) + 1}"
        return resolved

    @staticmethod
    def _first_name(full_name: str) -> str:
        """Extract first name only for display. Handles edge cases:
        - "Sarang Zendehrooh" → "Sarang"
        - "M Giorgio" → "Giorgio" (skip 1-2 char initials)
        - "Nr Rangwan" → "Rangwan"
        - "Christoph" → "Christoph" (already single word)
        - "Person 1" → "Person 1" (preserve generic labels)
        """
        # Preserve "Person N" labels — they're identifiers, not real names
        if re.match(r'^Person \d+$', full_name.strip()):
            return full_name.strip()
        words = full_name.strip().split()
        if not words:
            return full_name
        # If first word is just an initial (1-2 chars), use the second word
        if len(words) >= 2 and len(words[0]) <= 2:
            return words[1]
        return words[0]

    @classmethod
    def _names_are_same_person(cls, name_a: str, name_b: str) -> bool:
        """Check if two name strings refer to the same person.

        Matching rules (in order):
        1. Exact match / case-insensitive
        2. First name match (EXACT, >= 4 chars)
        3. Exact word overlap (word >= 4 chars)
        4. Shared prefix >= 4 chars for words >= 5 chars (OCR truncation: "Zendher" ~ "Zendehrooh")
        5. Edit distance <= 2 for words >= 6 chars (OCR typos: "Schnell" ~ "Schmell")
        """
        if name_a == name_b:
            return True
        a_lower = name_a.lower().strip()
        b_lower = name_b.lower().strip()
        if a_lower == b_lower:
            return True

        a_words = [w for w in a_lower.split() if len(w) >= 4]
        b_words = [w for w in b_lower.split() if len(w) >= 4]

        # First name comparison (EXACT match only)
        a_first = cls._first_name(name_a).lower().strip()
        b_first = cls._first_name(name_b).lower().strip()
        if a_first == b_first and len(a_first) >= 4:
            return True

        # Exact word overlap (word >= 4 chars)
        a_set = set(a_words)
        b_set = set(b_words)
        if a_set and b_set and (a_set & b_set):
            return True

        # Shared prefix check for OCR truncation: "Zendher" shares "Zend" with "Zendehrooh"
        # Both words must be >= 5 chars, shared prefix >= 4 chars
        for aw in a_words:
            for bw in b_words:
                if len(aw) >= 5 and len(bw) >= 5:
                    # Find shared prefix length
                    prefix_len = 0
                    for i in range(min(len(aw), len(bw))):
                        if aw[i] == bw[i]:
                            prefix_len += 1
                        else:
                            break
                    if prefix_len >= 4:
                        return True

        # Edit distance for OCR typos (words >= 6 chars, distance <= 2)
        for aw in a_words:
            for bw in b_words:
                if len(aw) >= 6 and len(bw) >= 6 and abs(len(aw) - len(bw)) <= 2:
                    if cls._edit_distance(aw, bw) <= 2:
                        return True

        return False

    @staticmethod
    def _edit_distance(s: str, t: str) -> int:
        """Simple Levenshtein distance (no external deps)."""
        if len(s) < len(t):
            s, t = t, s
        if not t:
            return len(s)
        prev = list(range(len(t) + 1))
        for i, sc in enumerate(s):
            curr = [i + 1]
            for j, tc in enumerate(t):
                cost = 0 if sc == tc else 1
                curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
            prev = curr
        return prev[-1]

    def _log_task_exception(self, task: asyncio.Task):
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(f"[{self.video_id}] Background task failed: {exc}")

    async def _send_json(self, data: dict):
        try:
            await self.ws.send_json(data)
        except WebSocketDisconnect:
            self._cancelled = True
        except Exception as e:
            logger.warning(f"[{self.video_id}] WebSocket send failed: {e}")

    async def _send_status(self, message: str):
        logger.info(f"[{self.video_id}] Status: {message}")
        await self._send_json({"type": "status", "data": {"message": message}})
