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
    generate_coaching,
    generate_summary_and_flags,
    extract_names_from_frame,
    assign_speakers_to_names,
)
from .pipelines.audio import AudioAnalyzer
from .signal_bus import SignalBus, SignalEvent, SignalType
from .engagement_engine import EngagementEngine, EngagementAlertDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "processed")


class RealtimeSession:
    """Pure real-time analysis — everything happens live as video plays."""

    FRAME_INTERVAL = 1.0
    MAX_CONCURRENT = 8
    AUDIO_CHUNK_SECONDS = 2.0   # Smaller chunks = faster individual responses
    AUDIO_MIN_CHUNK = 1.0       # Trigger threshold — start worker every 1.0s of new audio
    AUDIO_WORKER_MIN = 0.5      # Worker processes chunks down to 0.5s (finalize: 0s)
    MAX_PARALLEL_TRANSCRIBE = 2  # Number of parallel transcription workers
    PERSONALITY_INTERVAL = 30.0
    CORRELATION_INTERVAL = 45.0
    HEARTBEAT_INTERVAL = 1.0  # Send engagement update every 1s
    NAME_RESCAN_INTERVAL = 60.0  # Re-extract names from live frames every 60s

    # Whisper hallucination filter — these are commonly hallucinated on silence
    WHISPER_HALLUCINATIONS = {
        "you", "you.", "you!", "You", "You.", "bye", "bye.", "bye-bye", "bye-bye.",
        "thanks", "thanks.", "thank you", "thank you.", "Thank you.", "Thank you",
        "thanks for watching", "thanks for watching.", "Thanks for watching.",
        "subscribe", "like and subscribe", "Bye.", "Bye", "Bye-bye.",
        "uh", "um", "hmm", "huh", "oh", "ah",
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

        # Concurrency
        self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT)
        self._pending_tasks: set[asyncio.Task] = set()

        # Audio streaming state
        self._audio_path: str | None = None
        self._audio_ready = False
        self._audio_extracted = asyncio.Event()
        self._audio_transcribed_up_to = 0.0
        self._transcribe_lock = asyncio.Lock()  # Coordinate parallel workers
        self._active_transcribe_count = 0       # Number of workers currently running
        self._transcribe_queue: float = 0.0     # Queued target timestamp (latest wins)
        self._transcript_segments: list[dict] = []
        self._next_segment_idx = 0
        self._audio_task: asyncio.Task | None = None
        self._transcribe_tasks: list[asyncio.Task] = []

        # Accumulated analysis data
        self.signal_snapshots: list[dict] = []
        self.voice_signals: list[dict] = []
        self.word_signals: list[dict] = []
        self.personality_data: dict | None = None
        self.correlations: list[dict] = []
        self.flags: list[dict] = []
        self.participant_labels: set[str] = set()

        # Name mapping: "Person 1" -> "Anup Roy"
        self._name_map: dict[str, str] = {}
        self._initials_map: dict[str, str] = {}  # "RM" -> "Robin Mills" for camera-off matching
        self._last_name_rescan_ts: float = 0.0  # Last time we rescanned names during playback
        self._speaker_assignment_segments: list[dict] = []  # Buffer for batch speaker assignment
        self._last_speaker_assignment_ts: float = 0.0
        self._speaker_assignment_interval: float = 10.0  # Assign speakers every 10s of audio

        # Last known signal per participant (for heartbeat continuation)
        self._last_known_signal: dict | None = None

        # Audio analysis + engagement engine
        self.signal_bus = SignalBus()
        self.engagement_engine = EngagementEngine()
        self.alert_detector = EngagementAlertDetector(
            threshold=float(os.getenv("ENGAGEMENT_ALERT_THRESHOLD", "40")),
            min_duration=float(os.getenv("ENGAGEMENT_ALERT_DURATION", "30")),
        )
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

    # Timestamps to scan for participant gallery (skip 0 = splash screen)
    NAME_SCAN_TIMESTAMPS = [5.0, 10.0, 15.0, 30.0, 60.0, 120.0]

    async def initialize(self):
        """Called on WebSocket connect. Start detection + audio extraction + name OCR immediately
        so results are ready by the time user hits play (~1s head start)."""
        logger.info(f"[{self.video_id[:8]}] Session initialized — pre-loading...")
        await self._send_status("Loading analysis...")
        # Start all in parallel: detection+names, audio extraction, pre-transcription
        self._audio_task = asyncio.create_task(self._extract_audio_background())
        self._detect_task = asyncio.create_task(self._detect_participants_and_names())
        asyncio.create_task(self._pre_transcribe_first_chunk())
        # Progressive name scanning in background — tries multiple frames to find gallery
        asyncio.create_task(self._progressive_name_scan())

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
                self._name_map = {k: self._clean_name(v) for k, v in name_map.items()}
                self._build_initials_map()
                self.participant_labels = set(self._name_map.values())
                logger.info(f"[{self.video_id[:8]}] Extracted {len(self._name_map)} names at {detect_ts}s: {self._name_map}")
                await self._send_json({"type": "name_map", "data": self._name_map})
            elif isinstance(name_map, Exception):
                logger.warning(f"[{self.video_id[:8]}] Name extraction at {detect_ts}s failed: {name_map}")

            await self._send_status("Ready — press play")

            try:
                os.remove(frame_path)
            except OSError:
                pass

        except Exception as e:
            logger.error(f"[{self.video_id[:8]}] Detection failed: {e}")
            await self._send_status("Ready — press play")

    async def _progressive_name_scan(self):
        """Background progressive scan: try multiple frames to find the best one with ALL names.

        Teams recordings start with a splash screen (frame 0), then show 1-2 participants,
        then eventually show gallery view with ALL participants and name labels.
        We scan progressively and update the name map whenever we find more names.
        """
        try:
            # Wait briefly for initial detection to complete first
            await asyncio.sleep(3.0)

            logger.info(f"[{self.video_id[:8]}] Starting progressive name scan (current: {len(self._name_map)} names)")

            for ts in self.NAME_SCAN_TIMESTAMPS:
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
                        logger.info(f"[{self.video_id[:8]}] Name scan at {ts}s: no names found")
                        continue

                    logger.info(f"[{self.video_id[:8]}] Name scan at {ts}s: found {len(name_map)} names")
                    await self._merge_names(name_map)

                except Exception as e:
                    logger.warning(f"[{self.video_id[:8]}] Name scan at {ts}s failed: {e}")
                    continue

            logger.info(f"[{self.video_id[:8]}] Progressive name scan complete. Final: {len(self._name_map)} names")

        except Exception as e:
            logger.error(f"[{self.video_id[:8]}] Progressive name scan error: {e}")

    async def _merge_names(self, name_map: dict[str, str]):
        """Merge newly discovered names into the master name map.

        NEVER renumbers existing Person keys — only appends new ones at the end.
        This keeps the mapping stable across frames.
        """
        existing_clean = set(self._name_map.values())
        new_names = []
        changed = False

        for val in name_map.values():
            clean = self._clean_name(val)
            if not clean or clean in existing_clean:
                continue
            # Check for partial match (truncated names)
            is_partial = False
            for existing in list(existing_clean):
                if clean.startswith(existing) and len(clean) > len(existing):
                    # Replace truncated with full version (in-place, no renumber)
                    existing_clean.discard(existing)
                    existing_clean.add(clean)
                    for k, v in list(self._name_map.items()):
                        if v == existing:
                            self._name_map[k] = clean
                    is_partial = True
                    changed = True
                    break
                elif existing.startswith(clean):
                    is_partial = True
                    break
            if not is_partial:
                existing_clean.add(clean)
                new_names.append(clean)

        if not new_names and not changed:
            return

        # Append new names with NEXT available Person number (never renumber existing)
        next_idx = len(self._name_map) + 1
        for name in new_names:
            self._name_map[f"Person {next_idx}"] = name
            next_idx += 1

        self._build_initials_map()
        self.participant_labels = set(self._name_map.values())
        logger.info(f"[{self.video_id[:8]}] Name map updated ({len(self._name_map)} names): {list(self._name_map.values())}")
        await self._send_json({"type": "name_map", "data": self._name_map})

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
                seg["start"] = seg.get("start", 0)
                seg["end"] = seg.get("end", 0)
                self._transcript_segments.append(seg)
                self._speaker_assignment_segments.append(seg)

            self._audio_transcribed_up_to = 2.0
            logger.info(f"[{self.video_id[:8]}] Pre-transcribed first 2s: {len(segments)} segments")
            try:
                os.remove(chunk_path)
            except OSError:
                pass
        except Exception as e:
            logger.warning(f"[{self.video_id[:8]}] Pre-transcription failed: {e}")

    async def _extract_audio_background(self):
        """Extract audio track from video."""
        try:
            logger.info(f"[{self.video_id[:8]}] Extracting audio track...")
            self._audio_path = await extract_full_audio(self.video_path, self.output_dir)
            if self._audio_path:
                self._audio_ready = True
                self._audio_extracted.set()
                logger.info(f"[{self.video_id[:8]}] Audio extraction complete")
            else:
                self._audio_extracted.set()
                logger.warning(f"[{self.video_id[:8]}] No audio track found")
                await self._send_json({"type": "no_audio"})
        except Exception as e:
            logger.error(f"[{self.video_id[:8]}] Audio extraction failed: {e}")
            self._audio_extracted.set()

    # ─── Playback handlers ────────────────────────────────────────

    async def handle_play(self):
        """Video started playing. Detection already started in initialize()."""
        self.is_playing = True
        logger.info(f"[{self.video_id[:8]}] PLAY")

        # If initialize() somehow didn't start tasks, do it now as fallback
        if self._audio_task is None:
            self._audio_task = asyncio.create_task(self._extract_audio_background())
            self._detect_task = asyncio.create_task(self._detect_participants_and_names())

        await self._send_status("Live analysis active")

    async def handle_pause(self):
        self.is_playing = False
        logger.info(f"[{self.video_id[:8]}] PAUSE")
        await self._send_status("Paused — press play to resume")

    async def handle_seek(self, timestamp: float):
        """Reset throttle so analysis resumes at new position."""
        self.last_analyzed_ts = timestamp - self.FRAME_INTERVAL - 1
        self._last_heartbeat_ts = timestamp - 1
        self._next_segment_idx = 0
        for i, seg in enumerate(self._transcript_segments):
            if seg.get("start", 0) > timestamp:
                self._next_segment_idx = i
                break
        else:
            self._next_segment_idx = len(self._transcript_segments)
        logger.info(f"[{self.video_id[:8]}] SEEK to {timestamp:.1f}s")
        await self._send_status(f"Seeked to {timestamp:.1f}s")

    async def handle_tick(self, timestamp: float):
        """Called on every video timeupdate (~250ms intervals).

        1. Send heartbeat signal every 1s (keeps engagement chart flowing)
        2. Deliver pending transcript segments
        3. Transcribe new audio chunks
        4. Launch live frame analysis when due
        5. Trigger periodic personality/correlation analysis
        """
        if not self.is_playing:
            return

        # 1. HEARTBEAT — send engagement update every 1s to keep chart flowing
        if timestamp - self._last_heartbeat_ts >= self.HEARTBEAT_INTERVAL:
            self._last_heartbeat_ts = timestamp
            await self._send_heartbeat(timestamp)

        # 2. Deliver pending transcript segments
        await self._send_pending_transcript(timestamp)

        # 3. Transcription — parallel workers, never drops audio
        if timestamp - self._audio_transcribed_up_to >= self.AUDIO_MIN_CHUNK:
            self._transcribe_queue = timestamp  # Always update target
            if self._active_transcribe_count < self.MAX_PARALLEL_TRANSCRIBE:
                task = asyncio.create_task(self._transcription_worker())
                self._transcribe_tasks.append(task)
                task.add_done_callback(lambda t: self._transcribe_tasks.remove(t) if t in self._transcribe_tasks else None)

        # 4. Live frame analysis — launch when due
        if timestamp - self.last_analyzed_ts >= self.FRAME_INTERVAL:
            if self._semaphore._value > 0:  # noqa: SLF001
                self.last_analyzed_ts = timestamp
                task = asyncio.create_task(self._analyze_frame_live(timestamp))
                self._pending_tasks.add(task)
                task.add_done_callback(self._pending_tasks.discard)

        # 5. Periodic deeper analysis
        if (timestamp - self._last_personality_ts >= self.PERSONALITY_INTERVAL
                and len(self.signal_snapshots) >= 3):
            self._last_personality_ts = timestamp
            asyncio.create_task(self._analyze_personality_periodic())

        if (timestamp - self._last_correlation_ts >= self.CORRELATION_INTERVAL
                and len(self.signal_snapshots) >= 4):
            self._last_correlation_ts = timestamp
            asyncio.create_task(self._build_correlations_periodic())

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
        await self._send_json({"type": "signals", "data": heartbeat})

    # ─── Frame analysis (real-time) ───────────────────────────────

    async def _analyze_frame_live(self, timestamp: float):
        """Extract and analyze a single frame. Sends results as soon as API returns."""
        async with self._semaphore:
            if self._cancelled:
                return
            try:
                logger.info(f"[{self.video_id[:8]}] Analyzing frame at {timestamp:.1f}s...")

                frame_path = await extract_single_frame(
                    self.video_path, timestamp, self.output_dir
                )

                # Run frame analysis + periodic name rescan in parallel
                tasks = [analyze_frame_all_signals(frame_path, timestamp)]
                do_name_rescan = (
                    timestamp - self._last_name_rescan_ts >= self.NAME_RESCAN_INTERVAL
                )
                if do_name_rescan:
                    tasks.append(extract_names_from_frame(frame_path))
                    self._last_name_rescan_ts = timestamp

                results = await asyncio.gather(*tasks, return_exceptions=True)
                result = results[0] if not isinstance(results[0], Exception) else None

                # Handle name rescan result
                if do_name_rescan and len(results) > 1 and isinstance(results[1], dict) and results[1]:
                    await self._merge_names(results[1])

                if result is None:
                    logger.error(f"[{self.video_id[:8]}] Frame analysis failed: {results[0]}")
                    try:
                        os.remove(frame_path)
                    except OSError:
                        pass
                    return

                result["timestamp"] = timestamp

                # Map "Person N" labels to real names. No cap — accept all detected participants.
                for p in result.get("participants", []):
                    raw_label = p.get("label", "Person 1")
                    p["label"] = self._resolve_label(raw_label)
                    if not p.get("gestures"):
                        p["gestures"] = ["still posture", "resting hands"]
                    if not p.get("reactions"):
                        p["reactions"] = ["neutral gaze", "steady expression"]

                # Deduplicate: if multiple "Person N" mapped to same name, merge them
                seen_labels = {}
                deduped = []
                for p in result.get("participants", []):
                    label = p["label"]
                    if label not in seen_labels:
                        seen_labels[label] = p
                        deduped.append(p)
                    # else: skip duplicate — same person detected twice
                result["participants"] = deduped

                # Update known participant set
                for p in deduped:
                    self.participant_labels.add(p["label"])

                # MERGE: carry forward known participants not visible in this frame
                if self._last_known_signal:
                    new_labels = {p["label"] for p in deduped}
                    for prev_p in self._last_known_signal.get("participants", []):
                        if prev_p.get("label") not in new_labels:
                            deduped.append(prev_p)

                self.signal_snapshots.append(result)
                self._last_known_signal = result  # Update for heartbeats

                if not self._cancelled and not self._finalizing:
                    await self._send_json({"type": "signals", "data": result})

                    # Compute holistic engagement using latest audio data
                    for p in result.get("participants", []):
                        label = p.get("label", "Person 1")
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

                        score = self.engagement_engine.compute(
                            participant_id=label,
                            timestamp=timestamp,
                            visual_signal=visual_signal,
                            audio_signal=audio_signal,
                        )

                        alert = self.alert_detector.check(score)
                        if alert:
                            await self._send_json({
                                "type": "engagement_alert",
                                "data": {
                                    "participant": alert.participant_id,
                                    "score": alert.score,
                                    "duration_below_threshold": alert.duration_below,
                                    "timestamp": alert.timestamp,
                                    "suggested_action": alert.suggested_action,
                                },
                            })

                logger.info(f"[{self.video_id[:8]}] Frame {timestamp:.1f}s: "
                            f"{len(result.get('participants', []))} participants, "
                            f"gestures={sum(len(p.get('gestures', [])) for p in result.get('participants', []))}")

                try:
                    os.remove(frame_path)
                except OSError:
                    pass

            except Exception as e:
                logger.error(f"[{self.video_id[:8]}] Frame analysis at {timestamp:.1f}s failed: {e}")

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

                logger.info(f"[{self.video_id[:8]}] Transcribing [{start:.1f}-{end:.1f}]...")

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
                        logger.error(f"[{self.video_id[:8]}] Transcription failed [{start:.1f}-{end:.1f}]: {transcript_data}")
                        continue

                    segments = transcript_data.get("segments", [])
                    accepted = 0
                    for seg in segments:
                        text = seg.get("text", "").strip()
                        if not text:
                            continue
                        # Filter Whisper hallucinations on silence
                        if text in self.WHISPER_HALLUCINATIONS:
                            continue
                        # Also filter if entire segment is just 1-2 word repetition
                        if len(text.split()) <= 2 and text.lower().rstrip('.!?,') in (
                            "you", "bye", "bye-bye", "thank you", "thanks", "hmm",
                            "uh", "um", "oh", "ah", "huh", "subscribe",
                        ):
                            continue
                        seg["start"] = seg.get("start", 0) + start
                        seg["end"] = seg.get("end", 0) + start
                        self._transcript_segments.append(seg)
                        self._speaker_assignment_segments.append(seg)
                        accepted += 1

                    logger.info(f"[{self.video_id[:8]}] Transcribed [{start:.1f}-{end:.1f}]: {accepted}/{len(segments)} segments (filtered)")

                    # Immediately deliver
                    await self._send_pending_transcript(float("inf"))

                    # Voice/word signal analysis
                    chunk_text = " ".join(s.get("text", "") for s in segments)
                    if chunk_text.strip():
                        task = asyncio.create_task(self._analyze_audio_signals(chunk_text, (start + end) / 2))
                        task.add_done_callback(self._log_task_exception)

                    # Batch speaker assignment every ~10s
                    if (end - self._last_speaker_assignment_ts >= self._speaker_assignment_interval
                            and self._speaker_assignment_segments
                            and self._name_map):
                        asyncio.create_task(self._assign_speakers_batch())

                    try:
                        os.remove(chunk_path)
                    except OSError:
                        pass

                except Exception as e:
                    logger.error(f"[{self.video_id[:8]}] Transcription [{start:.1f}-{end:.1f}] failed: {e}")

                # Check if more audio was queued
                if self._transcribe_queue <= self._audio_transcribed_up_to:
                    break

        finally:
            self._active_transcribe_count -= 1

    async def _analyze_audio_features(self, chunk_path: str, start: float, end: float):
        """Analyze audio features from a transcription chunk."""
        try:
            features = self.audio_analyzer.analyze_file(chunk_path)
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
            if not self._cancelled and not self._finalizing:
                await self._send_json({"type": "audio_features", "data": audio_data})
        except Exception as e:
            logger.warning(f"[{self.video_id[:8]}] Audio analysis failed: {e}")

    async def _assign_speakers_batch(self):
        """Batch-assign real speaker names to accumulated transcript segments."""
        if not self._name_map or not self._speaker_assignment_segments:
            return

        segments_to_assign = self._speaker_assignment_segments.copy()
        self._speaker_assignment_segments.clear()
        self._last_speaker_assignment_ts = segments_to_assign[-1].get("end", 0)

        known_names = list(self._name_map.values())
        try:
            assigned = await assign_speakers_to_names(segments_to_assign, known_names)
            for seg in assigned:
                if seg.get("speaker"):
                    await self._send_json({
                        "type": "speaker_update",
                        "data": {
                            "start_time": seg.get("start", 0),
                            "speaker": seg["speaker"],
                        },
                    })
            logger.info(f"[{self.video_id[:8]}] Assigned speakers to {len(assigned)} segments")
        except Exception as e:
            logger.error(f"[{self.video_id[:8]}] Speaker assignment failed: {e}")

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
                await self._send_json({"type": "voice", "data": voice_result})

            if not isinstance(words_result, Exception):
                words_result["timestamp"] = timestamp
                self.word_signals.append(words_result)
                await self._send_json({"type": "words", "data": words_result})

        except Exception as e:
            logger.error(f"[{self.video_id[:8]}] Audio signal analysis failed: {e}")

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
            logger.error(f"[{self.video_id[:8]}] Personality analysis failed: {e}")

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
            logger.error(f"[{self.video_id[:8]}] Correlation analysis failed: {e}")

    # ─── Finalization ─────────────────────────────────────────────

    async def finalize(self):
        """On video_ended: wait for pending, generate summary + coaching, persist."""
        self.is_playing = False
        self._finalizing = True  # Prevent pending tasks from sending signals to chart
        logger.info(f"[{self.video_id[:8]}] Video ended — finalizing...")
        await self._send_json({"type": "video_ended_ack"})  # Tell frontend to freeze chart
        await self._send_status("Completing analysis...")

        # Wait for all in-flight analysis tasks
        if self._pending_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._pending_tasks, return_exceptions=True),
                    timeout=60.0,
                )
            except asyncio.TimeoutError:
                logger.warning(f"[{self.video_id[:8]}] Pending tasks timed out after 60s")

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

        # Final speaker assignment for any remaining segments
        if self._speaker_assignment_segments and self._name_map:
            await self._assign_speakers_batch()
        await self._send_pending_transcript(float("inf"))

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
            participant_names = sorted(self.participant_labels) or ["Person 1"]
            # Use real names if available from OCR
            if self._name_map:
                participant_names = sorted(set(
                    self._name_map.get(name, name) for name in participant_names
                ))

            analysis = await generate_summary_and_flags(
                transcript_text[:8000],
                emotions_summary[:4000],
                participant_names,
            )

            # Send summary
            await self._send_json({
                "type": "summary",
                "data": {
                    "summary": analysis.get("summary", ""),
                    "key_topics": analysis.get("key_topics", []),
                    "overall_sentiment": analysis.get("overall_sentiment", "neutral"),
                },
            })

            # Generate coaching per participant
            coaching_per_participant: dict[str, list[dict]] = {}
            for name in participant_names:
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

            logger.info(f"[{self.video_id[:8]}] Finalization complete")
            await self._send_status("Analysis complete")
            await self._send_json({"type": "complete"})

        except Exception as e:
            logger.error(f"[{self.video_id[:8]}] Finalization error: {e}")
            await self._send_json({"type": "error", "data": {"message": f"Analysis error: {str(e)[:100]}"}})
            await self._send_status(f"Error finalizing: {str(e)[:100]}")
            await self._send_json({"type": "complete"})

    async def cleanup(self):
        """Cancel everything on disconnect."""
        self._cancelled = True
        if self._audio_task and not self._audio_task.done():
            self._audio_task.cancel()
        if self._detect_task and not self._detect_task.done():
            self._detect_task.cancel()
        for task in self._transcribe_tasks:
            if not task.done():
                task.cancel()
        for task in self._pending_tasks:
            task.cancel()

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

                participant_id_map = {}
                for ps in p_scores:
                    pid = str(uuid.uuid4())
                    name = ps.get("name", "Unknown")
                    participant_id_map[name] = pid
                    await db.execute(
                        """INSERT INTO participants (id, video_id, name, engagement_score,
                           sentiment_score, speaking_pct, clarity_score, rapport_score, energy_score)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            pid, self.video_id, name,
                            ps.get("engagement", 70),
                            ps.get("sentiment", 0.3),
                            ps.get("speaking_pct", 25),
                            ps.get("clarity", 70),
                            ps.get("rapport", 70),
                            ps.get("energy", 70),
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
                logger.info(f"[{self.video_id[:8]}] Results persisted to DB")

        except Exception as e:
            logger.error(f"[{self.video_id[:8]}] DB persistence error: {e}")
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    "UPDATE videos SET status = ? WHERE id = ?",
                    (f"error: {str(e)[:200]}", self.video_id),
                )
                await db.commit()

    # ─── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _clean_name(name: str) -> str:
        """Normalize name format: 'Chong, Sam' -> 'Sam Chong', 'Giorgio Manenti (E)' -> 'Giorgio Manenti'."""
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
        # Remove trailing ellipsis
        name = re.sub(r'\.\.\.$', '', name).strip()
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

    def _resolve_label(self, label: str) -> str:
        """Resolve a participant label to a real name using all available maps.

        If the name_map is established, ONLY returns known names.
        Unknown labels fall back to "Person N" from the map (by position) or the label itself
        only if no name_map exists yet.
        """
        if not self._name_map:
            return label  # No name map yet, pass through

        # Direct name_map lookup: "Person 1" -> "Robin Mills"
        if label in self._name_map:
            return self._name_map[label]
        # Check if label IS a known real name (already resolved)
        known_names = set(self._name_map.values())
        if label in known_names:
            return label
        # Initials lookup: "RM" -> "Robin Mills"
        if label in self._initials_map:
            return self._initials_map[label]
        # Partial name match: "Robin Mehta" might match "Robin Mills" (first name)
        if ' ' in label:
            first_name = label.split()[0]
            matches = [n for n in known_names if n.split()[0] == first_name]
            if len(matches) == 1:
                return matches[0]
        # Extract "Person N" number and map it
        m = re.match(r'Person\s+(\d+)', label)
        if m:
            idx = int(m.group(1))
            key = f"Person {idx}"
            if key in self._name_map:
                return self._name_map[key]
        # Unknown label — keep as-is (will be deduped/filtered by caller)
        return label

    def _log_task_exception(self, task: asyncio.Task):
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(f"[{self.video_id[:8]}] Background task failed: {exc}")

    async def _send_json(self, data: dict):
        try:
            await self.ws.send_json(data)
        except WebSocketDisconnect:
            self._cancelled = True
        except Exception as e:
            logger.warning(f"[{self.video_id[:8]}] WebSocket send failed: {e}")

    async def _send_status(self, message: str):
        logger.info(f"[{self.video_id[:8]}] Status: {message}")
        await self._send_json({"type": "status", "data": {"message": message}})
