# Diarization-Based Speaking Distribution

**Goal:** Fix speaking distribution to show accurate multi-speaker percentages using pyannote diarization as source of truth.

**Root cause:** Current system relies on visual is_speaking detection (vision model detects one speaker per frame). Vision model is unreliable → always picks same person → 100% for one speaker.

**Fix:** Compute speaking time directly from diarization segments. Overwrite _speaking_durations when diarization completes. Before diarization, use transcript segment count.

## Changes

1. `_run_diarization()` → after diarization, compute per-speaker durations, overwrite _speaking_durations, send updated distribution to frontend
2. `_map_diarization_to_names()` → don't depend on _speaking_timeline; use name_map + order-based assignment as fallback
3. Remove visual is_speaking as source of speaking_pct (keep it for engagement/timeline only)
