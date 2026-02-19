# Design: Adaptive 3-Channel Engagement Judgment

**Date:** 2026-02-17
**Status:** Approved

## Problem

1. Camera-off participants get garbage visual data (engagement=30) at 55% weight — caps their score at ~45 even if they're the most active speaker
2. Audio channel is per-chunk (whole room), not per-speaker — camera-off person talking 80% of the time gets same audio score as silent person
3. Text channel uses sentiment as engagement proxy — negative sentiment ≠ disengaged
4. Summary fails silently due to GPT-5 API param incompatibility (fixed but needs re-analysis)

## Solution

### Part A — Adaptive Channel Weights

```
Camera ON:   55% visual + 38% audio + 7% text  (unchanged Mehrabian)
Camera OFF:   0% visual + 70% audio + 30% text  (audio-dominant)
```

Modify `EngagementEngine.compute()` to accept `camera_on: bool` and swap weights.

### Part B — Per-Speaker Audio Engagement

Replace global librosa `energy` with per-speaker metric derived from transcript:
- Speaking time ratio at timestamp (binary: are they talking?)
- Speaking frequency (segments in last 60s window)
- Word rate (words/sec when speaking)

### Part C — Text Channel = Participation

Replace `sentiment_score * 100` with composite:
- 40%: Speaking volume (words in chunk / max words any speaker)
- 30%: Question asking (? count)
- 30%: Sentiment magnitude (|score - 0.5| * 200 — strong feelings = engaged)

## Files to Change

- `engagement_engine.py` — adaptive weights, camera_on param
- `batch_pipeline.py` — per-speaker audio signal, text participation signal, pass camera_on
- `core42_client.py` — GPT-5 params already fixed
