# Design: Glass UI + Multi-Model Accuracy Engine

**Date:** 2026-02-17
**Status:** Approved

## Problem Statement

1. Dashboard KPI cards are static — no way to drill down into what scores mean or how they're computed
2. UI is inconsistent — some components use GlassCard, others use raw Tailwind
3. Accuracy is the #1 concern — single-model GPT calls can hallucinate, no cross-validation, no confidence scoring
4. Real Zoom/Teams recordings demand production-grade accuracy that a CEO can defend

## Design: Glass UI

### Expandable KPI Cards
Every KPI card clickable with framer-motion expand animation:
- **Overall Score** → channel breakdown (55/38/7), confidence level, rating band
- **Avg Engagement** → per-participant mini-bars, trend sparkline, highest/lowest moments
- **Sentiment** → progression over time, word distribution, top sentiment-driving phrases
- **Participants** → quick cards with scores, camera status, speaking time
- **Key Moments** → timeline visualization, severity color-coding

### Channel Score Cards Expandable
Visual/Audio/Text bars expand to show:
- Visual: emotion distribution, body language trends
- Audio: pitch, volume, speaking rate, pauses (from real librosa)
- Text: keyword frequency, sentiment distribution, key phrases

### Design System
- All components use GlassCard consistently
- framer-motion AnimatePresence for expand/collapse
- Hover micro-interactions (translateY, glow intensification)
- Chevron rotation indicator for expandable state
- Expanded panels get brighter glass (0.12 opacity) + gradient border

## Design: Multi-Model Voting Accuracy Engine

### Triple-Model Consensus (Frame Analysis)
- GPT-4o + GPT-4.1 + GPT-5 called concurrently for EVERY frame
- Consensus: majority vote for categorical, trimmed mean for numerical
- Confidence score 0-1 stored per snapshot

### Dual-Model Vote (Text Signals)
- Voice signals: GPT-4o-mini + GPT-4.1
- Word signals: GPT-4o-mini + GPT-4.1
- Personality: GPT-4.1 + GPT-5

### Cross-Signal Validation Rules
- Voice-Visual alignment check
- Speaking-Engagement coherence boost
- Camera-off reweighting (0% visual, 60% audio, 40% text)
- Sentiment consistency check

### DB Additions
- `signal_snapshots.confidence` column
- `voting_log` table for audit trail

### Frontend Confidence Display
- Colored dot indicator per signal (green/amber/red)
- Overall confidence in expanded KPI
- Hover tooltip showing model agreement details
