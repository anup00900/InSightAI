# Record Button on Home Page with Audio Capture

**Date:** 2026-02-22
**Status:** Approved

## Problem

The Record button is hidden inside `AnalysisDashboard` (only visible in realtime mode after uploading a video). Users need a prominent Record button on the home/landing page to capture the InsightAI dashboard experience with full audio for demos.

The existing `useScreenRecorder` hook already handles tab audio capture (`getDisplayMedia({ audio: true })`) + microphone (`getUserMedia`), and mixes them via `AudioContext`. The main issues are accessibility (button location) and discoverability.

## Design

### 1. Extract `useScreenRecorder` to a shared hook file

- Move from `AnalysisDashboard.tsx` lines 29-162 to `hooks/useScreenRecorder.ts`
- Accept a `videoName` parameter (string) for the download filename
- No logic changes — same `getDisplayMedia` + `getUserMedia` + `AudioContext` mixing
- Export for use in both `VideoUpload.tsx` and `AnalysisDashboard.tsx`

### 2. Add Record button to `VideoUpload.tsx` (home page)

- New section below the "Join Meeting" input
- Red circle icon when idle, pulsing animation when recording
- Timer display showing `MM:SS` while recording
- Stop button to end recording (triggers `.webm` download)
- Helper text: "Records your screen with audio — share the InsightAI tab"
- Uses the extracted `useScreenRecorder` hook with name `"InsightAI_Recording"`

### 3. Keep existing Record button in AnalysisDashboard

- Import from extracted hook instead of inline definition
- No UI changes to the dashboard record button

## What does NOT change

- No backend changes
- No new API endpoints
- No changes to the analysis pipeline
- Recording output remains a local `.webm` download
- All existing functionality stays the same

## Technical notes

- `getDisplayMedia({ audio: true })` works in Chrome/Edge when user shares a browser tab
- macOS cannot capture system audio via `getDisplayMedia` for screen/window shares — only tab shares provide audio
- The hook already handles the no-audio fallback gracefully
