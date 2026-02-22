# Record Button on Home Page — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move the Record button to the home page so users can capture InsightAI dashboard demos with full audio.

**Architecture:** Extract the existing `useScreenRecorder` hook from `AnalysisDashboard.tsx` into a shared hook file, then add a Record UI section to `VideoUpload.tsx` on the landing page. The dashboard keeps its own Record button using the same hook.

**Tech Stack:** React 19, TypeScript, MediaRecorder API, getDisplayMedia, getUserMedia, AudioContext

---

### Task 1: Extract `useScreenRecorder` hook to its own file

**Files:**
- Create: `frontend/src/hooks/useScreenRecorder.ts`
- Modify: `frontend/src/components/AnalysisDashboard.tsx:27-162` (remove hook, add import)

**Step 1: Create the hook file**

Create `frontend/src/hooks/useScreenRecorder.ts` with the exact code from `AnalysisDashboard.tsx` lines 29-162, exported as a named export:

```typescript
import { useState, useRef, useEffect, useCallback } from 'react';

export function useScreenRecorder(videoName: string) {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [hasAudio, setHasAudio] = useState(true);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startRecording = useCallback(async () => {
    try {
      const displayStream = await navigator.mediaDevices.getDisplayMedia({
        video: { displaySurface: 'browser' } as MediaTrackConstraints,
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        } as MediaTrackConstraints,
      });

      let micStream: MediaStream | null = null;
      try {
        micStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      } catch {
        console.warn('Microphone access denied — recording without mic audio');
      }

      const audioTracks = [
        ...displayStream.getAudioTracks(),
        ...(micStream?.getAudioTracks() || []),
      ];

      let stream: MediaStream;
      if (audioTracks.length > 0) {
        setHasAudio(true);
        try {
          const audioCtx = new AudioContext();
          const dest = audioCtx.createMediaStreamDestination();
          for (const track of audioTracks) {
            const source = audioCtx.createMediaStreamSource(new MediaStream([track]));
            source.connect(dest);
          }
          stream = new MediaStream([
            ...displayStream.getVideoTracks(),
            ...dest.stream.getAudioTracks(),
          ]);
        } catch {
          stream = displayStream;
        }
      } else {
        setHasAudio(false);
        stream = displayStream;
      }

      streamRef.current = stream;
      chunksRef.current = [];

      const recorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported('video/webm;codecs=vp9,opus')
          ? 'video/webm;codecs=vp9,opus'
          : 'video/webm',
      });

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'video/webm' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        const safeName = videoName.replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 50);
        a.href = url;
        a.download = `${safeName}_recording_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.webm`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        chunksRef.current = [];
      };

      displayStream.getVideoTracks()[0].addEventListener('ended', () => {
        micStream?.getTracks().forEach((t) => t.stop());
        stopRecording();
      });

      recorder.start(1000);
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
      setRecordingTime(0);
      timerRef.current = setInterval(() => setRecordingTime((t) => t + 1), 1000);
    } catch (err) {
      console.warn('Screen recording cancelled or failed:', err);
    }
  }, [videoName]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setIsRecording(false);
    setRecordingTime(0);
  }, []);

  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  return { isRecording, recordingTime, startRecording, stopRecording, hasAudio };
}
```

**Step 2: Update AnalysisDashboard.tsx — remove inline hook, add import**

In `AnalysisDashboard.tsx`:
- Remove lines 27-162 (the `// ─── Screen Recording Hook ───` section and the `useScreenRecorder` function)
- Add import at top: `import { useScreenRecorder } from '../hooks/useScreenRecorder';`

The existing usage on line 185 (`const { isRecording, ... } = useScreenRecorder(video?.name || 'analysis');`) stays unchanged.

**Step 3: Verify the app still compiles**

Run: `cd frontend && npm run build`
Expected: No errors — dashboard Record button works exactly as before.

**Step 4: Commit**

```bash
git add frontend/src/hooks/useScreenRecorder.ts frontend/src/components/AnalysisDashboard.tsx
git commit -m "refactor: extract useScreenRecorder hook to shared file"
```

---

### Task 2: Add Record button to VideoUpload.tsx (home page)

**Files:**
- Modify: `frontend/src/components/VideoUpload.tsx`

**Step 1: Add import and hook usage**

At the top of `VideoUpload.tsx`, add:
```typescript
import { Circle, Square } from 'lucide-react';
import { useScreenRecorder } from '../hooks/useScreenRecorder';
```

Inside the component function, add:
```typescript
const { isRecording, recordingTime, startRecording, stopRecording, hasAudio } = useScreenRecorder('InsightAI_Recording');
```

**Step 2: Add Record UI section**

After the "Join Live Meeting" section (after the `{activeBotId && ...}` block, before the error display), add a new Record section:

```tsx
{/* Record Screen */}
<div className="flex items-center gap-3">
  {isRecording ? (
    <>
      <button
        onClick={stopRecording}
        className="flex-1 flex items-center justify-center gap-3 px-5 py-3 bg-red-600 hover:bg-red-700 text-white rounded-xl font-medium text-sm transition-all shadow-lg"
      >
        <Square className="w-4 h-4 fill-current" />
        Stop Recording — {Math.floor(recordingTime / 60)}:{(recordingTime % 60).toString().padStart(2, '0')}
      </button>
      {!hasAudio && (
        <span className="text-xs text-amber-400 whitespace-nowrap">No audio detected</span>
      )}
    </>
  ) : (
    <button
      onClick={startRecording}
      disabled={isProcessing}
      className="flex-1 flex items-center justify-center gap-3 px-5 py-3 glass-card border border-red-500/30 hover:border-red-500/60 hover:bg-red-500/10 text-text-primary rounded-xl font-medium text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed"
    >
      <Circle className="w-4 h-4 text-red-400 fill-red-400" />
      Record
    </button>
  )}
</div>
{isRecording && (
  <p className="text-xs text-text-muted text-center">
    Recording your screen with audio — share the InsightAI browser tab for best results
  </p>
)}
```

**Step 3: Verify the app compiles and renders**

Run: `cd frontend && npm run build`
Expected: No errors. Home page shows the Record button below the Join Meeting section.

**Step 4: Commit**

```bash
git add frontend/src/components/VideoUpload.tsx
git commit -m "feat: add Record button to home page for demo capture"
```

---

### Task 3: Manual verification

**Step 1: Start the app**

```bash
# Terminal 1 — Backend
cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-online" && ~/.pyenv/versions/3.10.14/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend
cd "/Users/anup.roy/Downloads/Simpleem Work/simpleem-online/frontend" && npm run dev
```

**Step 2: Verify home page Record button**

1. Open `http://localhost:5173`
2. Confirm "Record" button appears below the "Join Meeting" section
3. Click Record → browser should prompt to share a tab
4. Share the InsightAI tab (check "Share audio" checkbox in Chrome)
5. Confirm timer is counting up
6. Click "Stop Recording" → `.webm` file should download
7. Open the downloaded file — confirm it has video AND audio

**Step 3: Verify dashboard Record button still works**

1. Upload any video file
2. In the realtime dashboard, confirm the Record button still appears in the header
3. Click it — should work identically to before
