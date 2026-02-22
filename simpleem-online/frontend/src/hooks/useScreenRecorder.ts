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
      // Capture screen — request tab audio (Chrome supports this for browser tabs)
      const displayStream = await navigator.mediaDevices.getDisplayMedia({
        video: { displaySurface: 'browser' } as MediaTrackConstraints,
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        } as MediaTrackConstraints,
      });

      // Also capture microphone audio as fallback (system audio often missing on macOS)
      let micStream: MediaStream | null = null;
      try {
        micStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      } catch {
        console.warn('Microphone access denied — recording without mic audio');
      }

      // Capture audio directly from any playing <video> elements in the DOM.
      // This is the reliable way to get video playback audio on macOS,
      // where getDisplayMedia does NOT capture system/tab audio for screen/window shares.
      const videoElementTracks: MediaStreamTrack[] = [];
      try {
        const videoElements = document.querySelectorAll('video');
        for (const videoEl of videoElements) {
          if (!videoEl.paused && !videoEl.muted && videoEl.readyState >= 2) {
            try {
              const videoStream = (videoEl as HTMLVideoElement & { captureStream(): MediaStream }).captureStream();
              const audioFromVideo = videoStream.getAudioTracks();
              videoElementTracks.push(...audioFromVideo);
            } catch {
              // captureStream may fail for cross-origin videos — skip silently
            }
          }
        }
      } catch {
        // querySelectorAll or iteration failed — continue without video element audio
      }

      // Merge all audio sources: display audio + microphone + video element audio
      const audioTracks = [
        ...displayStream.getAudioTracks(),
        ...(micStream?.getAudioTracks() || []),
        ...videoElementTracks,
      ];

      let stream: MediaStream;
      if (audioTracks.length > 0) {
        setHasAudio(true);
        // Mix audio tracks using AudioContext
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
          // Fallback: use display stream as-is
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

      // Stop recording if user stops screen share via browser UI
      displayStream.getVideoTracks()[0].addEventListener('ended', () => {
        // Also clean up mic stream
        micStream?.getTracks().forEach((t) => t.stop());
        stopRecording();
      });

      recorder.start(1000); // Collect data every 1s
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

  // Cleanup on unmount
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
