import { useState, useCallback, useRef, useEffect } from 'react';
import { Upload, X, Link, Video, Circle, Square } from 'lucide-react';
import GlassCard from '../ui/GlassCard';
import ProgressRing from '../ui/ProgressRing';
import { uploadVideo, importVideoUrl } from '../../lib/api';

type Tab = 'upload' | 'link' | 'record';

interface UploadStepProps {
  onUploadComplete: (videoId: string) => void;
}

export default function UploadStep({ onUploadComplete }: UploadStepProps) {
  const [activeTab, setActiveTab] = useState<Tab>('upload');

  // ── Upload File state ──
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // ── Meeting Link state ──
  const [meetingUrl, setMeetingUrl] = useState('');
  const [linkLoading, setLinkLoading] = useState(false);
  const [linkError, setLinkError] = useState<string | null>(null);

  // ── Screen Record state ──
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [recordUploading, setRecordUploading] = useState(false);
  const [recordProgress, setRecordProgress] = useState(0);
  const [recordError, setRecordError] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Clean up timer on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  // ── Upload File handlers ──
  const handleFile = useCallback(async (file: File) => {
    setSelectedFile(file);
    setUploading(true);
    setProgress(0);
    setError(null);
    try {
      const { id } = await uploadVideo(file, (pct) => setProgress(pct));
      onUploadComplete(id);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Upload failed');
      setUploading(false);
    }
  }, [onUploadComplete]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  // ── Meeting Link handler ──
  const handleMeetingLink = useCallback(async () => {
    const trimmed = meetingUrl.trim();
    if (!trimmed) return;
    setLinkLoading(true);
    setLinkError(null);
    try {
      const { id } = await importVideoUrl(trimmed);
      onUploadComplete(id);
    } catch (e) {
      setLinkError(e instanceof Error ? e.message : 'Failed to import meeting link');
      setLinkLoading(false);
    }
  }, [meetingUrl, onUploadComplete]);

  // ── Screen Record handlers ──
  const formatElapsed = (secs: number) => {
    const m = Math.floor(secs / 60).toString().padStart(2, '0');
    const s = (secs % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  const startRecording = useCallback(async () => {
    setRecordError(null);
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
      streamRef.current = stream;
      chunksRef.current = [];

      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        // Stop all tracks
        stream.getTracks().forEach((t) => t.stop());
        streamRef.current = null;

        // Stop timer
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
        }
        setRecording(false);

        // Build file
        const blob = new Blob(chunksRef.current, { type: 'video/webm' });
        const now = new Date();
        const pad2 = (n: number) => n.toString().padStart(2, '0');
        const filename = `screen-recording-${now.getFullYear()}-${pad2(now.getMonth() + 1)}-${pad2(now.getDate())}-${pad2(now.getHours())}${pad2(now.getMinutes())}.webm`;
        const file = new File([blob], filename, { type: 'video/webm' });

        // Upload
        setRecordUploading(true);
        setRecordProgress(0);
        try {
          const { id } = await uploadVideo(file, (pct) => setRecordProgress(pct));
          onUploadComplete(id);
        } catch (e) {
          setRecordError(e instanceof Error ? e.message : 'Upload failed');
          setRecordUploading(false);
        }
      };

      // Handle user clicking browser's native "Stop sharing" button
      stream.getVideoTracks()[0].onended = () => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
          mediaRecorderRef.current.stop();
        }
      };

      mediaRecorder.start();
      setRecording(true);
      setElapsed(0);
      timerRef.current = setInterval(() => setElapsed((prev) => prev + 1), 1000);
    } catch (e) {
      setRecordError(e instanceof Error ? e.message : 'Could not start screen recording');
    }
  }, [onUploadComplete]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
  }, []);

  // ── Tab button component ──
  const tabs: { key: Tab; label: string; icon: React.ReactNode }[] = [
    { key: 'upload', label: 'Upload File', icon: <Upload className="w-4 h-4" /> },
    { key: 'link', label: 'Meeting Link', icon: <Link className="w-4 h-4" /> },
    { key: 'record', label: 'Record Screen', icon: <Video className="w-4 h-4" /> },
  ];

  return (
    <div className="max-w-2xl mx-auto">
      <GlassCard className="overflow-hidden">
        <div className="text-center mb-6">
          <h2 className="text-2xl font-bold gradient-text">Upload Meeting Recording</h2>
          <p className="text-text-muted text-sm mt-2">Upload a file, paste a meeting link, or record your screen</p>
        </div>

        {/* Tab bar */}
        <div className="flex gap-2 mb-6">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl border text-sm font-medium transition-all duration-200 ${
                activeTab === tab.key
                  ? 'bg-white/10 border-white/20 text-text-primary'
                  : 'bg-transparent border-transparent text-text-muted hover:bg-white/5'
              }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>

        {/* ── Tab 1: Upload File ── */}
        {activeTab === 'upload' && (
          <>
            {uploading ? (
              <div className="flex flex-col items-center py-12 gap-4">
                <ProgressRing progress={progress} size={140} label="Uploading..." />
                <p className="text-text-secondary text-sm">{selectedFile?.name}</p>
              </div>
            ) : (
              <div
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                onClick={() => inputRef.current?.click()}
                className={`relative border-2 border-dashed rounded-2xl p-16 text-center cursor-pointer transition-all duration-300 ${
                  dragOver
                    ? 'border-accent bg-accent/5 shadow-glow-purple'
                    : 'border-white/10 hover:border-accent/50 hover:bg-white/[0.02]'
                }`}
              >
                <div className={`w-16 h-16 mx-auto rounded-2xl flex items-center justify-center mb-4 transition-all ${
                  dragOver ? 'bg-gradient-primary shadow-glow-purple' : 'bg-white/5'
                }`}>
                  <Upload className={`w-8 h-8 ${dragOver ? 'text-white' : 'text-text-muted'}`} />
                </div>
                <p className="text-text-primary font-medium">
                  {dragOver ? 'Drop to upload' : 'Click or drag video file here'}
                </p>
                <p className="text-text-muted text-xs mt-2">MP4, AVI, MOV, MKV, WebM — up to 2GB</p>
                <input
                  ref={inputRef}
                  type="file"
                  accept="video/*,audio/*"
                  className="hidden"
                  onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
                />
              </div>
            )}

            {error && (
              <div className="mt-4 p-3 rounded-xl bg-danger/10 border border-danger/20 text-red-400 text-sm flex items-center gap-2">
                <X className="w-4 h-4" />
                {error}
              </div>
            )}
          </>
        )}

        {/* ── Tab 2: Meeting Link ── */}
        {activeTab === 'link' && (
          <div className="py-4">
            {linkLoading ? (
              <div className="flex flex-col items-center py-12 gap-4">
                <div className="w-16 h-16 rounded-2xl bg-white/5 flex items-center justify-center">
                  <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
                </div>
                <p className="text-text-secondary text-sm">Downloading meeting recording...</p>
                <p className="text-text-muted text-xs">This may take a few minutes depending on the recording length</p>
              </div>
            ) : (
              <>
                <div className="flex gap-3">
                  <input
                    type="text"
                    value={meetingUrl}
                    onChange={(e) => setMeetingUrl(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleMeetingLink()}
                    placeholder="Paste Zoom, Teams, or Meet recording link..."
                    className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-text-primary placeholder:text-text-muted text-sm focus:outline-none focus:border-accent/50 focus:ring-1 focus:ring-accent/25 transition-all"
                  />
                  <button
                    onClick={handleMeetingLink}
                    disabled={!meetingUrl.trim()}
                    className="px-6 py-3 rounded-xl bg-gradient-to-r from-indigo-500 via-purple-500 to-cyan-500 text-white font-medium text-sm hover:shadow-glow-purple transition-all duration-300 disabled:opacity-40 disabled:cursor-not-allowed whitespace-nowrap"
                  >
                    Analyze Meeting
                  </button>
                </div>
                <p className="text-text-muted text-xs mt-3">
                  Supports Zoom, Microsoft Teams, Google Meet, YouTube, and SharePoint links
                </p>
              </>
            )}

            {linkError && (
              <div className="mt-4 p-3 rounded-xl bg-danger/10 border border-danger/20 text-red-400 text-sm flex items-center gap-2">
                <X className="w-4 h-4" />
                {linkError}
              </div>
            )}
          </div>
        )}

        {/* ── Tab 3: Record Screen ── */}
        {activeTab === 'record' && (
          <div className="py-4">
            {recordUploading ? (
              <div className="flex flex-col items-center py-12 gap-4">
                <ProgressRing progress={recordProgress} size={140} label="Uploading..." />
                <p className="text-text-secondary text-sm">Uploading screen recording...</p>
              </div>
            ) : recording ? (
              <div className="flex flex-col items-center py-12 gap-6">
                <div className="flex items-center gap-3">
                  <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
                  <span className="text-red-400 font-medium text-lg">Recording</span>
                </div>
                <span className="text-4xl font-mono text-text-primary tracking-wider">{formatElapsed(elapsed)}</span>
                <button
                  onClick={stopRecording}
                  className="flex items-center gap-2 px-6 py-3 rounded-xl bg-red-500/20 border border-red-500/30 text-red-400 font-medium text-sm hover:bg-red-500/30 transition-all duration-200"
                >
                  <Square className="w-4 h-4" />
                  Stop Recording
                </button>
              </div>
            ) : (
              <div className="flex flex-col items-center py-12 gap-4">
                <button
                  onClick={startRecording}
                  className="flex items-center gap-3 px-8 py-4 rounded-2xl bg-white/5 border border-white/10 hover:bg-white/10 hover:border-white/20 transition-all duration-300 group"
                >
                  <Circle className="w-5 h-5 text-red-500 fill-red-500 group-hover:scale-110 transition-transform" />
                  <span className="text-text-primary font-medium">Start Recording</span>
                </button>
                <p className="text-text-muted text-xs">Record your screen to capture a live meeting or demo</p>
              </div>
            )}

            {recordError && (
              <div className="mt-4 p-3 rounded-xl bg-danger/10 border border-danger/20 text-red-400 text-sm flex items-center gap-2">
                <X className="w-4 h-4" />
                {recordError}
              </div>
            )}
          </div>
        )}
      </GlassCard>
    </div>
  );
}
