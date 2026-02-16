import { useState, useEffect, useCallback, useRef } from 'react';
import { ChevronRight, Upload, FileText } from 'lucide-react';
import GlassCard from '../ui/GlassCard';
import GradientButton from '../ui/GradientButton';
import ProgressRing from '../ui/ProgressRing';
import { usePolling } from '../../hooks/usePolling';
import {
  getVideoStatus,
  getTranscript,
  getLiveTranscript,
  triggerTranscription,
  uploadTranscriptFile,
  uploadTranscriptText,
  type TranscriptSegment,
  type VideoStatus,
} from '../../lib/api';

const SPEAKER_COLORS = ['#8b5cf6', '#3b82f6', '#06b6d4', '#10b981', '#f59e0b', '#ec4899', '#ef4444', '#14b8a6'];

interface TranscriptStepProps {
  videoId: string;
  onContinue: () => void;
}

type Mode = 'transcribe' | 'upload';
type UploadTab = 'file' | 'paste';

export default function TranscriptStep({ videoId, onContinue }: TranscriptStepProps) {
  const [mode, setMode] = useState<Mode>('transcribe');
  const [uploadTab, setUploadTab] = useState<UploadTab>('file');

  // Transcribe mode state
  const [started, setStarted] = useState(false);
  const [done, setDone] = useState(false);
  const [transcript, setTranscript] = useState<TranscriptSegment[]>([]);
  const [liveSegments, setLiveSegments] = useState<TranscriptSegment[]>([]);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const prevCountRef = useRef(0);

  // Upload mode state
  const [dragOver, setDragOver] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [pasteText, setPasteText] = useState('');
  const [uploading, setUploading] = useState(false);

  const statusFetcher = useCallback(() => getVideoStatus(videoId), [videoId]);
  const { data: status } = usePolling<VideoStatus>(statusFetcher, 2000, started && !done);

  // Poll live transcript during transcription
  const liveFetcher = useCallback(() => getLiveTranscript(videoId), [videoId]);
  const { data: liveData } = usePolling<TranscriptSegment[]>(liveFetcher, 2000, started && !done);

  useEffect(() => {
    if (liveData && liveData.length > 0) {
      setLiveSegments(liveData);
      // Auto-scroll when new segments arrive
      if (liveData.length > prevCountRef.current && scrollRef.current) {
        setTimeout(() => {
          if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
          }
        }, 100);
      }
      prevCountRef.current = liveData.length;
    }
  }, [liveData]);

  // Auto-start transcription when in transcribe mode
  useEffect(() => {
    if (mode !== 'transcribe') return;
    const start = async () => {
      try {
        await triggerTranscription(videoId);
        setStarted(true);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to start transcription');
      }
    };
    start();
  }, [videoId, mode]);

  // Check if done
  useEffect(() => {
    if (status?.step === 'transcribed' || status?.step === 'complete') {
      setDone(true);
      getTranscript(videoId).then(setTranscript).catch(console.error);
    }
    if (status?.step === 'error') {
      setError(status.detail || 'Transcription failed');
    }
  }, [status, videoId]);

  const speakerMap = new Map<string, number>();
  const getSpeakerColor = (speaker: string) => {
    if (!speaker) return '#64748b';
    if (!speakerMap.has(speaker)) speakerMap.set(speaker, speakerMap.size);
    return SPEAKER_COLORS[speakerMap.get(speaker)! % SPEAKER_COLORS.length];
  };

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
  };

  const handleUpload = async () => {
    setUploading(true);
    setError(null);
    try {
      if (uploadTab === 'file' && uploadFile) {
        await uploadTranscriptFile(videoId, uploadFile);
        setDone(true);
        getTranscript(videoId).then(setTranscript).catch(console.error);
      } else if (uploadTab === 'paste' && pasteText.trim()) {
        await uploadTranscriptText(videoId, pasteText);
        setDone(true);
        getTranscript(videoId).then(setTranscript).catch(console.error);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) setUploadFile(file);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) setUploadFile(file);
  };

  // Render transcript list (reused for both live and final)
  const renderSegments = (segs: TranscriptSegment[]) => (
    <div ref={scrollRef} className="max-h-[400px] overflow-y-auto space-y-1 pr-2">
      {segs.map((seg, i) => (
        <div
          key={i}
          className="flex items-start gap-3 py-2 px-3 rounded-lg hover:bg-white/[0.03] transition-colors animate-fade-in"
        >
          <span className="text-text-muted text-xs font-mono w-12 flex-shrink-0 pt-0.5">
            {formatTime(seg.start_time)}
          </span>
          {seg.speaker && (
            <span
              className="text-xs font-semibold px-2 py-0.5 rounded-full flex-shrink-0"
              style={{
                color: getSpeakerColor(seg.speaker),
                backgroundColor: `${getSpeakerColor(seg.speaker)}15`,
                border: `1px solid ${getSpeakerColor(seg.speaker)}30`,
              }}
            >
              {seg.speaker}
            </span>
          )}
          <span className="text-text-primary text-sm leading-relaxed">{seg.text}</span>
        </div>
      ))}
    </div>
  );

  return (
    <div className="max-w-4xl mx-auto">
      <GlassCard>
        {/* Mode toggle - only show if not started transcription yet or in upload mode */}
        {!started && !done && (
          <div className="flex justify-center gap-2 mb-6">
            <button
              onClick={() => setMode('transcribe')}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all ${
                mode === 'transcribe'
                  ? 'bg-gradient-to-r from-indigo-500/20 to-purple-500/20 border border-purple-500/30 text-purple-300'
                  : 'bg-white/5 border border-white/10 text-text-muted hover:bg-white/10'
              }`}
            >
              <FileText className="w-4 h-4" />
              Transcribe Audio
            </button>
            <button
              onClick={() => setMode('upload')}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all ${
                mode === 'upload'
                  ? 'bg-gradient-to-r from-emerald-500/20 to-cyan-500/20 border border-emerald-500/30 text-emerald-300'
                  : 'bg-white/5 border border-white/10 text-text-muted hover:bg-white/10'
              }`}
            >
              <Upload className="w-4 h-4" />
              Upload Transcript
            </button>
          </div>
        )}

        <div className="text-center mb-6">
          <h2 className="text-2xl font-bold gradient-text">
            {done ? 'Review Transcript' : mode === 'upload' ? 'Upload Transcript' : 'Transcribing Audio'}
          </h2>
          <p className="text-text-muted text-sm mt-2">
            {done
              ? `${transcript.length} segments ready`
              : mode === 'upload'
              ? 'Upload a VTT, SRT, or text file from Meet/Teams/Zoom'
              : status?.detail || 'Preparing...'}
          </p>
        </div>

        {/* TRANSCRIBE MODE */}
        {mode === 'transcribe' && !done && !error && (
          <div className="space-y-6">
            <div className="flex flex-col items-center">
              <ProgressRing
                progress={status?.progress ?? 0}
                size={140}
                label={status?.detail || 'Starting...'}
              />
            </div>

            {/* Live transcript streaming */}
            {liveSegments.length > 0 && (
              <div className="animate-fade-in">
                <h3 className="text-xs font-semibold text-emerald-400 mb-2 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  Live Transcript ({liveSegments.length} segments)
                </h3>
                {renderSegments(liveSegments)}
              </div>
            )}
          </div>
        )}

        {/* UPLOAD MODE */}
        {mode === 'upload' && !done && (
          <div className="space-y-4">
            {/* Upload sub-tabs */}
            <div className="flex gap-2 justify-center">
              <button
                onClick={() => setUploadTab('file')}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  uploadTab === 'file'
                    ? 'bg-white/10 text-text-primary'
                    : 'text-text-muted hover:text-text-primary'
                }`}
              >
                File Upload
              </button>
              <button
                onClick={() => setUploadTab('paste')}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  uploadTab === 'paste'
                    ? 'bg-white/10 text-text-primary'
                    : 'text-text-muted hover:text-text-primary'
                }`}
              >
                Paste Text
              </button>
            </div>

            {uploadTab === 'file' ? (
              <div
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                className={`border-2 border-dashed rounded-2xl p-12 text-center transition-all cursor-pointer ${
                  dragOver
                    ? 'border-emerald-400/50 bg-emerald-500/5'
                    : uploadFile
                    ? 'border-purple-400/30 bg-purple-500/5'
                    : 'border-white/10 hover:border-white/20 hover:bg-white/[0.02]'
                }`}
                onClick={() => document.getElementById('transcript-file-input')?.click()}
              >
                <input
                  id="transcript-file-input"
                  type="file"
                  accept=".vtt,.srt,.txt"
                  onChange={handleFileSelect}
                  className="hidden"
                />
                {uploadFile ? (
                  <div>
                    <FileText className="w-10 h-10 mx-auto text-purple-400 mb-3" />
                    <p className="text-text-primary font-medium">{uploadFile.name}</p>
                    <p className="text-text-muted text-xs mt-1">
                      {(uploadFile.size / 1024).toFixed(1)} KB — Click to change
                    </p>
                  </div>
                ) : (
                  <div>
                    <Upload className="w-10 h-10 mx-auto text-text-muted mb-3" />
                    <p className="text-text-primary font-medium">Drop transcript file here</p>
                    <p className="text-text-muted text-xs mt-1">
                      Supports .vtt (Meet/Teams), .srt (Zoom), .txt
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <div>
                <textarea
                  value={pasteText}
                  onChange={(e) => setPasteText(e.target.value)}
                  placeholder={"Paste your transcript here...\n\nSupported formats:\n[00:01] Speaker: Hello everyone\nSpeaker: Welcome to the meeting\nOr just plain text..."}
                  className="w-full h-64 bg-white/5 border border-white/10 rounded-xl p-4 text-sm text-text-primary placeholder-text-muted/50 resize-none focus:outline-none focus:border-purple-500/30 transition-colors"
                />
                <p className="text-text-muted text-xs mt-1">
                  {pasteText.length > 0 ? `${pasteText.split('\n').filter(l => l.trim()).length} lines` : 'Paste transcript from Meet, Teams, or Zoom'}
                </p>
              </div>
            )}

            <div className="flex justify-end">
              <GradientButton
                onClick={handleUpload}
                disabled={uploading || (uploadTab === 'file' ? !uploadFile : !pasteText.trim())}
                icon={<ChevronRight className="w-4 h-4" />}
              >
                {uploading ? 'Processing...' : 'Use Transcript'}
              </GradientButton>
            </div>
          </div>
        )}

        {/* ERROR */}
        {error && (
          <div className="p-4 rounded-xl bg-danger/10 border border-danger/20 text-red-400 text-center">
            {error}
          </div>
        )}

        {/* DONE — Review transcript */}
        {done && transcript.length > 0 && (
          <>
            <div className="mb-6">
              {renderSegments(transcript)}
            </div>
            <div className="flex justify-end">
              <GradientButton onClick={onContinue} icon={<ChevronRight className="w-4 h-4" />}>
                Continue to Analysis
              </GradientButton>
            </div>
          </>
        )}
      </GlassCard>
    </div>
  );
}
