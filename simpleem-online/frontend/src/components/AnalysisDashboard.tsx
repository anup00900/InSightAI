import { useState, useRef, useEffect } from 'react';
import { type Video, type AnalysisResults, type Participant, formatTime } from '../lib/api';
import { useRealtimeAnalysis } from '../hooks/useRealtimeAnalysis';
import EmotionTimeline from './EmotionTimeline';
import ParticipantSidebar from './ParticipantSidebar';
import SpeakingDonut from './SpeakingDonut';
import SkillRadar from './SkillRadar';
import FlagTimeline from './FlagTimeline';
import CoachingPanel from './CoachingPanel';
import EmotionsSignal from './signals/EmotionsSignal';
import BodyLanguageSignal from './signals/BodyLanguageSignal';
import GesturesSignal from './signals/GesturesSignal';
import ReactionsSignal from './signals/ReactionsSignal';
import VoiceSignal from './signals/VoiceSignal';
import WordsSignal from './signals/WordsSignal';
import PersonalitySignal from './signals/PersonalitySignal';
import CorrelationPanel from './signals/CorrelationPanel';
import CaptionOverlay from './CaptionOverlay';
import SignalWeightCard from './SignalWeightCard';
import {
  Loader2, Clock, Users, TrendingUp,
  FileText, Lightbulb, MessageSquare, BarChart3, Radio, Link2 as Link2Icon, Upload,
  Download, FileDown,
} from 'lucide-react';

// ─── Types ──────────────────────────────────────────────────────

interface Props {
  results: AnalysisResults | null;
  loading: boolean;
  videoId: string;
  video?: Video | null;
  mode?: 'realtime' | 'review' | 'url';
}

type Tab = 'summary' | 'coaching' | 'transcript' | 'insights';

export default function AnalysisDashboard({ results, loading, videoId, video, mode = 'review' }: Props) {
  const [activeTab, setActiveTab] = useState<Tab>('transcript');
  const [selectedParticipantId, setSelectedParticipantId] = useState<string | null>(null);
  const [exporting, setExporting] = useState<'pdf' | 'csv' | null>(null);
  const [generatingReport, setGeneratingReport] = useState(false);
  const transcriptRef = useRef<HTMLDivElement>(null);

  // Always call hook (React rules) — disabled in review mode
  const { state: rtState, attachVideo } = useRealtimeAnalysis(videoId, mode === 'realtime');

  // Auto-scroll transcript to bottom (only if user hasn't scrolled up)
  useEffect(() => {
    if (mode === 'realtime' && activeTab === 'transcript' && transcriptRef.current) {
      const el = transcriptRef.current;
      const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
      if (isNearBottom) {
        el.scrollTop = el.scrollHeight;
      }
    }
  }, [rtState.transcript.length, activeTab, mode]);

  // ─── URL Viewing Mode ──────────────────────────────────────────
  if (mode === 'url' && video?.source_url) {
    const isStreamingPage = video.source_url.includes('stream.aspx') || video.source_url.includes('embed.aspx');
    const isDownloading = video.status === 'downloading';
    const downloadFailed = video.status === 'download_failed' || video.status === 'url';

    return (
      <div className="space-y-6">
        <div className="flex items-center gap-6 flex-wrap">
          <h2 className="text-2xl font-bold text-text-primary flex-1 min-w-0 truncate">
            {video.name || 'Video from URL'}
          </h2>
          <div className="flex items-center gap-3">
            {isDownloading && (
              <span className="flex items-center gap-1.5 text-xs font-semibold text-amber-400 bg-amber-400/10 px-3 py-1.5 rounded-full">
                <Loader2 className="w-3 h-3 animate-spin" />
                Downloading...
              </span>
            )}
            {downloadFailed && (
              <span className="flex items-center gap-1.5 text-xs font-semibold text-red-400 bg-red-400/10 px-3 py-1.5 rounded-full">
                Download Failed
              </span>
            )}
            <span className="flex items-center gap-1.5 text-xs font-semibold text-blue-400 bg-blue-400/10 px-3 py-1.5 rounded-full">
              <Link2Icon className="w-3 h-3" />
              URL
            </span>
            <a
              href={video.source_url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 text-xs font-medium text-accent hover:text-accent/80 bg-accent/10 px-3 py-1.5 rounded-full transition-colors"
            >
              Open in new tab &rarr;
            </a>
          </div>
        </div>

        <div className="glass-card overflow-hidden">
          {isStreamingPage ? (
            <div className="relative">
              <iframe
                src={video.source_url}
                className="w-full aspect-video"
                allow="autoplay; fullscreen"
                allowFullScreen
              />
              <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/60 to-transparent px-4 py-3">
                <p className="text-xs text-white/70">
                  If the video doesn&apos;t load, use &quot;Open in new tab&quot; above (requires login).
                </p>
              </div>
            </div>
          ) : (
            <video controls className="w-full aspect-video bg-black" src={video.source_url} />
          )}
        </div>

        {isDownloading && (
          <div className="glass-card p-6 text-center">
            <div className="flex flex-col items-center gap-3">
              <Loader2 className="w-8 h-8 text-accent animate-spin" />
              <p className="text-sm text-text-primary font-medium">Downloading video for analysis...</p>
              <p className="text-xs text-text-muted">Once downloaded, full AI analysis will start automatically.</p>
            </div>
          </div>
        )}

        {downloadFailed && (
          <div className="glass-card p-6">
            <div className="flex flex-col items-center gap-4 text-center">
              <Upload className="w-8 h-8 text-amber-400 opacity-70" />
              <div>
                <p className="text-sm text-text-primary font-medium">
                  Could not download video automatically
                </p>
                <p className="text-xs text-text-muted mt-1">
                  This URL requires authentication (SharePoint/Teams login).
                  To get full AI analysis, use one of these options:
                </p>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-lg">
                <div className="glass-depth-1 p-4 rounded-xl text-left">
                  <p className="text-xs font-bold text-accent mb-1">Option 1: Download & Upload</p>
                  <ol className="text-[11px] text-text-muted space-y-1 list-decimal list-inside">
                    <li>Open the video link in your browser</li>
                    <li>Download the recording (usually via &quot;...&quot; menu)</li>
                    <li>Upload the downloaded file here</li>
                  </ol>
                </div>
                <div className="glass-depth-1 p-4 rounded-xl text-left">
                  <p className="text-xs font-bold text-accent mb-1">Option 2: Upload Cookies</p>
                  <ol className="text-[11px] text-text-muted space-y-1 list-decimal list-inside">
                    <li>Install a cookies.txt browser extension</li>
                    <li>Export cookies while logged into SharePoint</li>
                    <li>Upload the cookies.txt file below</li>
                  </ol>
                  <label className="mt-2 flex items-center gap-2 px-3 py-1.5 text-xs font-medium text-accent bg-accent/10 rounded-lg cursor-pointer hover:bg-accent/20 transition-colors w-fit">
                    <Upload className="w-3 h-3" />
                    Upload cookies.txt
                    <input
                      type="file"
                      accept=".txt"
                      className="hidden"
                      onChange={async (e) => {
                        const file = e.target.files?.[0];
                        if (!file) return;
                        const form = new FormData();
                        form.append('file', file);
                        try {
                          await fetch('/api/upload-cookies', { method: 'POST', body: form });
                          // Retry download
                          await fetch(`/api/import-url`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ url: video.source_url }),
                          });
                          alert('Cookies uploaded! Retrying download...');
                        } catch {
                          alert('Failed to upload cookies.');
                        }
                      }}
                    />
                  </label>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }

  // ─── Realtime Mode ──────────────────────────────────────────────
  if (mode === 'realtime') {
    // Filter out ghost "Person N" entries that don't have real names
    const rtParticipants: Participant[] = Array.from(rtState.participants.entries())
      .filter(([label]) => !label.match(/^Person \d+$/))
      .map(([label, data]) => ({
        id: data.id || label,
        video_id: videoId,
        name: data.name || label,
        engagement_score: data.engagement_score || 0,
        sentiment_score: data.sentiment_score || 0,
        speaking_pct: data.speaking_pct || 0,
        clarity_score: data.clarity_score || 0,
        rapport_score: data.rapport_score || 0,
        energy_score: data.energy_score || 0,
      }));

    const selectedParticipant = rtParticipants.find((p) => p.id === selectedParticipantId) || null;

    // Build video URL from the video object
    const videoSrc = video
      ? `/uploads/${video.id}_${video.filename}`
      : `/uploads/${videoId}`;

    const tabs: { key: Tab; label: string; icon: React.ReactNode }[] = [
      { key: 'transcript', label: 'Live Transcript', icon: <MessageSquare className="w-4 h-4" /> },
      { key: 'summary', label: 'Summary', icon: <FileText className="w-4 h-4" /> },
      { key: 'coaching', label: 'Coaching', icon: <Lightbulb className="w-4 h-4" /> },
      { key: 'insights', label: 'Insights', icon: <BarChart3 className="w-4 h-4" /> },
    ];

    return (
      <div className="space-y-6">
        {/* Engagement Alerts — hidden for demo (covered by Critical Moments) */}
        {/* <EngagementAlerts alerts={rtState.engagementAlerts} /> */}

        {/* Header */}
        <div className="flex items-center gap-6 flex-wrap">
          <h2 className="text-2xl font-bold text-text-primary flex-1 min-w-0 truncate">
            {video?.name || 'Live Analysis'}
          </h2>
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1.5 text-xs font-semibold text-white bg-gradient-to-r from-red-500 to-orange-500 px-3 py-1.5 rounded-full shadow-lg">
              <Radio className="w-3 h-3 animate-pulse" />
              LIVE
            </span>
            {/* Generate Report Button — works anytime */}
            <button
              onClick={async () => {
                setGeneratingReport(true);
                try {
                  // First generate summary on demand
                  await fetch(`/api/videos/${videoId}/generate-summary`, { method: 'POST' });
                  // Then download PDF
                  const res = await fetch(`/api/videos/${videoId}/export/pdf`);
                  if (res.ok) {
                    const blob = await res.blob();
                    const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
                    a.download = `report_${videoId.slice(0, 8)}.pdf`; a.click();
                    URL.revokeObjectURL(a.href);
                  }
                } catch { /* silent */ } finally { setGeneratingReport(false); }
              }}
              disabled={generatingReport}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-accent glass-depth-2 rounded-full transition-colors hover:border-accent/30 disabled:opacity-50"
            >
              {generatingReport ? <Loader2 className="w-3 h-3 animate-spin" /> : <FileDown className="w-3 h-3" />}
              {generatingReport ? 'Generating...' : 'Generate Report'}
            </button>
            <a
              href={videoSrc}
              download
              className="flex items-center gap-2 px-3 py-1.5 text-xs font-semibold text-accent glass-depth-2 rounded-full transition-colors hover:bg-white/10"
              title="Download original video with audio"
            >
              <Download className="w-3 h-3" />
              Download Video
            </a>
            <span className="text-xs text-text-muted glass-depth-1 px-3 py-1.5 rounded-full">
              {rtState.isConnected ? rtState.statusMessage : 'Connecting...'}
            </span>
            {rtParticipants.length > 0 && (
              <div className="flex items-center gap-2 text-text-secondary">
                <Users className="w-4 h-4" />
                <span className="text-sm font-medium">{rtParticipants.length} detected</span>
              </div>
            )}
          </div>
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-12 gap-6">
          {/* Left: Video + Signals + Charts */}
          <div className="col-span-12 lg:col-span-8 space-y-5">
            {/* Video Player */}
            <RealtimePlayer videoSrc={videoSrc} attachVideo={attachVideo} rtState={rtState} />

            {/* 7 Signal Panels */}
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-4 gap-3">
              <EmotionsSignal signals={rtState.signals || []} />
              <BodyLanguageSignal signals={rtState.signals || []} />
              <GesturesSignal signals={rtState.signals || []} />
              <ReactionsSignal signals={rtState.signals || []} />
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              <VoiceSignal voiceSignals={rtState.voiceSignals || []} audioFeatures={rtState.audioFeatures || []} hasAudio={rtState.hasAudio} />
              <WordsSignal wordSignals={rtState.wordSignals || []} hasAudio={rtState.hasAudio} />
              <PersonalitySignal personality={rtState.personalitySignals || null} />
            </div>

            {/* Emotion Timeline */}
            <EmotionTimeline emotions={rtState.emotionTimeline || []} participants={rtParticipants} isRealtime />

            {/* Flags + Correlations */}
            {(rtState.flags?.length > 0) && <FlagTimeline flags={rtState.flags} />}
            {(rtState.correlations?.length > 0) && <CorrelationPanel correlations={rtState.correlations} />}
          </div>

          {/* Right: Participants */}
          <div className="col-span-12 lg:col-span-4 space-y-5">
            <ParticipantSidebar
              participants={rtParticipants}
              selectedId={selectedParticipantId}
              onSelect={setSelectedParticipantId}
            />
            <SkillRadar participant={selectedParticipant} />
            <SpeakingDonut participants={rtParticipants} />
            {/* 55/38/7 Signal Weight Visualization */}
            <SignalWeightCard
              visualScore={rtParticipants.length > 0 ? Math.round(rtParticipants.reduce((s, p) => s + p.engagement_score, 0) / rtParticipants.length) : 0}
              audioScore={rtState.audioFeatures.length > 0 ? Math.round(rtState.audioFeatures[rtState.audioFeatures.length - 1].energy) : 0}
              verbalScore={rtState.wordSignals.length > 0 ? Math.round((rtState.wordSignals[rtState.wordSignals.length - 1]?.sentiment_score || 0.5) * 100) : 0}
            />
          </div>
        </div>

        {/* Export buttons when analysis is complete */}
        {rtState.isComplete && (
          <div className="flex items-center gap-3 p-4 glass-card">
            <span className="text-sm text-text-secondary flex-1">Analysis complete. Export your results:</span>
            <button
              onClick={() => {
                fetch(`/api/videos/${videoId}/export/pdf`).then(r => r.blob()).then(b => {
                  const url = URL.createObjectURL(b);
                  const a = document.createElement('a'); a.href = url;
                  a.download = `report_${videoId.slice(0, 8)}.pdf`; a.click();
                  URL.revokeObjectURL(url);
                });
              }}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-text-primary glass-depth-2 rounded-lg hover:border-accent/50 transition-colors"
            >
              <Download className="w-3.5 h-3.5" /> PDF Report
            </button>
            <button
              onClick={() => {
                fetch(`/api/videos/${videoId}/export/csv`).then(r => r.blob()).then(b => {
                  const url = URL.createObjectURL(b);
                  const a = document.createElement('a'); a.href = url;
                  a.download = `data_${videoId.slice(0, 8)}.zip`; a.click();
                  URL.revokeObjectURL(url);
                });
              }}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-text-primary glass-depth-2 rounded-lg hover:border-accent/50 transition-colors"
            >
              <Download className="w-3.5 h-3.5" /> CSV Data
            </button>
          </div>
        )}

        {/* Tabs */}
        <div className="glass-card overflow-hidden">
          <div className="flex border-b border-white/[0.08]">
            {tabs.map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={`
                  flex items-center gap-2 px-6 py-4 text-sm font-medium transition-colors border-b-2
                  ${activeTab === tab.key
                    ? 'text-accent border-accent bg-accent/5'
                    : 'text-text-muted border-transparent hover:text-text-secondary'
                  }
                `}
              >
                {tab.icon}
                {tab.label}
                {tab.key === 'transcript' && rtState.transcript.length > 0 && (
                  <span className="text-[10px] bg-accent/20 text-accent px-1.5 py-0.5 rounded-full font-mono">
                    {rtState.transcript.length}
                  </span>
                )}
              </button>
            ))}
          </div>

          <div className="p-6">
            {activeTab === 'transcript' && (
              <div ref={transcriptRef} className="space-y-2 max-h-[500px] overflow-y-auto pr-2">
                {(rtState.transcript?.length || 0) === 0 ? (
                  <div className="flex flex-col items-center justify-center py-12 text-text-muted">
                    <MessageSquare className="w-8 h-8 mb-3 opacity-30" />
                    <p className="text-sm">
                      {rtState.hasAudio === false
                        ? 'No audio track in this video — transcript unavailable'
                        : 'Press play to start live transcription'}
                    </p>
                  </div>
                ) : (
                  rtState.transcript.map((seg, i) => (
                    <div key={i} className="flex gap-3 p-3 rounded-lg hover:bg-bg-primary/50 transition-colors animate-fade-in">
                      <span className="text-xs text-text-muted font-mono shrink-0 pt-0.5 w-12">
                        {formatTime(seg.start_time)}
                      </span>
                      {seg.speaker && (
                        <span className="text-xs font-bold text-accent shrink-0 pt-0.5 w-20 truncate">
                          {seg.speaker}
                        </span>
                      )}
                      <p className="text-sm text-text-primary flex-1">{seg.text}</p>
                    </div>
                  ))
                )}
              </div>
            )}

            {activeTab === 'summary' && (
              <div className="space-y-6">
                {rtState.summary ? (
                  <>
                    <div className="prose prose-invert max-w-none">
                      <p className="text-text-primary text-sm leading-relaxed whitespace-pre-line">
                        {rtState.summary.summary}
                      </p>
                    </div>
                    {rtState.summary.key_topics.length > 0 && (
                      <div>
                        <h4 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-3">Key Topics</h4>
                        <div className="flex flex-wrap gap-2">
                          {rtState.summary.key_topics.map((topic, i) => (
                            <span key={i} className="px-3 py-1.5 text-xs font-medium text-accent bg-accent/10 rounded-full border border-accent/20">
                              {topic}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="flex flex-col items-center justify-center py-12 text-text-muted">
                    <FileText className="w-8 h-8 mb-3 opacity-30" />
                    <p className="text-sm">Summary will be generated when the video ends</p>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'coaching' && (
              rtState.isComplete && rtState.coaching.size > 0 ? (
                <div className="space-y-6">
                  {Array.from(rtState.coaching.entries()).map(([participant, items]) => (
                    <div key={participant}>
                      <h4 className="text-sm font-bold text-accent mb-3">{participant}</h4>
                      <div className="space-y-3">
                        {items.map((item, i) => (
                          <div key={i} className="flex items-start gap-3 p-4 glass-depth-1">
                            <div className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 bg-accent/10">
                              <Lightbulb className="w-4 h-4 text-accent" />
                            </div>
                            <div className="flex-1">
                              <p className="text-sm text-text-primary">{item.recommendation}</p>
                              <div className="flex items-center gap-2 mt-2">
                                <span className="text-[10px] font-bold uppercase px-2 py-0.5 rounded-full text-accent bg-accent/10">
                                  {item.category}
                                </span>
                                <span className="text-[10px] text-text-muted">Priority: {item.priority}/5</span>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-12 text-text-muted">
                  <Lightbulb className="w-8 h-8 mb-3 opacity-30" />
                  <p className="text-sm">{rtState.isComplete ? 'Generating coaching...' : 'Coaching recommendations available after video ends'}</p>
                </div>
              )
            )}

            {activeTab === 'insights' && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {rtParticipants.map((p) => (
                  <div key={p.id} className="p-4 glass-depth-1">
                    <p className="text-sm font-medium text-text-primary mb-3">{p.name}</p>
                    <div className="space-y-2">
                      {[
                        { label: 'Engagement', value: p.engagement_score },
                        { label: 'Clarity', value: p.clarity_score },
                        { label: 'Rapport', value: p.rapport_score },
                        { label: 'Energy', value: p.energy_score },
                      ].map((metric) => (
                        <div key={metric.label} className="flex items-center justify-between">
                          <span className="text-xs text-text-muted">{metric.label}</span>
                          <div className="flex items-center gap-2">
                            <div className="w-20 h-1.5 glass-depth-1 rounded-full overflow-hidden">
                              <div
                                className="h-full rounded-full transition-all duration-700"
                                style={{
                                  width: `${metric.value}%`,
                                  backgroundColor:
                                    metric.value >= 80 ? '#10b981' :
                                    metric.value >= 60 ? '#3b82f6' :
                                    metric.value >= 40 ? '#f59e0b' : '#ef4444',
                                }}
                              />
                            </div>
                            <span className="text-xs font-bold text-text-secondary tabular-nums w-8 text-right">
                              {Math.round(metric.value)}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
                {rtParticipants.length === 0 && (
                  <div className="col-span-full flex flex-col items-center justify-center py-12 text-text-muted">
                    <Users className="w-8 h-8 mb-3 opacity-30" />
                    <p className="text-sm">Participant data appears as frames are analyzed</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  // ─── Review Mode ────────────────────────────────────────────────

  if (loading || !results) {
    return (
      <div className="flex flex-col items-center justify-center py-32">
        <Loader2 className="w-12 h-12 text-accent animate-spin mb-4" />
        <p className="text-text-secondary text-lg">
          {loading ? 'Loading analysis...' : 'Processing your recording...'}
        </p>
      </div>
    );
  }

  const { video: vid, participants, emotions, transcript, flags, summary } = results;
  const selectedParticipant = participants.find((p) => p.id === selectedParticipantId) || null;

  const tabs: { key: Tab; label: string; icon: React.ReactNode }[] = [
    { key: 'summary', label: 'Summary', icon: <FileText className="w-4 h-4" /> },
    { key: 'coaching', label: 'Coaching', icon: <Lightbulb className="w-4 h-4" /> },
    { key: 'transcript', label: 'Transcript', icon: <MessageSquare className="w-4 h-4" /> },
    { key: 'insights', label: 'Insights', icon: <BarChart3 className="w-4 h-4" /> },
  ];

  const handleExport = async (format: 'pdf' | 'csv') => {
    setExporting(format);
    try {
      const res = await fetch(`/api/videos/${videoId}/export/${format}`);
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Export failed' }));
        alert(err.detail || 'Export failed');
        return;
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = format === 'pdf'
        ? `insightai_report_${videoId.slice(0, 8)}.pdf`
        : `insightai_data_${videoId.slice(0, 8)}.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch {
      alert('Export failed. Please try again.');
    } finally {
      setExporting(null);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-6 flex-wrap">
        <h2 className="text-2xl font-bold text-text-primary flex-1 min-w-0 truncate">{vid.name}</h2>
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2 text-text-secondary">
            <Clock className="w-4 h-4" />
            <span className="text-sm">{formatTime(vid.duration)}</span>
          </div>
          <div className="flex items-center gap-2 text-text-secondary">
            <Users className="w-4 h-4" />
            <span className="text-sm">{participants.length} participants</span>
          </div>
          <div className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-accent" />
            <span className="text-lg font-bold text-accent">{Math.round(vid.overall_score)}%</span>
          </div>
          {/* Export Buttons */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => handleExport('pdf')}
              disabled={exporting !== null}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-text-primary glass-depth-2 rounded-lg transition-colors disabled:opacity-50 hover:border-accent/30"
            >
              {exporting === 'pdf' ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Download className="w-3.5 h-3.5" />}
              PDF Report
            </button>
            <button
              onClick={() => handleExport('csv')}
              disabled={exporting !== null}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-text-primary glass-depth-2 rounded-lg transition-colors disabled:opacity-50 hover:border-accent/30"
            >
              {exporting === 'csv' ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Download className="w-3.5 h-3.5" />}
              CSV Data
            </button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-6">
        <div className="col-span-12 lg:col-span-8 space-y-6">
          <div className="glass-card overflow-hidden">
            <video controls className="w-full aspect-video bg-black" src={`/uploads/${vid.id}_${vid.filename}`} />
          </div>
          <EmotionTimeline emotions={emotions} participants={participants} />
          <FlagTimeline flags={flags} />
        </div>
        <div className="col-span-12 lg:col-span-4 space-y-6">
          <ParticipantSidebar participants={participants} selectedId={selectedParticipantId} onSelect={setSelectedParticipantId} />
          <SkillRadar participant={selectedParticipant} />
          <SpeakingDonut participants={participants} />
        </div>
      </div>

      <div className="glass-card overflow-hidden">
        <div className="flex border-b border-white/[0.08]">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`flex items-center gap-2 px-6 py-4 text-sm font-medium transition-colors border-b-2 ${
                activeTab === tab.key
                  ? 'text-accent border-accent bg-accent/5'
                  : 'text-text-muted border-transparent hover:text-text-secondary'
              }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>
        <div className="p-6">
          {activeTab === 'summary' && (
            <div className="space-y-6">
              {summary ? (
                <>
                  <p className="text-text-primary text-sm leading-relaxed whitespace-pre-line">{summary.summary}</p>
                  {summary.key_topics.length > 0 && (
                    <div>
                      <h4 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-3">Key Topics</h4>
                      <div className="flex flex-wrap gap-2">
                        {summary.key_topics.map((t, i) => (
                          <span key={i} className="px-3 py-1.5 text-xs font-medium text-accent bg-accent/10 rounded-full border border-accent/20">{t}</span>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <p className="text-text-muted text-sm">Summary not available.</p>
              )}
            </div>
          )}
          {activeTab === 'coaching' && <CoachingPanel videoId={videoId} participants={participants} />}
          {activeTab === 'transcript' && (
            <div className="space-y-2 max-h-[500px] overflow-y-auto pr-2">
              {transcript.length === 0 ? (
                <p className="text-text-muted text-sm text-center py-8">No transcript available</p>
              ) : (
                transcript.map((seg, i) => (
                  <div key={i} className="flex gap-3 p-3 rounded-lg hover:bg-bg-primary/50 transition-colors">
                    <span className="text-xs text-text-muted font-mono shrink-0 pt-0.5 w-12">{formatTime(seg.start_time)}</span>
                    {seg.speaker && <span className="text-xs font-bold text-accent shrink-0 pt-0.5 w-20 truncate">{seg.speaker}</span>}
                    <p className="text-sm text-text-primary flex-1">{seg.text}</p>
                  </div>
                ))
              )}
            </div>
          )}
          {activeTab === 'insights' && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {participants.map((p) => (
                <div key={p.id} className="p-4 glass-depth-1">
                  <p className="text-sm font-medium text-text-primary mb-3">{p.name}</p>
                  <div className="space-y-2">
                    {[
                      { label: 'Engagement', value: p.engagement_score },
                      { label: 'Clarity', value: p.clarity_score },
                      { label: 'Rapport', value: p.rapport_score },
                      { label: 'Energy', value: p.energy_score },
                    ].map((m) => (
                      <div key={m.label} className="flex items-center justify-between">
                        <span className="text-xs text-text-muted">{m.label}</span>
                        <div className="flex items-center gap-2">
                          <div className="w-20 h-1.5 glass-depth-1 rounded-full overflow-hidden">
                            <div className="h-full rounded-full" style={{ width: `${m.value}%`, backgroundColor: m.value >= 80 ? '#10b981' : m.value >= 60 ? '#3b82f6' : m.value >= 40 ? '#f59e0b' : '#ef4444' }} />
                          </div>
                          <span className="text-xs font-bold text-text-secondary tabular-nums w-8 text-right">{Math.round(m.value)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Inline Realtime Player ─────────────────────────────────────

function RealtimePlayer({ videoSrc, attachVideo, rtState }: {
  videoSrc: string;
  attachVideo: (el: HTMLVideoElement) => (() => void);
  rtState: ReturnType<typeof useRealtimeAnalysis>['state'];
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [captionsEnabled, setCaptionsEnabled] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    if (!videoRef.current) return;
    const cleanup = attachVideo(videoRef.current);

    // Track current time for captions
    const video = videoRef.current;
    const onTimeUpdate = () => setCurrentTime(video.currentTime);
    video.addEventListener('timeupdate', onTimeUpdate);

    return () => {
      cleanup();
      video.removeEventListener('timeupdate', onTimeUpdate);
    };
  }, [attachVideo]);

  return (
    <div className="glass-card overflow-hidden relative group">
      <video ref={videoRef} controls className="w-full aspect-video bg-black" src={videoSrc} />
      <CaptionOverlay
        transcript={rtState.transcript}
        currentTime={currentTime}
        enabled={captionsEnabled}
        onToggle={() => setCaptionsEnabled(!captionsEnabled)}
      />
      <div className="absolute bottom-12 left-0 right-0 flex items-center justify-between px-4 py-1.5 bg-black/60 backdrop-blur-sm">
        <div className="flex items-center gap-2">
          {rtState.isAnalyzing && <Radio className="w-3 h-3 text-red-400 animate-pulse" />}
          <span className="text-xs text-white/70">{rtState.statusMessage}</span>
        </div>
        <span className={`w-2 h-2 rounded-full ${rtState.isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
      </div>
    </div>
  );
}
