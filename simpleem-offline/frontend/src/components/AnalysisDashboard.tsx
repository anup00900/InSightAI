import { useState, useEffect } from 'react';
import {
  type Video, type AnalysisResults, type Participant, type SignalSnapshot,
  type VoiceSignalData, type WordSignalData, type PersonalitySignalData,
  type CorrelationPattern, formatTime, getSignalSnapshots, getVoiceSignals,
  getWordSignals, getPersonalitySignals, getCorrelations,
} from '../lib/api';
import EmotionTimeline from './EmotionTimeline';
import ParticipantSidebar from './ParticipantSidebar';
import SpeakingDonut from './SpeakingDonut';
import SkillRadar from './SkillRadar';
import FlagTimeline from './FlagTimeline';
import CoachingPanel from './CoachingPanel';
import GlassCard from './ui/GlassCard';
import EmotionsSignal from './signals/EmotionsSignal';
import BodyLanguageSignal from './signals/BodyLanguageSignal';
import GesturesSignal from './signals/GesturesSignal';
import ReactionsSignal from './signals/ReactionsSignal';
import VoiceSignal from './signals/VoiceSignal';
import WordsSignal from './signals/WordsSignal';
import PersonalitySignal from './signals/PersonalitySignal';
import CorrelationPanel from './signals/CorrelationPanel';
import {
  Loader2, Clock, Users, TrendingUp,
  FileText, Lightbulb, MessageSquare, BarChart3,
  Download, Activity, Heart,
} from 'lucide-react';

// ─── Skeleton Placeholder ────────────────────────────────────────

function SectionSkeleton({ label }: { label: string }) {
  return (
    <GlassCard className="animate-pulse">
      <div className="flex items-center gap-2 mb-4">
        <div className="w-4 h-4 rounded bg-purple-500/30 animate-glow-pulse" />
        <span className="text-xs text-purple-400">{label}</span>
      </div>
      <div className="space-y-2">
        <div className="h-3 bg-white/10 rounded w-3/4" />
        <div className="h-3 bg-white/5 rounded w-1/2" />
        <div className="h-20 bg-white/5 rounded mt-4" />
      </div>
    </GlassCard>
  );
}

// ─── Types ──────────────────────────────────────────────────────

interface Props {
  results: AnalysisResults | null;
  loading: boolean;
  videoId: string;
  video?: Video | null;
}

type Tab = 'summary' | 'coaching' | 'transcript' | 'insights' | 'signals';

// ─── Tab color mapping ──────────────────────────────────────────

const tabColors: Record<Tab, { active: string; border: string; bg: string; icon: string }> = {
  summary:    { active: 'text-indigo-300',  border: 'from-indigo-400 to-purple-400',  bg: 'bg-indigo-500/10',  icon: 'text-indigo-400' },
  coaching:   { active: 'text-amber-300',   border: 'from-amber-400 to-orange-400',   bg: 'bg-amber-500/10',   icon: 'text-amber-400' },
  transcript: { active: 'text-purple-300',  border: 'from-purple-400 to-pink-400',    bg: 'bg-purple-500/10',  icon: 'text-purple-400' },
  insights:   { active: 'text-cyan-300',    border: 'from-cyan-400 to-blue-400',      bg: 'bg-cyan-500/10',    icon: 'text-cyan-400' },
  signals:    { active: 'text-emerald-300', border: 'from-emerald-400 to-teal-400',  bg: 'bg-emerald-500/10', icon: 'text-emerald-400' },
};

export default function AnalysisDashboard({ results, loading, videoId, video }: Props) {
  const [activeTab, setActiveTab] = useState<Tab>('transcript');
  const [selectedParticipantId, setSelectedParticipantId] = useState<string | null>(null);
  const [exporting, setExporting] = useState<'pdf' | 'csv' | null>(null);

  const [snapshots, setSnapshots] = useState<SignalSnapshot[]>([]);
  const [voiceSignals, setVoiceSignals] = useState<VoiceSignalData[]>([]);
  const [wordSignals, setWordSignals] = useState<WordSignalData[]>([]);
  const [personality, setPersonality] = useState<PersonalitySignalData | null>(null);
  const [correlations, setCorrelations] = useState<CorrelationPattern[]>([]);

  useEffect(() => {
    if (!results) return;
    Promise.all([
      getSignalSnapshots(videoId),
      getVoiceSignals(videoId),
      getWordSignals(videoId),
      getPersonalitySignals(videoId),
      getCorrelations(videoId),
    ]).then(([snap, voice, words, pers, corr]) => {
      setSnapshots(snap);
      setVoiceSignals(voice);
      setWordSignals(words);
      setPersonality(pers);
      setCorrelations(corr);
    });
  }, [results, videoId]);

  // ─── Loading / Empty State ──────────────────────────────────────

  if (loading || !results) {
    return (
      <div className="flex flex-col items-center justify-center py-32">
        <Loader2 className="w-12 h-12 text-purple-400 animate-spin mb-4" />
        <p className="text-slate-300 text-lg">
          {loading ? 'Loading analysis...' : 'Processing your recording...'}
        </p>
      </div>
    );
  }

  // ─── Review Mode Dashboard ──────────────────────────────────────

  const { video: vid, participants, emotions, transcript, flags, summary } = results;
  const selectedParticipant = participants.find((p) => p.id === selectedParticipantId) || null;

  const avgEngagement =
    participants.length > 0
      ? Math.round(participants.reduce((sum, p) => sum + p.engagement_score, 0) / participants.length)
      : 0;

  const sentimentLabel = summary?.overall_sentiment
    ? summary.overall_sentiment.charAt(0).toUpperCase() + summary.overall_sentiment.slice(1).toLowerCase()
    : 'N/A';

  const sentimentColor = (() => {
    const s = summary?.overall_sentiment?.toLowerCase();
    if (s === 'positive') return { text: 'text-emerald-400', gradient: 'from-emerald-400 to-green-300' };
    if (s === 'negative') return { text: 'text-red-400', gradient: 'from-red-400 to-rose-300' };
    return { text: 'text-amber-400', gradient: 'from-amber-400 to-yellow-300' };
  })();

  const tabs: { key: Tab; label: string; icon: React.ReactNode }[] = [
    { key: 'summary', label: 'Summary', icon: <FileText className="w-4 h-4" /> },
    { key: 'coaching', label: 'Coaching', icon: <Lightbulb className="w-4 h-4" /> },
    { key: 'transcript', label: 'Transcript', icon: <MessageSquare className="w-4 h-4" /> },
    { key: 'insights', label: 'Insights', icon: <BarChart3 className="w-4 h-4" /> },
    { key: 'signals' as Tab, label: 'Signals', icon: <Activity className="w-4 h-4" /> },
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
        ? `simpleem_report_${videoId.slice(0, 8)}.pdf`
        : `simpleem_data_${videoId.slice(0, 8)}.zip`;
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
      {/* ─── KPI Header Row ──────────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Overall Score */}
        <GlassCard className="text-center py-6" hover>
          <TrendingUp className="w-5 h-5 text-purple-400 mx-auto mb-2" />
          <p className="text-3xl font-black gradient-text">{Math.round(vid.overall_score)}%</p>
          <p className="text-xs text-slate-500 mt-1">Overall Score</p>
        </GlassCard>

        {/* Avg Engagement */}
        <GlassCard className="text-center py-6" hover>
          <Activity className="w-5 h-5 text-blue-400 mx-auto mb-2" />
          <p className="text-3xl font-black gradient-text">{avgEngagement}%</p>
          <p className="text-xs text-slate-500 mt-1">Avg Engagement</p>
        </GlassCard>

        {/* Sentiment */}
        <GlassCard className="text-center py-6" hover>
          <Heart className={`w-5 h-5 ${sentimentColor.text} mx-auto mb-2`} />
          <p className={`text-3xl font-black bg-gradient-to-r ${sentimentColor.gradient} bg-clip-text text-transparent`}>
            {sentimentLabel}
          </p>
          <p className="text-xs text-slate-500 mt-1">Sentiment</p>
        </GlassCard>

        {/* Participants */}
        <GlassCard className="text-center py-6" hover>
          <Users className="w-5 h-5 text-cyan-400 mx-auto mb-2" />
          <p className="text-3xl font-black gradient-text">{participants.length}</p>
          <p className="text-xs text-slate-500 mt-1">Participants</p>
        </GlassCard>
      </div>

      {/* Per-Channel Sub-Scores */}
      <div className="grid grid-cols-3 gap-3">
        <GlassCard className="text-center py-3" hover>
          <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Visual (55%)</div>
          <p className="text-lg font-bold text-indigo-400">{(() => {
            if (!snapshots.length) return '--';
            const avgVis = snapshots.reduce((sum, s) =>
              sum + s.participants.reduce((ps, p) => ps + p.emotions.engagement, 0) / Math.max(s.participants.length, 1),
            0) / snapshots.length;
            return Math.round(avgVis) + '%';
          })()}</p>
        </GlassCard>
        <GlassCard className="text-center py-3" hover>
          <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Audio (38%)</div>
          <p className="text-lg font-bold text-violet-400">{(() => {
            if (!voiceSignals.length) return '--';
            const avgAudio = voiceSignals.reduce((s, v) => s + v.energy, 0) / voiceSignals.length;
            return Math.round(avgAudio) + '%';
          })()}</p>
        </GlassCard>
        <GlassCard className="text-center py-3" hover>
          <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Text (7%)</div>
          <p className="text-lg font-bold text-cyan-400">{(() => {
            if (!wordSignals.length) return '--';
            const avgText = wordSignals.reduce((s, w) => s + w.sentiment_score * 100, 0) / wordSignals.length;
            return Math.round(avgText) + '%';
          })()}</p>
        </GlassCard>
      </div>

      {/* Engagement Timeline */}
      {snapshots.length > 1 && (
        <GlassCard>
          <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Engagement Timeline</h4>
          <div className="relative h-32">
            <svg viewBox={`0 0 ${snapshots.length * 10} 100`} className="w-full h-full" preserveAspectRatio="none">
              {/* Grid lines */}
              {[25, 50, 75].map(y => (
                <line key={y} x1="0" y1={100 - y} x2={snapshots.length * 10} y2={100 - y} stroke="rgba(255,255,255,0.05)" strokeWidth="0.5" />
              ))}
              {/* Per-participant engagement lines */}
              {participants.map((part, pi) => {
                const colors = ['#818cf8', '#c084fc', '#22d3ee', '#f59e0b', '#10b981', '#f43f5e'];
                const color = colors[pi % colors.length];
                const points = snapshots.map((s, i) => {
                  const pData = s.participants.find(p => p.label === part.name);
                  const eng = pData ? pData.emotions.engagement : 50;
                  return `${i * 10},${100 - eng}`;
                }).join(' ');
                return <polyline key={pi} points={points} fill="none" stroke={color} strokeWidth="1.5" opacity="0.8" />;
              })}
            </svg>
            {/* Legend */}
            <div className="flex flex-wrap gap-3 mt-2">
              {participants.map((p, i) => {
                const colors = ['#818cf8', '#c084fc', '#22d3ee', '#f59e0b', '#10b981', '#f43f5e'];
                return (
                  <span key={i} className="flex items-center gap-1 text-[10px] text-slate-400">
                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: colors[i % colors.length] }} />
                    {p.name}
                  </span>
                );
              })}
            </div>
          </div>
        </GlassCard>
      )}

      {/* Speaking Distribution */}
      {participants.some(p => p.speaking_pct > 0) && (
        <GlassCard>
          <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Speaking Distribution</h4>
          <div className="space-y-2">
            {participants.map((p, i) => {
              const colors = ['from-indigo-500 to-purple-500', 'from-violet-500 to-pink-500', 'from-cyan-500 to-blue-500', 'from-amber-500 to-orange-500', 'from-emerald-500 to-teal-500'];
              return (
                <div key={i} className="flex items-center gap-3">
                  <span className="text-xs text-slate-400 w-24 truncate">{p.name}</span>
                  <div className="flex-1 h-4 bg-white/5 rounded-full overflow-hidden">
                    <div className={`h-full rounded-full bg-gradient-to-r ${colors[i % colors.length]}`} style={{ width: `${p.speaking_pct}%` }} />
                  </div>
                  <span className="text-xs font-bold text-slate-400 w-10 text-right">{Math.round(p.speaking_pct)}%</span>
                </div>
              );
            })}
          </div>
        </GlassCard>
      )}

      {/* Engagement Alerts */}
      {flags.filter(f => f.type === 'engagement_alert').length > 0 && (
        <GlassCard>
          <h4 className="text-xs font-semibold text-red-400 uppercase tracking-wider mb-3">Engagement Alerts</h4>
          <div className="space-y-2">
            {flags.filter(f => f.type === 'engagement_alert').map((f, i) => (
              <div key={i} className="flex items-start gap-3 p-2 rounded-lg bg-red-500/5 border border-red-500/20">
                <span className="text-xs font-mono text-slate-500">{formatTime(f.timestamp)}</span>
                <p className="text-xs text-red-300 flex-1">{f.description}</p>
              </div>
            ))}
          </div>
        </GlassCard>
      )}

      {/* ─── Header ──────────────────────────────────────────────── */}
      <GlassCard className="flex items-center gap-6 flex-wrap">
        <div className="flex-1 min-w-0">
          <h2 className="text-xl font-bold gradient-text truncate">{vid.name}</h2>
          <div className="flex items-center gap-4 mt-1.5">
            <div className="flex items-center gap-1.5 text-slate-400">
              <Clock className="w-3.5 h-3.5" />
              <span className="text-xs">{formatTime(vid.duration)}</span>
            </div>
            <div className="flex items-center gap-1.5 text-slate-400">
              <Users className="w-3.5 h-3.5" />
              <span className="text-xs">{participants.length} participants</span>
            </div>
            <div className="flex items-center gap-1.5">
              <TrendingUp className="w-3.5 h-3.5 text-purple-400" />
              <span className="text-xs font-bold text-purple-300">{Math.round(vid.overall_score)}%</span>
            </div>
          </div>
        </div>
        {/* Export Buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => handleExport('pdf')}
            disabled={exporting !== null}
            className="flex items-center gap-1.5 px-4 py-2 text-xs font-semibold text-white bg-gradient-to-r from-indigo-500 to-purple-500 hover:from-indigo-400 hover:to-purple-400 rounded-lg transition-all shadow-lg shadow-purple-500/20 disabled:opacity-50"
          >
            {exporting === 'pdf' ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Download className="w-3.5 h-3.5" />}
            PDF Report
          </button>
          <button
            onClick={() => handleExport('csv')}
            disabled={exporting !== null}
            className="flex items-center gap-1.5 px-4 py-2 text-xs font-semibold text-white bg-gradient-to-r from-violet-500 to-cyan-500 hover:from-violet-400 hover:to-cyan-400 rounded-lg transition-all shadow-lg shadow-cyan-500/20 disabled:opacity-50"
          >
            {exporting === 'csv' ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Download className="w-3.5 h-3.5" />}
            CSV Data
          </button>
        </div>
      </GlassCard>

      {/* ─── Main Grid ───────────────────────────────────────────── */}
      <div className="grid grid-cols-12 gap-6">
        {/* Left: Video + Charts */}
        <div className="col-span-12 lg:col-span-8 space-y-6">
          <GlassCard padding={false} className="overflow-hidden">
            <video controls className="w-full aspect-video bg-black" src={`/uploads/${vid.id}_${vid.filename}`} />
          </GlassCard>
          {emotions.length > 0 ? (
            <EmotionTimeline emotions={emotions} participants={participants} />
          ) : (
            <SectionSkeleton label="Emotion Timeline" />
          )}
          {flags.length > 0 ? (
            <FlagTimeline flags={flags} />
          ) : (
            <SectionSkeleton label="Flag Timeline" />
          )}
        </div>

        {/* Right: Participants */}
        <div className="col-span-12 lg:col-span-4 space-y-6">
          <ParticipantSidebar participants={participants} selectedId={selectedParticipantId} onSelect={setSelectedParticipantId} />
          <SkillRadar participant={selectedParticipant} />
          <SpeakingDonut participants={participants} />
        </div>
      </div>

      {/* ─── Tabs ────────────────────────────────────────────────── */}
      <GlassCard padding={false} className="overflow-hidden">
        <div className="flex border-b border-white/10">
          {tabs.map((tab) => {
            const isActive = activeTab === tab.key;
            const colors = tabColors[tab.key];
            return (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={`relative flex items-center gap-2 px-6 py-4 text-sm font-medium transition-all ${
                  isActive
                    ? `${colors.active} ${colors.bg}`
                    : 'text-slate-500 hover:text-slate-300 hover:bg-white/5'
                }`}
              >
                <span className={isActive ? colors.icon : ''}>
                  {tab.icon}
                </span>
                {tab.label}
                {/* Gradient underline for active tab */}
                {isActive && (
                  <span
                    className={`absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r ${colors.border}`}
                  />
                )}
              </button>
            );
          })}
        </div>
        <div className="p-6">
          {activeTab === 'summary' && (
            <div className="space-y-6">
              {summary ? (
                <>
                  <p className="text-slate-200 text-sm leading-relaxed whitespace-pre-line">{summary.summary}</p>
                  {summary.key_topics.length > 0 && (
                    <div>
                      <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Key Topics</h4>
                      <div className="flex flex-wrap gap-2">
                        {summary.key_topics.map((t, i) => (
                          <span key={i} className="px-3 py-1.5 text-xs font-medium text-purple-300 bg-purple-500/10 rounded-full border border-purple-500/20">{t}</span>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <SectionSkeleton label="Summary" />
              )}
            </div>
          )}
          {activeTab === 'coaching' && <CoachingPanel videoId={videoId} participants={participants} />}
          {activeTab === 'transcript' && (
            <div className="space-y-2 max-h-[500px] overflow-y-auto pr-2">
              {transcript.length === 0 ? (
                <SectionSkeleton label="Transcript" />
              ) : (
                transcript.map((seg, i) => (
                  <div key={i} className="flex gap-3 p-3 rounded-lg hover:bg-white/5 transition-colors">
                    <span className="text-xs text-slate-500 font-mono shrink-0 pt-0.5 w-12">{formatTime(seg.start_time)}</span>
                    {seg.speaker && <span className="text-xs font-bold text-purple-400 shrink-0 pt-0.5 w-20 truncate">{seg.speaker}</span>}
                    <p className="text-sm text-slate-200 flex-1">{seg.text}</p>
                  </div>
                ))
              )}
            </div>
          )}
          {activeTab === 'insights' && (
            <div className="space-y-6">
              {/* Red Flags & Opportunities */}
              {flags.length > 0 && (
                <div className="mb-6">
                  <h4 className="text-sm font-semibold text-slate-300 mb-3">Red Flags & Opportunities</h4>
                  <div className="space-y-2">
                    {flags.slice(0, 5).map((f, i) => (
                      <div key={i} className="flex items-start gap-3 p-3 rounded-lg bg-white/5 border border-white/10">
                        <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${
                          f.severity === 'high' ? 'text-red-400 bg-red-500/10' :
                          f.severity === 'medium' ? 'text-amber-400 bg-amber-500/10' :
                          'text-emerald-400 bg-emerald-500/10'
                        }`}>{f.type}</span>
                        <p className="text-sm text-slate-300 flex-1">{f.description}</p>
                        <span className="text-xs text-slate-500 font-mono">{formatTime(f.timestamp)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Participant Insight Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {participants.map((p) => (
                  <div key={p.id} className="p-4 rounded-xl bg-white/5 border border-white/10 backdrop-blur-sm">
                    <p className="text-sm font-medium text-slate-200 mb-3">{p.name}</p>
                    <div className="space-y-2">
                      {[
                        { label: 'Engagement', value: p.engagement_score },
                        { label: 'Clarity', value: p.clarity_score },
                        { label: 'Rapport', value: p.rapport_score },
                        { label: 'Energy', value: p.energy_score },
                      ].map((m) => (
                        <div key={m.label} className="flex items-center justify-between">
                          <span className="text-xs text-slate-500">{m.label}</span>
                          <div className="flex items-center gap-2">
                            <div className="w-20 h-1.5 bg-white/10 rounded-full overflow-hidden">
                              <div className="h-full rounded-full transition-all duration-700" style={{ width: `${m.value}%`, backgroundColor: m.value >= 80 ? '#10b981' : m.value >= 60 ? '#818cf8' : m.value >= 40 ? '#f59e0b' : '#ef4444' }} />
                            </div>
                            <span className="text-xs font-bold text-slate-300 tabular-nums w-8 text-right">{Math.round(m.value)}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          {activeTab === 'signals' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="rounded-xl bg-white/5 border border-white/10">
                <EmotionsSignal snapshots={snapshots} />
              </div>
              <div className="rounded-xl bg-white/5 border border-white/10">
                <BodyLanguageSignal snapshots={snapshots} />
              </div>
              <div className="rounded-xl bg-white/5 border border-white/10">
                <GesturesSignal snapshots={snapshots} />
              </div>
              <div className="rounded-xl bg-white/5 border border-white/10">
                <ReactionsSignal snapshots={snapshots} />
              </div>
              <div className="rounded-xl bg-white/5 border border-white/10">
                <VoiceSignal signals={voiceSignals} />
              </div>
              <div className="rounded-xl bg-white/5 border border-white/10">
                <WordsSignal signals={wordSignals} />
              </div>
              <div className="rounded-xl bg-white/5 border border-white/10">
                <PersonalitySignal data={personality} />
              </div>
              <div className="rounded-xl bg-white/5 border border-white/10">
                <CorrelationPanel correlations={correlations} />
              </div>
            </div>
          )}
        </div>
      </GlassCard>
    </div>
  );
}
