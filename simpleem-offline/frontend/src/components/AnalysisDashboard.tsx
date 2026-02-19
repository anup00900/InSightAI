import { useState, useEffect } from 'react';
import {
  type Video, type AnalysisResults, type Participant, type SignalSnapshot,
  type VoiceSignalData, type WordSignalData, type PersonalitySignalData,
  type CorrelationPattern, type SpeakerAudioData, type MeetingNotes,
  type ConfidenceSummary, formatTime, getSignalSnapshots, getVoiceSignals,
  getWordSignals, getPersonalitySignals, getCorrelations,
  getSpeakerAudio, getMeetingNotes, getConfidenceSummary,
} from '../lib/api';
import ParticipantSidebar from './ParticipantSidebar';
import SkillRadar from './SkillRadar';
import CoachingPanel from './CoachingPanel';
import GlassCard from './ui/GlassCard';
import ExpandableCard from './ui/ExpandableCard';
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
  Download, Activity, Heart, AlertTriangle, CheckCircle2, Info,
  Mic, Eye, Type, ChevronDown, Sparkles, Hand, Zap, Brain, GitCompareArrows,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// ─── Types ──────────────────────────────────────────────────────

interface Props {
  results: AnalysisResults | null;
  loading: boolean;
  videoId: string;
  video?: Video | null;
}

type Tab = 'overview' | 'coaching' | 'transcript' | 'signals';

const tabColors: Record<Tab, { active: string; border: string; bg: string; icon: string }> = {
  overview:   { active: 'text-indigo-300',  border: 'from-indigo-400 to-purple-400',  bg: 'bg-indigo-500/10',  icon: 'text-indigo-400' },
  coaching:   { active: 'text-amber-300',   border: 'from-amber-400 to-orange-400',   bg: 'bg-amber-500/10',   icon: 'text-amber-400' },
  transcript: { active: 'text-purple-300',  border: 'from-purple-400 to-pink-400',    bg: 'bg-purple-500/10',  icon: 'text-purple-400' },
  signals:    { active: 'text-emerald-300', border: 'from-emerald-400 to-teal-400',  bg: 'bg-emerald-500/10', icon: 'text-emerald-400' },
};

const CHART_COLORS = ['#818cf8', '#c084fc', '#22d3ee', '#f59e0b', '#10b981', '#f43f5e', '#fb7185', '#a78bfa', '#34d399', '#fbbf24', '#f87171'];
const DONUT_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1'];

export default function AnalysisDashboard({ results, loading, videoId }: Props) {
  const [activeTab, setActiveTab] = useState<Tab>('overview');
  const [selectedParticipantId, setSelectedParticipantId] = useState<string | null>(null);
  const [selectedEngagement, setSelectedEngagement] = useState<string | null>(null);
  const [exporting, setExporting] = useState<'pdf' | 'csv' | null>(null);

  const [snapshots, setSnapshots] = useState<SignalSnapshot[]>([]);
  const [voiceSignals, setVoiceSignals] = useState<VoiceSignalData[]>([]);
  const [wordSignals, setWordSignals] = useState<WordSignalData[]>([]);
  const [personality, setPersonality] = useState<PersonalitySignalData | null>(null);
  const [correlations, setCorrelations] = useState<CorrelationPattern[]>([]);
  const [speakerAudio, setSpeakerAudio] = useState<SpeakerAudioData | null>(null);
  const [meetingNotes, setMeetingNotes] = useState<MeetingNotes | null>(null);
  const [confidenceSummary, setConfidenceSummary] = useState<ConfidenceSummary | null>(null);

  useEffect(() => {
    if (!results) return;
    Promise.all([
      getSignalSnapshots(videoId),
      getVoiceSignals(videoId),
      getWordSignals(videoId),
      getPersonalitySignals(videoId),
      getCorrelations(videoId),
      getSpeakerAudio(videoId),
      getMeetingNotes(videoId),
      getConfidenceSummary(videoId),
    ]).then(([snap, voice, words, pers, corr, spkAudio, notes, conf]) => {
      setSnapshots(snap);
      setVoiceSignals(voice);
      setWordSignals(words);
      setPersonality(pers);
      setCorrelations(corr);
      setSpeakerAudio(spkAudio);
      setMeetingNotes(notes);
      setConfidenceSummary(conf);
    });
  }, [results, videoId]);

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

  // Channel averages
  const avgVisual = (() => {
    if (!snapshots.length) return 0;
    const scores = snapshots.flatMap(s => s.participants.map(p => p.emotions.engagement));
    return scores.length ? Math.round(scores.reduce((a, b) => a + b, 0) / scores.length) : 0;
  })();
  const avgAudio = (() => {
    if (!voiceSignals.length) return 0;
    return Math.round(voiceSignals.reduce((s, v) => s + v.energy, 0) / voiceSignals.length);
  })();
  const avgVerbal = (() => {
    if (!wordSignals.length) return 0;
    return Math.round(wordSignals.reduce((s, w) => s + Math.abs(w.sentiment_score) * 100, 0) / wordSignals.length);
  })();

  const tabs: { key: Tab; label: string; icon: React.ReactNode }[] = [
    { key: 'overview', label: 'Overview', icon: <BarChart3 className="w-4 h-4" /> },
    { key: 'coaching', label: 'Coaching', icon: <Lightbulb className="w-4 h-4" /> },
    { key: 'transcript', label: 'Transcript', icon: <MessageSquare className="w-4 h-4" /> },
    { key: 'signals', label: 'Signals', icon: <Activity className="w-4 h-4" /> },
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

  // Speaking distribution - normalize to always sum to 100
  const speakingData = participants.map(p => ({ name: p.name, pct: p.speaking_pct }));
  const totalSpk = speakingData.reduce((s, d) => s + d.pct, 0);
  const normalizedSpk = totalSpk > 0
    ? speakingData.map(d => ({ ...d, pct: Math.round((d.pct / totalSpk) * 100) }))
    : speakingData.map(d => ({ ...d, pct: Math.round(100 / Math.max(participants.length, 1)) }));

  return (
    <div className="space-y-6">
      {/* ─── Header + Export ──────────────────────────────────────── */}
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
          </div>
        </div>
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

      {/* ─── Expandable KPI Row ─────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
        <ExpandableCard
          glowColor="rgba(139,92,246,0.2)"
          expandedContent={
            <div className="space-y-3">
              <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">How this score is computed</p>
              <p className="text-xs text-slate-400 leading-relaxed">
                The Overall Score combines 3 behavioral channels using the Mehrabian model:
              </p>
              <div className="space-y-2">
                {[
                  { label: 'Visual (55%)', value: avgVisual, color: '#818cf8', desc: 'Facial expressions, body language, gestures, eye contact' },
                  { label: 'Audio (38%)', value: avgAudio, color: '#a78bfa', desc: 'Voice energy, pitch variation, speaking rate, pauses' },
                  { label: 'Text (7%)', value: avgVerbal, color: '#22d3ee', desc: 'Word sentiment, keyword positivity, conversational tone' },
                ].map(ch => (
                  <div key={ch.label}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-[10px] text-slate-400">{ch.label}</span>
                      <span className="text-[10px] font-bold text-slate-300">{ch.value}%</span>
                    </div>
                    <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                      <div className="h-full rounded-full transition-all" style={{ width: `${ch.value}%`, backgroundColor: ch.color }} />
                    </div>
                    <p className="text-[9px] text-slate-600 mt-0.5">{ch.desc}</p>
                  </div>
                ))}
              </div>
              {confidenceSummary && (
                <div className="mt-3 p-2 rounded-lg bg-white/[0.03] border border-white/5">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] text-slate-500">Model Confidence</span>
                    <span className={`text-xs font-bold ${confidenceSummary.avg_confidence >= 0.8 ? 'text-emerald-400' : confidenceSummary.avg_confidence >= 0.6 ? 'text-amber-400' : 'text-red-400'}`}>
                      {Math.round(confidenceSummary.avg_confidence * 100)}% ({confidenceSummary.accuracy_grade})
                    </span>
                  </div>
                  <p className="text-[9px] text-slate-600 mt-1">
                    {confidenceSummary.disagreement_count === 0
                      ? 'All 3 AI models agreed on every analysis point'
                      : `${confidenceSummary.disagreement_count} points where models disagreed (resolved by majority vote)`}
                  </p>
                </div>
              )}
            </div>
          }
        >
          <TrendingUp className="w-5 h-5 text-purple-400 mx-auto mb-1.5" />
          <p className="text-3xl font-black gradient-text text-center">{Math.round(vid.overall_score)}%</p>
          <p className="text-[10px] text-slate-500 mt-1 text-center">Overall Score</p>
        </ExpandableCard>

        <ExpandableCard
          glowColor="rgba(56,189,248,0.2)"
          expandedContent={
            <div className="space-y-3">
              <p className="text-[10px] text-slate-500 uppercase tracking-wider">Per-Participant Engagement</p>
              {participants.map((p, i) => {
                const eng = Math.round(p.engagement_score);
                const barColor = eng > 50 ? 'bg-emerald-400' : eng >= 30 ? 'bg-amber-400' : 'bg-red-400';
                const textColor = eng > 50 ? 'text-emerald-400' : eng >= 30 ? 'text-amber-400' : 'text-red-400';
                return (
                  <div key={p.id} className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: CHART_COLORS[i % CHART_COLORS.length] }} />
                    <span className="text-[10px] text-slate-400 flex-1 truncate">{p.name}</span>
                    <div className="w-16 h-1.5 bg-white/10 rounded-full overflow-hidden">
                      <div className={`h-full rounded-full ${barColor}`} style={{ width: `${p.engagement_score}%` }} />
                    </div>
                    <span className={`text-[10px] font-bold w-8 text-right ${textColor}`}>{eng}%</span>
                  </div>
                );
              })}
            </div>
          }
        >
          <Activity className="w-5 h-5 text-sky-400 mx-auto mb-1.5" />
          <p className="text-3xl font-black gradient-text text-center">{avgEngagement}%</p>
          <p className="text-[10px] text-slate-500 mt-1 text-center">Engagement</p>
        </ExpandableCard>

        <ExpandableCard
          glowColor={summary?.overall_sentiment?.toLowerCase() === 'positive' ? 'rgba(16,185,129,0.2)' : summary?.overall_sentiment?.toLowerCase() === 'negative' ? 'rgba(239,68,68,0.2)' : 'rgba(245,158,11,0.2)'}
          expandedContent={
            <div className="space-y-3">
              <p className="text-[10px] text-slate-500 uppercase tracking-wider">Sentiment Analysis</p>
              {wordSignals.length > 0 && (() => {
                const pos = wordSignals.filter(w => w.sentiment === 'positive').length;
                const neg = wordSignals.filter(w => w.sentiment === 'negative').length;
                const neu = wordSignals.filter(w => w.sentiment === 'neutral').length;
                const total = Math.max(pos + neg + neu, 1);
                return (
                  <div>
                    <div className="flex h-2 rounded-full overflow-hidden bg-white/10">
                      <div className="bg-emerald-500" style={{ width: `${(pos/total)*100}%` }} />
                      <div className="bg-amber-500" style={{ width: `${(neu/total)*100}%` }} />
                      <div className="bg-red-500" style={{ width: `${(neg/total)*100}%` }} />
                    </div>
                    <div className="flex justify-between mt-1">
                      <span className="text-[9px] text-emerald-400">Positive</span>
                      <span className="text-[9px] text-amber-400">Neutral</span>
                      <span className="text-[9px] text-red-400">Negative</span>
                    </div>
                  </div>
                );
              })()}
            </div>
          }
        >
          <Heart className={`w-5 h-5 ${sentimentColor.text} mx-auto mb-1.5`} />
          <p className={`text-2xl font-black bg-gradient-to-r ${sentimentColor.gradient} bg-clip-text text-transparent text-center`}>
            {sentimentLabel}
          </p>
          <p className="text-[10px] text-slate-500 mt-1 text-center">Sentiment</p>
        </ExpandableCard>

        <ExpandableCard
          glowColor="rgba(6,182,212,0.2)"
          expandedContent={
            <div className="space-y-2">
              <p className="text-[10px] text-slate-500 uppercase tracking-wider">Quick View</p>
              {participants.map((p, i) => (
                <div key={p.id} className="flex items-center gap-2 p-1.5 rounded-lg hover:bg-white/5">
                  <div className="w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold" style={{ backgroundColor: CHART_COLORS[i % CHART_COLORS.length] + '40', border: `1px solid ${CHART_COLORS[i % CHART_COLORS.length]}60` }}>
                    {p.name.charAt(0).toUpperCase()}
                  </div>
                  <span className="text-[10px] text-slate-300 flex-1 truncate">{p.name}</span>
                  <span className="text-[9px] text-slate-500">{p.speaking_pct}%</span>
                </div>
              ))}
            </div>
          }
        >
          <Users className="w-5 h-5 text-cyan-400 mx-auto mb-1.5" />
          <p className="text-3xl font-black gradient-text text-center">{participants.length}</p>
          <p className="text-[10px] text-slate-500 mt-1 text-center">Participants</p>
        </ExpandableCard>

        <ExpandableCard
          glowColor="rgba(239,68,68,0.2)"
          expandedContent={
            <div className="space-y-2">
              <p className="text-[10px] text-slate-500 uppercase tracking-wider">Flagged Moments</p>
              {flags.length === 0 ? (
                <p className="text-xs text-slate-500">No critical moments detected</p>
              ) : (
                <div className="space-y-1.5 max-h-40 overflow-y-auto">
                  {flags.slice(0, 5).map((flag, i) => (
                    <div key={i} className="flex items-start gap-2">
                      <span className={`w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0 ${
                        flag.severity === 'high' ? 'bg-red-400' : flag.severity === 'medium' ? 'bg-amber-400' : 'bg-emerald-400'
                      }`} />
                      <div>
                        <p className="text-[10px] text-slate-300 leading-relaxed">{flag.description.slice(0, 80)}{flag.description.length > 80 ? '...' : ''}</p>
                        <span className="text-[9px] text-slate-600 font-mono">{formatTime(flag.timestamp)}</span>
                      </div>
                    </div>
                  ))}
                  {flags.length > 5 && <p className="text-[9px] text-slate-600">+{flags.length - 5} more</p>}
                </div>
              )}
            </div>
          }
        >
          <AlertTriangle className="w-5 h-5 text-red-400 mx-auto mb-1.5" />
          <p className="text-3xl font-black gradient-text text-center">{flags.length}</p>
          <p className="text-[10px] text-slate-500 mt-1 text-center">Key Moments</p>
        </ExpandableCard>
      </div>

      {/* ─── Expandable Channel Scores ──────────────────────────── */}
      <div className="grid grid-cols-3 gap-3">
        <ExpandableCard
          glowColor="rgba(99,102,241,0.2)"
          expandedContent={
            <div className="space-y-2">
              <p className="text-[10px] text-slate-500 uppercase tracking-wider">Visual Signal Breakdown</p>
              {snapshots.length > 0 && (() => {
                const emotionCounts: Record<string, number> = {};
                snapshots.forEach(s => s.participants.forEach(p => {
                  const e = p.emotions.primary;
                  emotionCounts[e] = (emotionCounts[e] || 0) + 1;
                }));
                const sorted = Object.entries(emotionCounts).sort((a, b) => b[1] - a[1]).slice(0, 4);
                const total = Object.values(emotionCounts).reduce((a, b) => a + b, 0);
                return (
                  <div className="space-y-1">
                    {sorted.map(([emotion, count]) => (
                      <div key={emotion} className="flex items-center gap-2">
                        <span className="text-[9px] text-slate-400 w-16 capitalize">{emotion}</span>
                        <div className="flex-1 h-1 bg-white/10 rounded-full overflow-hidden">
                          <div className="h-full rounded-full bg-indigo-400" style={{ width: `${(count/total)*100}%` }} />
                        </div>
                        <span className="text-[9px] text-slate-500">{Math.round((count/total)*100)}%</span>
                      </div>
                    ))}
                  </div>
                );
              })()}
              <p className="text-[9px] text-slate-600">Analyzed from {snapshots.length} video frames using triple-model consensus voting</p>
            </div>
          }
        >
          <div className="flex items-center gap-3">
            <Eye className="w-4 h-4 text-indigo-400 flex-shrink-0" />
            <div className="flex-1">
              <div className="text-[10px] text-slate-500 uppercase tracking-wider">Visual (55%)</div>
              <div className="flex items-center gap-2 mt-1">
                <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
                  <div className="h-full rounded-full bg-indigo-500 transition-all" style={{ width: `${avgVisual}%` }} />
                </div>
                <span className="text-sm font-bold text-indigo-400">{avgVisual || '--'}%</span>
              </div>
            </div>
          </div>
        </ExpandableCard>

        <ExpandableCard
          glowColor="rgba(139,92,246,0.2)"
          expandedContent={
            <div className="space-y-2">
              <p className="text-[10px] text-slate-500 uppercase tracking-wider">Audio Signal Breakdown</p>
              {speakerAudio && speakerAudio.participants.length > 0 ? (
                speakerAudio.participants.map(p => {
                  if (p.features.length === 0) return null;
                  const avgPitch = Math.round(p.features.reduce((s, f) => s + f.pitch_mean, 0) / p.features.length);
                  const avgVol = Math.round(p.features.reduce((s, f) => s + f.volume_energy, 0) / p.features.length);
                  const avgRate = (p.features.reduce((s, f) => s + f.speaking_rate, 0) / p.features.length).toFixed(1);
                  return (
                    <div key={p.name} className="flex items-center gap-3 text-[10px]">
                      <span className="text-slate-300 w-20 truncate">{p.name}</span>
                      <span className="text-slate-500">Pitch: <span className="text-violet-300">{avgPitch}Hz</span></span>
                      <span className="text-slate-500">Vol: <span className="text-violet-300">{avgVol}%</span></span>
                      <span className="text-slate-500">Rate: <span className="text-violet-300">{avgRate}/s</span></span>
                    </div>
                  );
                })
              ) : (
                <p className="text-[9px] text-slate-600">Per-speaker audio from librosa analysis</p>
              )}
              <p className="text-[9px] text-slate-600">Dual-model validation (GPT-4o-mini + GPT-4.1)</p>
            </div>
          }
        >
          <div className="flex items-center gap-3">
            <Mic className="w-4 h-4 text-violet-400 flex-shrink-0" />
            <div className="flex-1">
              <div className="text-[10px] text-slate-500 uppercase tracking-wider">Audio (38%)</div>
              <div className="flex items-center gap-2 mt-1">
                <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
                  <div className="h-full rounded-full bg-violet-500 transition-all" style={{ width: `${avgAudio}%` }} />
                </div>
                <span className="text-sm font-bold text-violet-400">{avgAudio || '--'}%</span>
              </div>
            </div>
          </div>
        </ExpandableCard>

        <ExpandableCard
          glowColor="rgba(6,182,212,0.2)"
          expandedContent={
            <div className="space-y-2">
              <p className="text-[10px] text-slate-500 uppercase tracking-wider">Text Signal Breakdown</p>
              {wordSignals.length > 0 && (() => {
                const allKw = wordSignals.flatMap(s => s.keywords);
                const kwCounts: Record<string, number> = {};
                allKw.forEach(k => { kwCounts[k] = (kwCounts[k] || 0) + 1; });
                const topKw = Object.entries(kwCounts).sort((a, b) => b[1] - a[1]).slice(0, 6);
                return (
                  <div className="flex flex-wrap gap-1">
                    {topKw.map(([kw, count]) => (
                      <span key={kw} className="px-1.5 py-0.5 text-[9px] rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">
                        {kw} <span className="text-cyan-500/50">x{count}</span>
                      </span>
                    ))}
                  </div>
                );
              })()}
              <p className="text-[9px] text-slate-600">Dual-model voting (GPT-4o-mini + GPT-4.1)</p>
            </div>
          }
        >
          <div className="flex items-center gap-3">
            <Type className="w-4 h-4 text-cyan-400 flex-shrink-0" />
            <div className="flex-1">
              <div className="text-[10px] text-slate-500 uppercase tracking-wider">Text (7%)</div>
              <div className="flex items-center gap-2 mt-1">
                <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
                  <div className="h-full rounded-full bg-cyan-500 transition-all" style={{ width: `${avgVerbal}%` }} />
                </div>
                <span className="text-sm font-bold text-cyan-400">{avgVerbal || '--'}%</span>
              </div>
            </div>
          </div>
        </ExpandableCard>
      </div>

      {/* ─── Main Grid: Video + Sidebar ───────────────────────────── */}
      <div className="grid grid-cols-12 gap-6">
        <div className="col-span-12 lg:col-span-8 space-y-6">
          <GlassCard padding={false} className="overflow-hidden">
            <video controls className="w-full aspect-video bg-black" src={`/uploads/${vid.id}_${vid.filename}`} />
          </GlassCard>

          {/* Engagement Timeline — Single interactive chart */}
          {snapshots.length > 1 && (() => {
            const svgW = snapshots.length * 12;
            return (
              <GlassCard>
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Engagement Timeline</h4>
                  {selectedEngagement && (
                    <button onClick={() => setSelectedEngagement(null)} className="text-[10px] text-slate-500 hover:text-slate-300 transition-colors">
                      Show all
                    </button>
                  )}
                </div>
                <div className="relative h-44 mb-2">
                  <div className="absolute left-0 top-0 bottom-0 w-8 flex flex-col justify-between text-[9px] text-slate-600 py-1">
                    <span>100</span><span>75</span><span>50</span><span>25</span><span>0</span>
                  </div>
                  <div className="ml-8 h-full">
                    <svg viewBox={`0 0 ${svgW} 100`} className="w-full h-full" preserveAspectRatio="none">
                      {[25, 50, 75].map(y => (
                        <line key={y} x1="0" y1={100 - y} x2={svgW} y2={100 - y} stroke="rgba(255,255,255,0.06)" strokeWidth="0.5" />
                      ))}
                      {participants.map((part, pi) => {
                        const color = CHART_COLORS[pi % CHART_COLORS.length];
                        const isSelected = selectedEngagement === part.id;
                        const isAnySelected = selectedEngagement !== null;
                        const opacity = isAnySelected ? (isSelected ? 1 : 0.08) : 0.7;
                        const width = isSelected ? 2.5 : 1.2;
                        const points = snapshots.map((s, i) => {
                          const pData = s.participants.find(p => p.label === part.name);
                          const eng = pData ? pData.emotions.engagement : 50;
                          return `${i * 12},${100 - eng}`;
                        }).join(' ');
                        return <polyline key={pi} points={points} fill="none" stroke={color} strokeWidth={width} opacity={opacity} style={{ transition: 'opacity 0.3s, stroke-width 0.3s' }} />;
                      })}
                    </svg>
                  </div>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {participants.map((p, i) => {
                    const color = CHART_COLORS[i % CHART_COLORS.length];
                    const isSelected = selectedEngagement === p.id;
                    return (
                      <button
                        key={p.id}
                        onClick={() => setSelectedEngagement(isSelected ? null : p.id)}
                        className={`flex items-center gap-1.5 px-2 py-1 rounded-lg text-[10px] transition-all border ${
                          isSelected
                            ? 'bg-white/10 border-white/20 text-white font-semibold'
                            : 'bg-white/[0.02] border-transparent hover:bg-white/5 text-slate-400 hover:text-slate-300'
                        }`}
                      >
                        <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
                        <span className="truncate max-w-[100px]">{p.name}</span>
                        <span className="font-mono text-[9px] opacity-60">{Math.round(p.engagement_score)}%</span>
                      </button>
                    );
                  })}
                </div>
              </GlassCard>
            );
          })()}
        </div>

        {/* Right: Participants + Speaking + Radar */}
        <div className="col-span-12 lg:col-span-4 space-y-6">
          <ParticipantSidebar participants={participants} selectedId={selectedParticipantId} onSelect={setSelectedParticipantId} />
          <SkillRadar participant={selectedParticipant} />

          {/* Speaking Distribution — Donut */}
          <GlassCard>
            <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Speaking Distribution</h4>
            {(() => {
              const donutData = normalizedSpk;
              const totalPct = donutData.reduce((s, d) => s + d.pct, 0);
              return (
                <>
                  <div className="relative w-48 h-48 mx-auto mb-3">
                    <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                      {(() => {
                        let cumPct = 0;
                        return donutData.filter(d => d.pct > 0).map((d, i) => {
                          const pct = totalPct > 0 ? Math.round((d.pct / totalPct) * 100) : 0;
                          const dashArray = `${pct * 2.827} ${282.7 - pct * 2.827}`;
                          const offset = -cumPct * 2.827;
                          cumPct += pct;
                          return (
                            <circle
                              key={i}
                              cx="50" cy="50" r="45"
                              fill="none"
                              stroke={DONUT_COLORS[i % DONUT_COLORS.length]}
                              strokeWidth="8"
                              strokeDasharray={dashArray}
                              strokeDashoffset={offset}
                              className="transition-all duration-500"
                            />
                          );
                        });
                      })()}
                    </svg>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="text-center">
                        <p className="text-lg font-bold text-white">{participants.length}</p>
                        <p className="text-[9px] text-slate-500">participants</p>
                      </div>
                    </div>
                  </div>
                  <div className="space-y-1.5">
                    {donutData.map((d, i) => (
                      <div key={i} className="flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: DONUT_COLORS[i % DONUT_COLORS.length] }} />
                        <span className="text-xs text-slate-400 flex-1 truncate">{d.name}</span>
                        <span className="text-xs font-bold text-slate-300 tabular-nums">{d.pct}%</span>
                      </div>
                    ))}
                  </div>
                </>
              );
            })()}
          </GlassCard>
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
                <span className={isActive ? colors.icon : ''}>{tab.icon}</span>
                {tab.label}
                {isActive && (
                  <span className={`absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r ${colors.border}`} />
                )}
              </button>
            );
          })}
        </div>
        <div className="p-6">
          {/* ─── Overview Tab ──────────────────────────────────────── */}
          {activeTab === 'overview' && (
            <div className="space-y-8">
              {/* Summary Section */}
              {summary && (
                <section className="pb-6 border-b border-white/5">
                  <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                    <FileText className="w-3.5 h-3.5" /> Meeting Summary
                  </h4>
                  <div className="bg-gradient-to-br from-indigo-500/5 to-purple-500/5 rounded-xl p-5 border border-indigo-500/10">
                    <p className="text-slate-200 text-sm leading-relaxed whitespace-pre-line">{summary.summary}</p>
                    {summary.key_topics.length > 0 && (
                      <div className="flex flex-wrap gap-2 mt-4 pt-4 border-t border-white/5">
                        <span className="text-[10px] text-slate-500 uppercase tracking-wider self-center mr-1">Topics:</span>
                        {summary.key_topics.map((t, i) => (
                          <span key={i} className="px-3 py-1 text-xs font-medium text-purple-300 bg-purple-500/10 rounded-full border border-purple-500/20">{t}</span>
                        ))}
                      </div>
                    )}
                  </div>
                </section>
              )}

              {/* Meeting Notes Section */}
              {meetingNotes && (meetingNotes.action_items.length > 0 || meetingNotes.decisions.length > 0 || meetingNotes.follow_ups.length > 0) && (
                <section className="pb-6 border-b border-white/5">
                  <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                    <CheckCircle2 className="w-3.5 h-3.5" /> Meeting Notes
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {meetingNotes.action_items.length > 0 && (
                      <div className="p-4 rounded-xl bg-blue-500/5 border border-blue-500/10">
                        <h5 className="text-xs font-semibold text-blue-400 mb-3">Action Items</h5>
                        <div className="space-y-2">
                          {meetingNotes.action_items.map((item: any, i: number) => (
                            <div key={i} className="flex items-start gap-2">
                              <span className="text-blue-400 mt-0.5 text-xs">&#9679;</span>
                              <div>
                                <p className="text-sm text-slate-200">{item.task}</p>
                                <p className="text-[10px] text-slate-500 mt-0.5">
                                  {item.owner && <span className="text-blue-300">{item.owner}</span>}
                                  {item.deadline && <span> &middot; {item.deadline}</span>}
                                </p>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    {meetingNotes.decisions.length > 0 && (
                      <div className="p-4 rounded-xl bg-emerald-500/5 border border-emerald-500/10">
                        <h5 className="text-xs font-semibold text-emerald-400 mb-3">Decisions Made</h5>
                        <div className="space-y-2">
                          {meetingNotes.decisions.map((d: any, i: number) => (
                            <div key={i} className="flex items-start gap-2">
                              <CheckCircle2 className="w-3 h-3 text-emerald-400 mt-0.5 flex-shrink-0" />
                              <div>
                                <p className="text-sm text-slate-200">{d.decision}</p>
                                {d.context && <p className="text-[10px] text-slate-500 mt-0.5">{d.context}</p>}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    {meetingNotes.follow_ups.length > 0 && (
                      <div className="p-4 rounded-xl bg-amber-500/5 border border-amber-500/10">
                        <h5 className="text-xs font-semibold text-amber-400 mb-3">Follow-ups</h5>
                        <div className="space-y-2">
                          {meetingNotes.follow_ups.map((f: any, i: number) => (
                            <div key={i} className="flex items-start gap-2">
                              <span className="text-amber-400 mt-0.5 text-xs">&#9679;</span>
                              <div>
                                <p className="text-sm text-slate-200">{f.topic}</p>
                                <p className="text-[10px] text-slate-500 mt-0.5">
                                  {f.owner && <span className="text-amber-300">{f.owner}</span>}
                                  {f.timeline && <span> &middot; {f.timeline}</span>}
                                </p>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    {meetingNotes.key_questions.length > 0 && (
                      <div className="p-4 rounded-xl bg-purple-500/5 border border-purple-500/10">
                        <h5 className="text-xs font-semibold text-purple-400 mb-3">Open Questions</h5>
                        <div className="space-y-2">
                          {meetingNotes.key_questions.map((q: any, i: number) => (
                            <div key={i} className="flex items-start gap-2">
                              <span className="text-purple-400 mt-0.5 text-xs">?</span>
                              <div>
                                <p className="text-sm text-slate-200">{q.question}</p>
                                {q.raised_by && <p className="text-[10px] text-slate-500 mt-0.5">Raised by: <span className="text-purple-300">{q.raised_by}</span></p>}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </section>
              )}

              {/* Key Moments / Flags Section */}
              <section className="pb-6 border-b border-white/5">
                <div className="flex items-center justify-between mb-4">
                  <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-2">
                    <AlertTriangle className="w-3.5 h-3.5" /> Key Moments &amp; Flags
                  </h4>
                  {flags.length > 0 && (
                    <span className="text-[10px] font-bold text-slate-500 bg-white/5 px-2 py-0.5 rounded-full">{flags.length} flagged</span>
                  )}
                </div>
                {flags.length === 0 ? (
                  <div className="text-center py-8 rounded-xl bg-white/[0.02] border border-white/5">
                    <CheckCircle2 className="w-8 h-8 text-emerald-400/50 mx-auto mb-2" />
                    <p className="text-slate-500 text-sm">No critical moments flagged in this meeting</p>
                  </div>
                ) : (
                  <div className="space-y-2 max-h-[320px] overflow-y-auto pr-1 custom-scrollbar">
                    {flags.map((flag, i) => (
                      <div
                        key={i}
                        className={`flex items-start gap-3 p-3.5 rounded-xl border transition-colors ${
                          flag.type === 'positive'
                            ? 'border-emerald-500/20 bg-emerald-500/5'
                            : flag.severity === 'high'
                              ? 'border-red-500/20 bg-red-500/5'
                              : 'border-amber-500/20 bg-amber-500/5'
                        }`}
                      >
                        {flag.type === 'positive' ? (
                          <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
                        ) : flag.severity === 'high' ? (
                          <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                        ) : (
                          <Info className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                        )}
                        <div className="flex-1 min-w-0">
                          <p className="text-sm text-slate-200 leading-relaxed">{flag.description}</p>
                          <div className="flex items-center gap-2 mt-1.5">
                            <span className="text-[10px] text-slate-500 font-mono bg-white/5 px-1.5 py-0.5 rounded">{formatTime(flag.timestamp)}</span>
                            <span className={`text-[10px] font-bold uppercase px-1.5 py-0.5 rounded-full ${
                              flag.type === 'positive' ? 'text-emerald-400 bg-emerald-500/10'
                              : flag.type === 'engagement_alert' ? 'text-red-400 bg-red-500/10'
                              : 'text-amber-400 bg-amber-500/10'
                            }`}>{flag.type === 'engagement_alert' ? 'alert' : flag.type}</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </section>

              {/* Participant Breakdown Section */}
              <section>
                <div className="flex items-center justify-between mb-4">
                  <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-2">
                    <Users className="w-3.5 h-3.5" /> Participant Breakdown
                  </h4>
                  <span className="text-[10px] text-slate-600">{participants.length} participants</span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {participants.map((p, pi) => {
                    const talkPct = normalizedSpk.find(s => s.name === p.name)?.pct ?? 0;
                    const engScore = Math.round(p.engagement_score);
                    const engColor = engScore > 50 ? '#10b981' : engScore >= 30 ? '#f59e0b' : '#ef4444';
                    const metrics = [
                      { label: 'Engagement', value: p.engagement_score, color: engColor },
                      { label: 'Clarity', value: p.clarity_score, color: '#22d3ee' },
                      { label: 'Rapport', value: p.rapport_score, color: '#818cf8' },
                      { label: 'Energy', value: p.energy_score, color: '#f59e0b' },
                      { label: 'Speaking', value: talkPct, color: '#f472b6' },
                    ];
                    return (
                      <div key={p.id} className="p-5 rounded-xl bg-white/[0.03] border border-white/10 hover:border-white/15 transition-colors">
                        {/* Header with name + engagement score */}
                        <div className="flex items-center gap-3 mb-4 pb-3 border-b border-white/5">
                          <div className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold text-white"
                            style={{ backgroundColor: CHART_COLORS[pi % CHART_COLORS.length] + '40',
                                     border: `1px solid ${CHART_COLORS[pi % CHART_COLORS.length]}60` }}>
                            {p.name.charAt(0).toUpperCase()}
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-semibold text-slate-200 truncate">{p.name}</p>
                            <div className="flex items-center gap-2 mt-0.5">
                              <span className="text-[10px] text-slate-500">Speaking: <span className="font-bold text-slate-400">{talkPct}%</span></span>
                            </div>
                          </div>
                          <div className="text-right">
                            <span className={`text-lg font-black ${engScore > 50 ? 'text-emerald-400' : engScore >= 30 ? 'text-amber-400' : 'text-red-400'}`}>
                              {engScore}%
                            </span>
                          </div>
                        </div>
                        {/* Metric bars */}
                        <div className="space-y-2.5">
                          {metrics.map((m) => (
                            <div key={m.label} className="flex items-center gap-2.5">
                              <span className="text-[10px] text-slate-500 w-20">{m.label}</span>
                              <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
                                <div
                                  className="h-full rounded-full transition-all duration-700"
                                  style={{ width: `${m.value}%`, backgroundColor: m.color }}
                                />
                              </div>
                              <span className="text-[10px] font-bold text-slate-300 tabular-nums w-8 text-right">{Math.round(m.value)}%</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </section>
            </div>
          )}

          {/* ─── Coaching Tab ──────────────────────────────────────── */}
          {activeTab === 'coaching' && <CoachingPanel videoId={videoId} participants={participants} />}

          {/* ─── Transcript Tab ────────────────────────────────────── */}
          {activeTab === 'transcript' && (
            <div className="space-y-2 max-h-[500px] overflow-y-auto pr-2">
              {transcript.length === 0 ? (
                <p className="text-slate-500 text-sm text-center py-8">No transcript available</p>
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

          {/* ─── Signals Tab ──────────────────────────────────────── */}
          {activeTab === 'signals' && (
            <SignalAccordion
              snapshots={snapshots}
              voiceSignals={voiceSignals}
              speakerAudio={speakerAudio}
              wordSignals={wordSignals}
              personality={personality}
              correlations={correlations}
            />
          )}
        </div>
      </GlassCard>
    </div>
  );
}

// ─── Signal Accordion ──────────────────────────────────────────────
// Collapsible panels for each signal type — collapsed by default

interface SignalAccordionProps {
  snapshots: SignalSnapshot[];
  voiceSignals: VoiceSignalData[];
  speakerAudio: SpeakerAudioData | null;
  wordSignals: WordSignalData[];
  personality: PersonalitySignalData | null;
  correlations: CorrelationPattern[];
}

function SignalAccordion({ snapshots, voiceSignals, speakerAudio, wordSignals, personality, correlations }: SignalAccordionProps) {
  const [openPanels, setOpenPanels] = useState<Set<string>>(new Set());

  const toggle = (key: string) => {
    setOpenPanels(prev => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const panels: { key: string; title: string; icon: React.ReactNode; color: string; content: React.ReactNode }[] = [
    { key: 'emotions', title: 'Emotions', icon: <Sparkles className="w-4 h-4" />, color: 'text-amber-400', content: <EmotionsSignal snapshots={snapshots} /> },
    { key: 'body', title: 'Body Language', icon: <Eye className="w-4 h-4" />, color: 'text-blue-400', content: <BodyLanguageSignal snapshots={snapshots} /> },
    { key: 'gestures', title: 'Gestures', icon: <Hand className="w-4 h-4" />, color: 'text-cyan-400', content: <GesturesSignal snapshots={snapshots} /> },
    { key: 'reactions', title: 'Reactions', icon: <Zap className="w-4 h-4" />, color: 'text-yellow-400', content: <ReactionsSignal snapshots={snapshots} /> },
    { key: 'voice', title: 'Voice Analysis', icon: <Mic className="w-4 h-4" />, color: 'text-purple-400', content: <VoiceSignal signals={voiceSignals} speakerAudio={speakerAudio} /> },
    { key: 'words', title: 'Word Analysis', icon: <Type className="w-4 h-4" />, color: 'text-emerald-400', content: <WordsSignal signals={wordSignals} /> },
    { key: 'personality', title: 'Personality Profiles', icon: <Brain className="w-4 h-4" />, color: 'text-pink-400', content: <PersonalitySignal data={personality} /> },
    { key: 'correlations', title: 'Signal Correlations', icon: <GitCompareArrows className="w-4 h-4" />, color: 'text-indigo-400', content: <CorrelationPanel correlations={correlations} /> },
  ];

  return (
    <div className="space-y-2">
      {panels.map(({ key, title, icon, color, content }) => {
        const isOpen = openPanels.has(key);
        return (
          <div key={key} className="rounded-xl bg-white/[0.03] border border-white/10 overflow-hidden transition-colors hover:border-white/15">
            <button
              onClick={() => toggle(key)}
              className="w-full flex items-center gap-3 px-5 py-3.5 text-left"
            >
              <span className={color}>{icon}</span>
              <span className="text-sm font-semibold text-slate-200 flex-1">{title}</span>
              <motion.div animate={{ rotate: isOpen ? 180 : 0 }} transition={{ duration: 0.2 }}>
                <ChevronDown className="w-4 h-4 text-slate-500" />
              </motion.div>
            </button>
            <AnimatePresence initial={false}>
              {isOpen && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.25, ease: 'easeInOut' }}
                  className="overflow-hidden"
                >
                  <div className="border-t border-white/5">
                    {content}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        );
      })}
    </div>
  );
}
