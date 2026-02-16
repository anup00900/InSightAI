import { useState, useEffect, useCallback } from 'react';
import { Zap } from 'lucide-react';
import GlassCard from '../ui/GlassCard';
import ProgressRing from '../ui/ProgressRing';
import GlowBadge from '../ui/GlowBadge';
import { usePolling } from '../../hooks/usePolling';
import { getVideoStatus, triggerAnalysis, getPartialResults, formatTime, type VideoStatus, type PartialResults } from '../../lib/api';

interface AnalysisStepProps {
  videoId: string;
  onComplete: () => void;
}

const STAGES = [
  { min: 0, label: 'Detecting participants' },
  { min: 10, label: 'Analyzing video frames' },
  { min: 45, label: 'Processing audio features' },
  { min: 55, label: 'Analyzing text signals' },
  { min: 65, label: 'Building personality profiles' },
  { min: 75, label: 'Computing engagement scores' },
  { min: 80, label: 'Generating insights' },
  { min: 93, label: 'Saving results' },
];

export default function AnalysisStep({ videoId, onComplete }: AnalysisStepProps) {
  const [started, setStarted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const statusFetcher = useCallback(() => getVideoStatus(videoId), [videoId]);
  const { data: status } = usePolling<VideoStatus>(statusFetcher, 2000, started);

  const partialFetcher = useCallback(() => getPartialResults(videoId), [videoId]);
  const { data: partial } = usePolling<PartialResults>(partialFetcher, 3000, started);

  useEffect(() => {
    const start = async () => {
      try {
        await triggerAnalysis(videoId);
        setStarted(true);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to start analysis');
      }
    };
    start();
  }, [videoId]);

  useEffect(() => {
    if (status?.step === 'complete') {
      setTimeout(onComplete, 1000);
    }
    if (status?.step === 'error') {
      setError(status.detail || 'Analysis failed');
    }
  }, [status, onComplete]);

  const progress = status?.progress ?? 0;
  const currentStageIdx = STAGES.findLastIndex(s => progress >= s.min);

  return (
    <div className="max-w-2xl mx-auto">
      <GlassCard>
        <div className="text-center mb-8">
          <h2 className="text-2xl font-bold gradient-text">Analyzing Meeting</h2>
          <p className="text-text-muted text-sm mt-2">{status?.detail || 'Starting analysis...'}</p>
        </div>

        {error ? (
          <div className="p-4 rounded-xl bg-danger/10 border border-danger/20 text-red-400 text-center">
            {error}
          </div>
        ) : (
          <>
            <div className="flex justify-center mb-8">
              <ProgressRing progress={progress} size={180} strokeWidth={10} />
            </div>

            <div className="grid grid-cols-2 gap-2">
              {STAGES.map((stage, idx) => (
                <div key={idx} className="flex items-center gap-2 py-1.5">
                  <div className={`w-2 h-2 rounded-full flex-shrink-0 transition-all duration-300 ${
                    idx < currentStageIdx ? 'bg-success' :
                    idx === currentStageIdx ? 'bg-accent animate-pulse' : 'bg-white/10'
                  }`} />
                  <span className={`text-xs ${
                    idx <= currentStageIdx ? 'text-text-primary' : 'text-text-muted'
                  }`}>
                    {stage.label}
                  </span>
                  {idx < currentStageIdx && (
                    <GlowBadge variant="success" size="sm">Done</GlowBadge>
                  )}
                </div>
              ))}
            </div>

            {/* Live Results Preview */}
            {partial && partial.ready_sections.length > 0 && (
              <div className="mt-8 space-y-3 animate-fade-in">
                <h3 className="text-sm font-semibold text-slate-400 flex items-center gap-2">
                  <Zap className="w-4 h-4 text-amber-400" />
                  Live Results
                </h3>

                {partial.participants && partial.participants.length > 0 && (
                  <div className="p-3 rounded-xl bg-white/5 border border-white/10 animate-slide-up">
                    <p className="text-xs font-medium text-emerald-400 mb-2">
                      ✓ {partial.participants.length} Participants Detected
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {partial.participants.map(p => (
                        <span key={p.id} className="px-2 py-1 text-xs text-slate-300 bg-white/5 rounded-lg">
                          {p.name}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {partial.transcript && partial.transcript.length > 0 && (
                  <div className="p-3 rounded-xl bg-white/5 border border-white/10 animate-slide-up">
                    <p className="text-xs font-medium text-blue-400 mb-2">
                      ✓ Transcript ({partial.transcript.length} segments)
                    </p>
                    <div className="space-y-1 max-h-32 overflow-y-auto">
                      {partial.transcript.slice(0, 5).map((seg, i) => (
                        <div key={i} className="flex gap-2 text-xs">
                          <span className="text-slate-500 font-mono shrink-0">{formatTime(seg.start_time)}</span>
                          {seg.speaker && <span className="text-purple-400 shrink-0">{seg.speaker}</span>}
                          <span className="text-slate-400 truncate">{seg.text}</span>
                        </div>
                      ))}
                      {partial.transcript.length > 5 && (
                        <p className="text-xs text-slate-500">...and {partial.transcript.length - 5} more</p>
                      )}
                    </div>
                  </div>
                )}

                {partial.summary && (
                  <div className="p-3 rounded-xl bg-white/5 border border-white/10 animate-slide-up">
                    <p className="text-xs font-medium text-purple-400">✓ Summary Ready</p>
                  </div>
                )}

                {partial.emotions && partial.emotions.length > 0 && (
                  <div className="p-3 rounded-xl bg-white/5 border border-white/10 animate-slide-up">
                    <p className="text-xs font-medium text-cyan-400">✓ Emotion Analysis ({partial.emotions.length} data points)</p>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </GlassCard>
    </div>
  );
}
