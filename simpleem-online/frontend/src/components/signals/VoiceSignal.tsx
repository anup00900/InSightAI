import { Mic } from 'lucide-react';
import SignalPanel from './SignalPanel';
import type { VoiceSignal as VoiceSignalType, AudioFeatureData } from '../../hooks/useRealtimeAnalysis';

interface Props {
  voiceSignals: VoiceSignalType[];
  audioFeatures?: AudioFeatureData[];
  hasAudio?: boolean | null;
}

const paceColor: Record<string, string> = {
  slow: '#3b82f6',
  moderate: '#10b981',
  fast: '#f59e0b',
};

function MiniBar({ value, max, color }: { value: number; max: number; color: string }) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  return (
    <div className="w-14 h-1.5 glass-depth-1 rounded-full overflow-hidden">
      <div className="h-full rounded-full transition-all duration-500" style={{ width: `${pct}%`, backgroundColor: color }} />
    </div>
  );
}

export default function VoiceSignal({ voiceSignals, audioFeatures = [], hasAudio }: Props) {
  const latest = voiceSignals[voiceSignals.length - 1];
  const latestAudio = audioFeatures[audioFeatures.length - 1];

  if (!latest && !latestAudio) {
    return (
      <SignalPanel icon={<Mic className="w-4 h-4" />} title="Voice" color="#06b6d4">
        <p className="text-xs text-text-muted">
          {hasAudio === false ? 'No audio track in video' : 'Waiting for audio...'}
        </p>
      </SignalPanel>
    );
  }

  return (
    <SignalPanel
      icon={<Mic className="w-4 h-4" />}
      title="Voice"
      color="#06b6d4"
      lastUpdate={latestAudio?.timestamp || latest?.timestamp}
    >
      <div className="space-y-1.5">
        {/* Real librosa audio features (priority) */}
        {latestAudio && (
          <>
            <div className="flex items-center justify-between">
              <span className="text-xs text-text-muted">Pitch</span>
              <div className="flex items-center gap-1.5">
                <MiniBar value={latestAudio.pitch_mean} max={400} color="#8b5cf6" />
                <span className="text-xs font-bold text-purple-400 tabular-nums w-10 text-right">{Math.round(latestAudio.pitch_mean)} Hz</span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-text-muted">Volume</span>
              <div className="flex items-center gap-1.5">
                <MiniBar value={latestAudio.volume_energy} max={100} color="#06b6d4" />
                <span className="text-xs font-bold text-cyan-400 tabular-nums w-10 text-right">{Math.round(latestAudio.volume_energy)}</span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-text-muted">Rate</span>
              <div className="flex items-center gap-1.5">
                <MiniBar value={latestAudio.speaking_rate} max={8} color="#10b981" />
                <span className="text-xs font-bold text-emerald-400 tabular-nums w-10 text-right">{latestAudio.speaking_rate.toFixed(1)}/s</span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-text-muted">Energy</span>
              <div className="flex items-center gap-1.5">
                <MiniBar value={latestAudio.energy} max={100} color="#f59e0b" />
                <span className="text-xs font-bold text-amber-400 tabular-nums w-10 text-right">{Math.round(latestAudio.energy)}</span>
              </div>
            </div>
            {latestAudio.pause_count > 0 && (
              <p className="text-[10px] text-text-muted">{latestAudio.pause_count} pauses detected</p>
            )}
          </>
        )}

        {/* GPT voice analysis (fallback or supplement) */}
        {latest && !latestAudio && (
          <>
            <div className="flex items-center justify-between">
              <span className="text-xs text-text-muted">Tone</span>
              <span className="text-xs font-bold text-cyan-400 capitalize">{latest.tone}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-text-muted">Pace</span>
              <div className="flex items-center gap-2">
                <div className="flex gap-0.5">
                  {['slow', 'moderate', 'fast'].map((p) => (
                    <div
                      key={p}
                      className="w-5 h-1.5 rounded-full"
                      style={{ backgroundColor: latest.pace === p ? paceColor[p] : '#1e293b' }}
                    />
                  ))}
                </div>
                <span className="text-xs text-text-secondary capitalize">{latest.pace}</span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-text-muted">Energy</span>
              <div className="flex items-center gap-2">
                <div className="w-16 h-1.5 glass-depth-1 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${Math.max(0, Math.min(100, latest.energy))}%`,
                      backgroundColor: latest.energy >= 70 ? '#10b981' : latest.energy >= 40 ? '#f59e0b' : '#ef4444',
                    }}
                  />
                </div>
                <span className="text-xs font-bold text-text-secondary tabular-nums">{Math.round(latest.energy)}</span>
              </div>
            </div>
            {latest.dynamics && <p className="text-[10px] text-text-muted italic">{latest.dynamics}</p>}
          </>
        )}
      </div>
    </SignalPanel>
  );
}
