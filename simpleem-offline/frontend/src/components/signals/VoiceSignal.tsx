import { type VoiceSignalData, type SpeakerAudioData, formatTime } from '../../lib/api';
import { Mic, Activity, Gauge, Timer } from 'lucide-react';

interface Props {
  signals: VoiceSignalData[];
  speakerAudio: SpeakerAudioData | null;
}

export default function VoiceSignal({ signals, speakerAudio }: Props) {
  const hasRealAudio = speakerAudio && speakerAudio.participants.length > 0;
  const hasGptVoice = signals.length > 0;

  if (!hasRealAudio && !hasGptVoice) {
    return (
      <div className="p-4">
        <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2">
          <Mic className="w-3.5 h-3.5" /> Voice Analysis
        </h4>
        <p className="text-slate-600 text-xs">No voice data available</p>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4">
      <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-2">
        <Mic className="w-3.5 h-3.5" /> Voice Analysis
      </h4>

      {/* Per-speaker audio metrics (from librosa) */}
      {hasRealAudio && (
        <div className="space-y-3">
          <p className="text-[10px] text-slate-600 uppercase tracking-wider">Per-Speaker Audio (librosa)</p>
          {speakerAudio!.participants.map((p) => {
            if (p.features.length === 0) return null;
            const avgPitch = Math.round(p.features.reduce((s, f) => s + f.pitch_mean, 0) / p.features.length);
            const avgEnergy = Math.round(p.features.reduce((s, f) => s + f.volume_energy, 0) / p.features.length);
            const avgRate = (p.features.reduce((s, f) => s + f.speaking_rate, 0) / p.features.length).toFixed(1);
            const avgPause = Math.round(p.features.reduce((s, f) => s + f.pause_ratio, 0) / p.features.length * 100);
            const avgEngagement = Math.round(p.features.reduce((s, f) => s + f.engagement_score, 0) / p.features.length);

            return (
              <div key={p.name} className="p-3 rounded-lg bg-white/[0.03] border border-white/5">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold text-slate-300">{p.name}</span>
                  <span className={`text-xs font-bold ${avgEngagement >= 60 ? 'text-emerald-400' : avgEngagement >= 40 ? 'text-amber-400' : 'text-red-400'}`}>
                    {avgEngagement}% energy
                  </span>
                </div>
                <div className="grid grid-cols-4 gap-2">
                  <div className="text-center">
                    <Activity className="w-3 h-3 text-violet-400 mx-auto mb-0.5" />
                    <p className="text-[10px] text-slate-500">Pitch</p>
                    <p className="text-xs font-bold text-slate-300">{avgPitch} Hz</p>
                  </div>
                  <div className="text-center">
                    <Gauge className="w-3 h-3 text-blue-400 mx-auto mb-0.5" />
                    <p className="text-[10px] text-slate-500">Volume</p>
                    <p className="text-xs font-bold text-slate-300">{avgEnergy}%</p>
                  </div>
                  <div className="text-center">
                    <Mic className="w-3 h-3 text-cyan-400 mx-auto mb-0.5" />
                    <p className="text-[10px] text-slate-500">Rate</p>
                    <p className="text-xs font-bold text-slate-300">{avgRate}/s</p>
                  </div>
                  <div className="text-center">
                    <Timer className="w-3 h-3 text-amber-400 mx-auto mb-0.5" />
                    <p className="text-[10px] text-slate-500">Pauses</p>
                    <p className="text-xs font-bold text-slate-300">{avgPause}%</p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Energy timeline bar chart */}
      {hasGptVoice && (
        <div>
          <p className="text-[10px] text-slate-600 uppercase tracking-wider mb-2">Energy Timeline</p>
          <div className="flex items-end gap-px h-16">
            {signals.map((s, i) => (
              <div
                key={i}
                className="flex-1 bg-gradient-to-t from-violet-500/40 to-violet-400/80 rounded-t-sm transition-all hover:from-violet-400/60 hover:to-violet-300"
                style={{ height: `${Math.max(4, s.energy)}%` }}
                title={`${formatTime(s.timestamp)} — ${s.tone}, ${s.pace} pace, energy: ${s.energy}%`}
              />
            ))}
          </div>
        </div>
      )}

      {/* GPT qualitative voice analysis (last 3) */}
      {hasGptVoice && (
        <div className="space-y-1.5">
          <p className="text-[10px] text-slate-600 uppercase tracking-wider">Tone & Dynamics</p>
          {signals.slice(-3).map((s, i) => (
            <div key={i} className="flex items-center gap-2 text-xs">
              <span className="text-slate-500 font-mono w-10">{formatTime(s.timestamp)}</span>
              <span className="px-1.5 py-0.5 rounded bg-violet-500/10 text-violet-300 text-[10px]">{s.tone}</span>
              <span className="px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-300 text-[10px]">{s.pace}</span>
              <span className="text-slate-400 flex-1 truncate text-[10px]">{s.dynamics}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
