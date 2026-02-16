import type { VoiceSignalData } from '../../lib/api';

interface Props { signals: VoiceSignalData[] }

export default function VoiceSignal({ signals }: Props) {
  if (!signals.length) return <div className="p-4 text-sm text-slate-500">No voice data</div>;

  return (
    <div className="p-4 space-y-3">
      <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Voice Analysis</h4>
      {/* Energy timeline mini chart */}
      <div className="h-16 flex items-end gap-0.5">
        {signals.map((s, i) => (
          <div
            key={i}
            className="flex-1 rounded-t-sm bg-gradient-to-t from-violet-500/40 to-violet-400/80 transition-all"
            style={{ height: `${Math.max(4, s.energy)}%` }}
            title={`${Math.round(s.timestamp)}s: ${s.tone} (${s.energy}%)`}
          />
        ))}
      </div>
      <div className="space-y-1.5">
        {signals.slice(-3).map((s, i) => (
          <div key={i} className="flex items-center gap-2 text-xs">
            <span className="text-slate-500 font-mono w-10">{Math.round(s.timestamp)}s</span>
            <span className="px-1.5 py-0.5 rounded bg-violet-500/10 text-violet-300 border border-violet-500/20">{s.tone}</span>
            <span className="text-slate-500">{s.pace}</span>
            <span className="ml-auto font-bold text-slate-400">{s.energy}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
