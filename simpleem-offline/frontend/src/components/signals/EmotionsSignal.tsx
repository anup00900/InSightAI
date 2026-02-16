import type { SignalSnapshot } from '../../lib/api';

interface Props { snapshots: SignalSnapshot[] }

export default function EmotionsSignal({ snapshots }: Props) {
  if (!snapshots.length) return <div className="p-4 text-sm text-slate-500">No emotion data available</div>;

  // Show latest snapshot emotions for each participant
  const latest = snapshots[snapshots.length - 1];

  return (
    <div className="p-4 space-y-3">
      <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Emotions</h4>
      <div className="space-y-2">
        {latest.participants.map((p, i) => (
          <div key={i} className="flex items-center justify-between p-2 rounded-lg bg-white/5">
            <span className="text-sm text-slate-300 font-medium">{p.label}</span>
            <div className="flex items-center gap-2">
              <span className="text-xs px-2 py-0.5 rounded-full bg-purple-500/10 text-purple-300 border border-purple-500/20">
                {p.emotions.primary}
              </span>
              <span className="text-xs text-slate-500">{Math.round(p.emotions.engagement)}%</span>
            </div>
          </div>
        ))}
      </div>
      <p className="text-xs text-slate-600">{snapshots.length} snapshots analyzed</p>
    </div>
  );
}
