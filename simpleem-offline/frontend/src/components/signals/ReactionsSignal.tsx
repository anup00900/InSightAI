import type { SignalSnapshot } from '../../lib/api';

interface Props { snapshots: SignalSnapshot[] }

export default function ReactionsSignal({ snapshots }: Props) {
  if (!snapshots.length) return <div className="p-4 text-sm text-slate-500">No reaction data</div>;

  const latest = snapshots[snapshots.length - 1];

  return (
    <div className="p-4 space-y-3">
      <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Reactions</h4>
      <div className="space-y-2">
        {latest.participants.map((p, i) => (
          <div key={i} className="p-2 rounded-lg bg-white/5">
            <span className="text-sm text-slate-300 font-medium">{p.label}</span>
            <div className="flex flex-wrap gap-1.5 mt-1.5">
              {p.reactions.map((r, j) => (
                <span key={j} className="text-[10px] px-2 py-0.5 rounded-full bg-rose-500/10 text-rose-300 border border-rose-500/20">
                  {r}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
