import type { SignalSnapshot } from '../../lib/api';

interface Props { snapshots: SignalSnapshot[] }

export default function BodyLanguageSignal({ snapshots }: Props) {
  if (!snapshots.length) return <div className="p-4 text-sm text-slate-500">No body language data</div>;

  const latest = snapshots[snapshots.length - 1];

  return (
    <div className="p-4 space-y-3">
      <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Body Language</h4>
      <div className="space-y-2">
        {latest.participants.map((p, i) => (
          <div key={i} className="p-2 rounded-lg bg-white/5 space-y-1">
            <span className="text-sm text-slate-300 font-medium">{p.label}</span>
            <div className="flex flex-wrap gap-1.5 mt-1">
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-300 border border-blue-500/20">
                {p.body_language.posture}
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">
                {p.body_language.openness}
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
                {p.body_language.leaning}
              </span>
              {p.body_language.mirroring && (
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">
                  mirroring
                </span>
              )}
            </div>
            {p.body_language.description && (
              <p className="text-xs text-slate-500 mt-1">{p.body_language.description}</p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
