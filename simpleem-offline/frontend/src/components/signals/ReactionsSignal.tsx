import { type SignalSnapshot } from '../../lib/api';
import { Zap } from 'lucide-react';

interface Props { snapshots: SignalSnapshot[] }

const USELESS = new Set(['not visible', 'none', 'no reaction', 'n/a', 'unknown', 'neutral']);

export default function ReactionsSignal({ snapshots }: Props) {
  if (!snapshots.length) return <div className="p-4"><h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2"><Zap className="w-3.5 h-3.5" /> Reactions</h4><p className="text-slate-600 text-xs">No reaction data</p></div>;

  // Aggregate reaction frequency across ALL snapshots, per participant
  const participantReactions: Record<string, Record<string, number>> = {};
  snapshots.forEach(s => s.participants.forEach(p => {
    if (!participantReactions[p.label]) participantReactions[p.label] = {};
    (p.reactions || []).forEach(r => {
      const rl = r.toLowerCase().trim();
      if (!USELESS.has(rl)) {
        participantReactions[p.label][rl] = (participantReactions[p.label][rl] || 0) + 1;
      }
    });
  }));

  const hasData = Object.values(participantReactions).some(r => Object.keys(r).length > 0);

  return (
    <div className="p-4">
      <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2">
        <Zap className="w-3.5 h-3.5" /> Reactions
        <span className="text-[10px] text-slate-600 ml-auto">{snapshots.length} snapshots</span>
      </h4>
      {!hasData ? (
        <p className="text-slate-600 text-xs">No reaction data</p>
      ) : (
        <div className="space-y-3">
          {Object.entries(participantReactions).map(([name, reactions]) => {
            const sorted = Object.entries(reactions).sort((a, b) => b[1] - a[1]).slice(0, 5);
            if (sorted.length === 0) return null;
            return (
              <div key={name}>
                <p className="text-xs text-slate-300 mb-1.5">{name}</p>
                <div className="flex flex-wrap gap-1">
                  {sorted.map(([reaction, count]) => (
                    <span key={reaction} className="px-2 py-0.5 text-[10px] rounded-full bg-rose-500/10 text-rose-300 border border-rose-500/20">
                      {reaction} <span className="text-rose-500/60">x{count}</span>
                    </span>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
