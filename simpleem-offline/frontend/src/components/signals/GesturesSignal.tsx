import { type SignalSnapshot } from '../../lib/api';
import { Hand } from 'lucide-react';

interface Props { snapshots: SignalSnapshot[] }

const USELESS = new Set(['still posture', 'not visible', 'no gesture', 'none', 'n/a', 'unknown']);

export default function GesturesSignal({ snapshots }: Props) {
  if (!snapshots.length) return <div className="p-4"><h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2"><Hand className="w-3.5 h-3.5" /> Gestures</h4><p className="text-slate-600 text-xs">No gesture data</p></div>;

  // Aggregate gesture frequency across ALL snapshots, per participant
  const participantGestures: Record<string, Record<string, number>> = {};
  snapshots.forEach(s => s.participants.forEach(p => {
    if (!participantGestures[p.label]) participantGestures[p.label] = {};
    (p.gestures || []).forEach(g => {
      const gl = g.toLowerCase().trim();
      if (!USELESS.has(gl)) {
        participantGestures[p.label][gl] = (participantGestures[p.label][gl] || 0) + 1;
      }
    });
  }));

  const hasData = Object.values(participantGestures).some(g => Object.keys(g).length > 0);

  return (
    <div className="p-4">
      <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2">
        <Hand className="w-3.5 h-3.5" /> Gestures
        <span className="text-[10px] text-slate-600 ml-auto">{snapshots.length} snapshots</span>
      </h4>
      {!hasData ? (
        <p className="text-slate-600 text-xs">No visible gesture data</p>
      ) : (
        <div className="space-y-3">
          {Object.entries(participantGestures).map(([name, gestures]) => {
            const sorted = Object.entries(gestures).sort((a, b) => b[1] - a[1]).slice(0, 5);
            if (sorted.length === 0) return null;
            return (
              <div key={name}>
                <p className="text-xs text-slate-300 mb-1.5">{name}</p>
                <div className="flex flex-wrap gap-1">
                  {sorted.map(([gesture, count]) => (
                    <span key={gesture} className="px-2 py-0.5 text-[10px] rounded-full bg-amber-500/10 text-amber-300 border border-amber-500/20">
                      {gesture} <span className="text-amber-500/60">x{count}</span>
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
