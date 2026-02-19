import type { SignalSnapshot } from '../../lib/api';

interface Props { snapshots: SignalSnapshot[] }

// Descriptions to filter out (fallback/error/camera-off text)
const SKIP_DESCRIPTIONS = ['no visual data available', 'model error', 'camera off', 'camera off.', 'camera is off', 'camera is off.', 'camera off, no visible body language.', 'camera off, no body language visible.', 'camera is off; no visible body language.'];
function isRealDescription(desc: string | undefined): boolean {
  if (!desc) return false;
  return !SKIP_DESCRIPTIONS.includes(desc.toLowerCase().trim());
}

export default function BodyLanguageSignal({ snapshots }: Props) {
  if (!snapshots.length) return <div className="p-4 text-sm text-slate-500">No body language data</div>;

  // Build best body language per participant: use the most recent snapshot with real data
  const bestPerParticipant: Record<string, { label: string; body_language: any }> = {};
  // Walk snapshots from newest to oldest
  for (let i = snapshots.length - 1; i >= 0; i--) {
    for (const p of snapshots[i].participants) {
      if (bestPerParticipant[p.label]) continue; // already found best for this person
      if (isRealDescription(p.body_language.description)) {
        bestPerParticipant[p.label] = p;
      }
    }
  }
  // Fallback: if no real data found, use latest snapshot
  const latest = snapshots[snapshots.length - 1];
  for (const p of latest.participants) {
    if (!bestPerParticipant[p.label]) {
      bestPerParticipant[p.label] = p;
    }
  }

  const entries = Object.values(bestPerParticipant);

  return (
    <div className="p-4 space-y-3">
      <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Body Language</h4>
      <div className="space-y-2">
        {entries.map((p, i) => {
          const desc = p.body_language.description;
          const isCameraOff = desc && desc.toLowerCase().includes('camera off');
          return (
            <div key={i} className="p-2 rounded-lg bg-white/5 space-y-1">
              <span className="text-sm text-slate-300 font-medium">{p.label}</span>
              {isCameraOff ? (
                <p className="text-xs text-slate-600 mt-1">Camera off</p>
              ) : (
                <>
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
                  {desc && isRealDescription(desc) && (
                    <p className="text-xs text-slate-500 mt-1">{desc}</p>
                  )}
                </>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
