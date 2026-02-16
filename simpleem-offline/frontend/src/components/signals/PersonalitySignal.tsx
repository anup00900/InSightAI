import type { PersonalitySignalData } from '../../lib/api';

interface Props { data: PersonalitySignalData | null }

export default function PersonalitySignal({ data }: Props) {
  if (!data?.participants?.length) return <div className="p-4 text-sm text-slate-500">No personality data</div>;

  return (
    <div className="p-4 space-y-3">
      <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Personality Profiles</h4>
      <div className="space-y-3">
        {data.participants.map((p, i) => (
          <div key={i} className="p-3 rounded-lg bg-white/5 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-300 font-medium">{p.label}</span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">
                {p.confidence_level}
              </span>
            </div>
            <div className="flex flex-wrap gap-1">
              {p.traits.map((t, j) => (
                <span key={j} className="text-[10px] px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
                  {t}
                </span>
              ))}
            </div>
            <p className="text-xs text-slate-500">{p.communication_style}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
