import type { CorrelationPattern } from '../../lib/api';

interface Props { correlations: CorrelationPattern[] }

export default function CorrelationPanel({ correlations }: Props) {
  if (!correlations.length) return <div className="p-4 text-sm text-slate-500">No correlation data</div>;

  return (
    <div className="p-4 space-y-3">
      <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Signal Correlations</h4>
      <div className="space-y-2">
        {correlations.map((c, i) => (
          <div key={i} className="p-3 rounded-lg bg-white/5 border border-white/10 space-y-1.5">
            <div className="flex items-center gap-2">
              {c.signal_pair.map((s, j) => (
                <span key={j} className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">
                  {s}
                </span>
              ))}
              <span className="ml-auto text-xs font-bold text-slate-400">{Math.round(c.confidence * 100)}%</span>
            </div>
            <p className="text-xs text-slate-300">{c.pattern}</p>
            <p className="text-[10px] text-slate-500">{c.implication}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
