import type { WordSignalData } from '../../lib/api';

interface Props { signals: WordSignalData[] }

export default function WordsSignal({ signals }: Props) {
  if (!signals.length) return <div className="p-4 text-sm text-slate-500">No word signal data</div>;

  // Collect all keywords across segments
  const allKeywords = signals.flatMap(s => s.keywords);
  const keywordCounts = allKeywords.reduce<Record<string, number>>((acc, k) => {
    acc[k] = (acc[k] || 0) + 1;
    return acc;
  }, {});
  const topKeywords = Object.entries(keywordCounts)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 10);

  const avgSentiment = signals.reduce((s, w) => s + w.sentiment_score, 0) / signals.length;

  return (
    <div className="p-4 space-y-3">
      <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Word Analysis</h4>
      <div className="flex items-center gap-3">
        <span className="text-xs text-slate-500">Avg Sentiment:</span>
        <span className={`text-sm font-bold ${avgSentiment > 0.6 ? 'text-emerald-400' : avgSentiment > 0.4 ? 'text-amber-400' : 'text-red-400'}`}>
          {(avgSentiment * 100).toFixed(0)}%
        </span>
      </div>
      {topKeywords.length > 0 && (
        <div>
          <span className="text-xs text-slate-500">Top Keywords:</span>
          <div className="flex flex-wrap gap-1.5 mt-1.5">
            {topKeywords.map(([kw, count]) => (
              <span key={kw} className="text-[10px] px-2 py-0.5 rounded-full bg-teal-500/10 text-teal-300 border border-teal-500/20">
                {kw}{count > 1 && ` (${count})`}
              </span>
            ))}
          </div>
        </div>
      )}
      {/* Key phrases from latest segment */}
      {signals[signals.length - 1]?.key_phrases?.length > 0 && (
        <div>
          <span className="text-xs text-slate-500">Key Phrases:</span>
          <div className="space-y-1 mt-1">
            {signals[signals.length - 1].key_phrases.map((phrase, i) => (
              <p key={i} className="text-xs text-slate-400 italic">&quot;{phrase}&quot;</p>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
