import { MessageSquare } from 'lucide-react';
import SignalPanel from './SignalPanel';
import type { WordSignal } from '../../hooks/useRealtimeAnalysis';

interface Props {
  wordSignals: WordSignal[];
  hasAudio?: boolean | null;
}

const sentimentColors: Record<string, string> = {
  positive: '#10b981',
  neutral: '#94a3b8',
  negative: '#ef4444',
  mixed: '#f59e0b',
};

export default function WordsSignal({ wordSignals, hasAudio }: Props) {
  const latest = wordSignals[wordSignals.length - 1];

  // Aggregate all keywords for a word cloud effect
  const allKeywords = new Map<string, number>();
  for (const ws of wordSignals.slice(-5)) {
    for (const kw of ws.keywords) {
      allKeywords.set(kw, (allKeywords.get(kw) || 0) + 1);
    }
  }

  if (!latest) {
    return (
      <SignalPanel icon={<MessageSquare className="w-4 h-4" />} title="Words" color="#10b981">
        <p className="text-xs text-text-muted">
          {hasAudio === false ? 'No audio track in video' : 'Waiting for transcript...'}
        </p>
      </SignalPanel>
    );
  }

  return (
    <SignalPanel
      icon={<MessageSquare className="w-4 h-4" />}
      title="Words"
      color="#10b981"
      lastUpdate={latest.timestamp}
    >
      <div className="space-y-2">
        {/* Sentiment badge */}
        <div className="flex items-center justify-between">
          <span className="text-xs text-text-muted">Sentiment</span>
          <span
            className="text-xs font-bold px-2 py-0.5 rounded-full capitalize"
            style={{
              color: sentimentColors[latest.sentiment] || '#94a3b8',
              backgroundColor: `${sentimentColors[latest.sentiment] || '#94a3b8'}15`,
            }}
          >
            {latest.sentiment}
          </span>
        </div>

        {/* Keyword cloud */}
        {allKeywords.size > 0 && (
          <div className="flex flex-wrap gap-1">
            {Array.from(allKeywords.entries())
              .sort((a, b) => b[1] - a[1])
              .slice(0, 8)
              .map(([word, count]) => (
                <span
                  key={word}
                  className="text-[10px] px-1.5 py-0.5 rounded bg-bg-primary text-text-secondary"
                  style={{ opacity: Math.min(1, 0.5 + count * 0.2) }}
                >
                  {word}
                </span>
              ))}
          </div>
        )}

        {/* Key phrases */}
        {latest.key_phrases.length > 0 && (
          <div className="space-y-0.5">
            {latest.key_phrases.slice(0, 2).map((phrase, i) => (
              <p key={i} className="text-[10px] text-text-muted italic truncate">
                &ldquo;{phrase}&rdquo;
              </p>
            ))}
          </div>
        )}
      </div>
    </SignalPanel>
  );
}
