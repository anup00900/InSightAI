import { Heart } from 'lucide-react';
import SignalPanel from './SignalPanel';
import type { SignalSnapshot } from '../../hooks/useRealtimeAnalysis';
import { getEmotionColor } from '../../lib/api';

interface Props {
  signals: SignalSnapshot[];
}

export default function EmotionsSignal({ signals }: Props) {
  const latest = signals[signals.length - 1];

  if (!latest) {
    return (
      <SignalPanel icon={<Heart className="w-4 h-4" />} title="Emotions" color="#ef4444">
        <p className="text-xs text-text-muted">Waiting for data...</p>
      </SignalPanel>
    );
  }

  return (
    <SignalPanel
      icon={<Heart className="w-4 h-4" />}
      title="Emotions"
      color="#ef4444"
      lastUpdate={latest.timestamp}
    >
      <div className="space-y-2">
        {latest.participants.map((p) => (
          <div key={p.label} className="flex items-center justify-between">
            <span className="text-xs text-text-secondary truncate max-w-[80px]">{p.label}</span>
            <div className="flex items-center gap-2">
              <span
                className="text-xs font-bold px-2 py-0.5 rounded-full"
                style={{
                  color: getEmotionColor(p.emotions.primary),
                  backgroundColor: `${getEmotionColor(p.emotions.primary)}15`,
                }}
              >
                {p.emotions.primary}
              </span>
              <div className="w-12 h-1.5 glass-depth-1 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${p.emotions.engagement}%`,
                    backgroundColor: getEmotionColor(p.emotions.primary),
                  }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>
    </SignalPanel>
  );
}
