import { type SignalSnapshot, getEmotionColor } from '../../lib/api';
import { Sparkles } from 'lucide-react';

interface Props { snapshots: SignalSnapshot[] }

export default function EmotionsSignal({ snapshots }: Props) {
  if (!snapshots.length) return <div className="p-4"><h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2"><Sparkles className="w-3.5 h-3.5" /> Emotions</h4><p className="text-slate-600 text-xs">No emotion data</p></div>;

  // Aggregate emotion distribution across ALL snapshots
  const emotionCounts: Record<string, number> = {};
  let totalEntries = 0;
  snapshots.forEach(s => s.participants.forEach(p => {
    const e = p.emotions.primary;
    emotionCounts[e] = (emotionCounts[e] || 0) + 1;
    totalEntries++;
  }));
  const sortedEmotions = Object.entries(emotionCounts).sort((a, b) => b[1] - a[1]).slice(0, 6);

  const latest = snapshots[snapshots.length - 1];

  return (
    <div className="p-4">
      <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2">
        <Sparkles className="w-3.5 h-3.5" /> Emotions
        <span className="text-[10px] text-slate-600 ml-auto">{snapshots.length} snapshots</span>
      </h4>

      {/* Emotion distribution bars */}
      <div className="space-y-1.5 mb-4">
        {sortedEmotions.map(([emotion, count]) => {
          const pct = Math.round((count / totalEntries) * 100);
          return (
            <div key={emotion} className="flex items-center gap-2">
              <span className="text-[10px] text-slate-400 w-20 capitalize">{emotion}</span>
              <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
                <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, backgroundColor: getEmotionColor(emotion) }} />
              </div>
              <span className="text-[10px] font-bold text-slate-400 w-8 text-right">{pct}%</span>
            </div>
          );
        })}
      </div>

      {/* Latest snapshot */}
      <p className="text-[10px] text-slate-600 uppercase tracking-wider mb-2">Current</p>
      {latest.participants.map((p, i) => {
        const conf = (p as any).confidence ?? 1.0;
        return (
          <div key={i} className="flex items-center justify-between py-1">
            <div className="flex items-center gap-1.5">
              <span className={`confidence-dot ${conf >= 0.8 ? 'confidence-high' : conf >= 0.5 ? 'confidence-medium' : 'confidence-low'}`}
                title={`Model confidence: ${Math.round(conf * 100)}%`} />
              <span className="text-xs text-slate-300">{p.label}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs capitalize" style={{ color: getEmotionColor(p.emotions.primary) }}>{p.emotions.primary}</span>
              <span className="text-[10px] text-slate-500">{Math.round(p.emotions.engagement)}%</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
