import { Hand, ChevronDown, ChevronUp } from 'lucide-react';
import SignalPanel from './SignalPanel';
import type { SignalSnapshot } from '../../hooks/useRealtimeAnalysis';
import { useEffect, useState, useRef } from 'react';

interface Props {
  signals: SignalSnapshot[];
}

interface GestureBadge {
  text: string;
  label: string;
  addedAt: number;
}

export default function GesturesSignal({ signals }: Props) {
  const [badges, setBadges] = useState<GestureBadge[]>([]);
  const [expanded, setExpanded] = useState(false);
  const latest = signals[signals.length - 1];
  const prevSignalCount = useRef(0);

  useEffect(() => {
    if (!latest || signals.length <= prevSignalCount.current) return;
    prevSignalCount.current = signals.length;

    const now = Date.now();
    const newBadges: GestureBadge[] = [];
    for (const p of latest.participants) {
      for (const g of p.gestures) {
        newBadges.push({ text: g, label: p.label, addedAt: now });
      }
    }

    // Keep badges but cap at 50 to prevent memory leak in long sessions
    setBadges((prev) => {
      const combined = [...prev, ...newBadges];
      return combined.length > 50 ? combined.slice(-50) : combined;
    });
  }, [latest, signals.length]);

  // Show only latest 6 when collapsed, all when expanded
  const visibleBadges = expanded ? badges : badges.slice(-6);
  const hasOverflow = badges.length > 6;

  return (
    <SignalPanel
      icon={<Hand className="w-4 h-4" />}
      title="Gestures"
      color="#f59e0b"
      lastUpdate={latest?.timestamp ?? null}
    >
      <div className="relative">
        {badges.length === 0 ? (
          <p className="text-xs text-text-muted">Waiting for gestures...</p>
        ) : (
          <>
            <div className={`flex flex-wrap gap-1.5 ${expanded ? 'max-h-[200px] overflow-y-auto pr-1' : 'max-h-[52px] overflow-hidden'} transition-all duration-300`}>
              {visibleBadges.map((b, i) => {
                const isNew = Date.now() - b.addedAt < 3000;
                return (
                  <span
                    key={`${b.text}-${b.label}-${i}`}
                    className={`text-xs px-2 py-1 rounded-full bg-warning/10 text-warning font-medium ${isNew ? 'animate-fade-in' : ''}`}
                  >
                    {b.text}
                  </span>
                );
              })}
            </div>
            {hasOverflow && (
              <button
                onClick={() => setExpanded(!expanded)}
                className="flex items-center gap-1 mt-1.5 text-[10px] text-text-muted hover:text-text-secondary transition-colors"
              >
                {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                {expanded ? 'Collapse' : `+${badges.length - 6} more`}
              </button>
            )}
          </>
        )}
      </div>
    </SignalPanel>
  );
}
