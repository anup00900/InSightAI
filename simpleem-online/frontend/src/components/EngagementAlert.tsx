import { AlertTriangle, X } from 'lucide-react';
import { useState, useEffect } from 'react';
import type { EngagementAlertData } from '../hooks/useRealtimeAnalysis';
import { formatTime } from '../lib/api';

interface Props {
  alerts: EngagementAlertData[];
}

export default function EngagementAlerts({ alerts }: Props) {
  const [dismissed, setDismissed] = useState<Set<number>>(new Set());

  // Auto-dismiss after 8 seconds
  useEffect(() => {
    if (alerts.length === 0) return;
    const timer = setTimeout(() => {
      setDismissed(new Set(alerts.map((_, i) => i)));
    }, 8000);
    return () => clearTimeout(timer);
  }, [alerts.length]);

  // Only show latest 2 non-dismissed alerts (skip "Person N" ghost alerts)
  const visible = alerts
    .map((alert, i) => ({ alert, i }))
    .filter(({ alert, i }) => !dismissed.has(i) && !alert.participant.match(/^Person \d+$/))
    .slice(-2);

  if (visible.length === 0) return null;

  return (
    <div className="fixed top-20 right-6 z-50 space-y-2 max-w-sm">
      {visible.map(({ alert, i }) => (
        <div
          key={`${alert.timestamp}-${i}`}
          className="glass-card p-4 border-red-500/30 shadow-2xl animate-fade-in"
          style={{ background: 'rgba(127,29,29,0.6)' }}
        >
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="text-sm font-semibold text-red-200">
                Low Engagement: {alert.participant}
              </p>
              <p className="text-xs text-red-300 mt-1">
                Score dropped to {Math.round(alert.score)}% for {Math.round(alert.duration_below_threshold)}s
                (at {formatTime(alert.timestamp)})
              </p>
              <p className="text-xs text-red-400 mt-1 italic">
                {alert.suggested_action}
              </p>
            </div>
            <button
              onClick={() => setDismissed(new Set([...dismissed, i]))}
              className="text-red-400 hover:text-red-200"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}
