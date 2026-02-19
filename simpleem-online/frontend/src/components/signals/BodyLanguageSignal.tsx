import { User } from 'lucide-react';
import SignalPanel from './SignalPanel';
import type { SignalSnapshot } from '../../hooks/useRealtimeAnalysis';

interface Props {
  signals: SignalSnapshot[];
}

const postureIcons: Record<string, string> = {
  upright: '\u2191',    // ↑
  slouched: '\u2193',   // ↓
  leaning: '\u2197',    // ↗
};

const opennessColors: Record<string, string> = {
  open: '#10b981',
  closed: '#ef4444',
  mixed: '#f59e0b',
};

export default function BodyLanguageSignal({ signals }: Props) {
  const latest = signals[signals.length - 1];

  if (!latest) {
    return (
      <SignalPanel icon={<User className="w-4 h-4" />} title="Body Language" color="#8b5cf6">
        <p className="text-xs text-text-muted">Waiting for data...</p>
      </SignalPanel>
    );
  }

  return (
    <SignalPanel
      icon={<User className="w-4 h-4" />}
      title="Body Language"
      color="#8b5cf6"
      lastUpdate={latest.timestamp}
    >
      <div className="space-y-2">
        {latest.participants.map((p) => {
          const bl = p.body_language;
          return (
            <div key={p.label} className="space-y-1">
              <span className="text-xs text-text-secondary">{p.label}</span>
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-xs px-2 py-0.5 rounded bg-bg-primary text-text-secondary">
                  {postureIcons[bl.posture] || ''} {bl.posture}
                </span>
                <span
                  className="text-xs px-2 py-0.5 rounded font-medium"
                  style={{
                    color: opennessColors[bl.openness] || '#94a3b8',
                    backgroundColor: `${opennessColors[bl.openness] || '#94a3b8'}15`,
                  }}
                >
                  {bl.openness}
                </span>
                <span className="text-xs px-2 py-0.5 rounded bg-bg-primary text-text-secondary">
                  {bl.leaning === 'forward' ? '\u21D2' : bl.leaning === 'back' ? '\u21D0' : '\u2194'}{' '}
                  {bl.leaning}
                </span>
                {bl.mirroring && (
                  <span className="text-xs px-2 py-0.5 rounded bg-accent/10 text-accent font-medium">
                    mirroring
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </SignalPanel>
  );
}
