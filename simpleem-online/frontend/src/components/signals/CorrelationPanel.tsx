import { GitBranch } from 'lucide-react';
import type { CorrelationData } from '../../hooks/useRealtimeAnalysis';

interface Props {
  correlations: CorrelationData[];
}

const signalColors: Record<string, string> = {
  emotions: '#ef4444',
  body_language: '#8b5cf6',
  gestures: '#f59e0b',
  reactions: '#ec4899',
  voice: '#06b6d4',
  words: '#10b981',
  personality: '#6366f1',
};

export default function CorrelationPanel({ correlations }: Props) {
  if (correlations.length === 0) {
    return (
      <div className="glass-card p-4">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-7 h-7 rounded-lg bg-teal-500/20 flex items-center justify-center">
            <GitBranch className="w-4 h-4 text-teal-400" />
          </div>
          <h4 className="text-xs font-semibold text-text-secondary uppercase tracking-wider">
            Cross-Signal Patterns
          </h4>
        </div>
        <p className="text-xs text-text-muted">Patterns will appear as data accumulates...</p>
      </div>
    );
  }

  return (
    <div className="glass-card p-4">
      <div className="flex items-center gap-2 mb-3">
        <div className="w-7 h-7 rounded-lg bg-teal-500/20 flex items-center justify-center">
          <GitBranch className="w-4 h-4 text-teal-400" />
        </div>
        <h4 className="text-xs font-semibold text-text-secondary uppercase tracking-wider">
          Cross-Signal Patterns
        </h4>
      </div>
      <div className="space-y-3 max-h-[300px] overflow-y-auto pr-1">
        {correlations.map((corr, i) => (
          <div key={i} className="p-3 bg-bg-primary/50 rounded-lg border border-border/50">
            {/* Signal pair indicators */}
            <div className="flex items-center gap-1.5 mb-1.5">
              {corr.signal_pair.map((signal) => (
                <span
                  key={signal}
                  className="text-[10px] px-1.5 py-0.5 rounded font-medium"
                  style={{
                    color: signalColors[signal] || '#94a3b8',
                    backgroundColor: `${signalColors[signal] || '#94a3b8'}15`,
                  }}
                >
                  {signal.replace('_', ' ')}
                </span>
              ))}
              <span className="text-[10px] text-text-muted ml-auto tabular-nums">
                {Math.round(corr.confidence * 100)}%
              </span>
            </div>
            <p className="text-xs text-text-primary leading-relaxed">{corr.pattern}</p>
            {corr.implication && (
              <p className="text-[10px] text-text-muted mt-1 italic">{corr.implication}</p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
