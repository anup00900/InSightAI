import { type Flag, formatTime } from '../lib/api';
import { AlertTriangle, CheckCircle2, Info } from 'lucide-react';

interface Props {
  flags: Flag[];
}

function FlagIcon({ type, severity }: { type: string; severity: string }) {
  if (type === 'positive') {
    return <CheckCircle2 className="w-4 h-4 text-success" />;
  }
  if (severity === 'high') {
    return <AlertTriangle className="w-4 h-4 text-danger" />;
  }
  return <Info className="w-4 h-4 text-warning" />;
}

export default function FlagTimeline({ flags }: Props) {
  if (flags.length === 0) {
    return (
      <div className="bg-bg-card border border-border rounded-xl p-6">
        <h3 className="text-sm font-semibold text-text-secondary mb-4 uppercase tracking-wider">
          Critical Moments
        </h3>
        <p className="text-text-muted text-sm text-center py-4">No critical moments detected</p>
      </div>
    );
  }

  return (
    <div className="bg-bg-card border border-border rounded-xl p-6">
      <h3 className="text-sm font-semibold text-text-secondary mb-4 uppercase tracking-wider">
        Critical Moments
      </h3>
      <div className="space-y-3 max-h-[300px] overflow-y-auto pr-1">
        {flags.map((flag) => (
          <div
            key={flag.id}
            className={`
              flex items-start gap-3 p-3 rounded-lg border transition-colors
              ${flag.type === 'positive'
                ? 'border-success/20 bg-success/5'
                : flag.severity === 'high'
                  ? 'border-danger/20 bg-danger/5'
                  : 'border-warning/20 bg-warning/5'
              }
            `}
          >
            <FlagIcon type={flag.type} severity={flag.severity} />
            <div className="flex-1 min-w-0">
              <p className="text-sm text-text-primary">{flag.description}</p>
              <p className="text-xs text-text-muted mt-1">
                at {formatTime(flag.timestamp)}
              </p>
            </div>
            <span
              className={`
                text-[10px] font-bold uppercase px-2 py-0.5 rounded-full
                ${flag.type === 'positive'
                  ? 'text-success bg-success/10'
                  : 'text-danger bg-danger/10'
                }
              `}
            >
              {flag.type}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
