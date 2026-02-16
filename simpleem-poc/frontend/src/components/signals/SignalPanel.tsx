import type { ReactNode } from 'react';
import { formatTime } from '../../lib/api';

interface Props {
  icon: ReactNode;
  title: string;
  color: string;
  lastUpdate?: number | null;
  children: ReactNode;
}

export default function SignalPanel({ icon, title, color, lastUpdate, children }: Props) {
  const hasData = lastUpdate != null;

  return (
    <div
      className="bg-bg-card border rounded-xl p-4 min-w-[200px] flex-1 transition-all duration-300"
      style={{
        borderColor: hasData ? `${color}30` : undefined,
      }}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div
            className="w-7 h-7 rounded-lg flex items-center justify-center transition-all duration-500"
            style={{ backgroundColor: `${color}${hasData ? '25' : '10'}` }}
          >
            <span style={{ color }}>{icon}</span>
          </div>
          <h4 className="text-xs font-semibold text-text-secondary uppercase tracking-wider">
            {title}
          </h4>
        </div>
        {hasData && (
          <span className="text-[10px] text-text-muted tabular-nums">
            {formatTime(lastUpdate)}
          </span>
        )}
      </div>
      <div className="animate-fade-in">{children}</div>
    </div>
  );
}
