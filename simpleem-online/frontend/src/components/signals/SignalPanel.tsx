import { type ReactNode, useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { formatTime } from '../../lib/api';

interface Props {
  icon: ReactNode;
  title: string;
  color: string;
  lastUpdate?: number | null;
  children: ReactNode;
  defaultExpanded?: boolean;
}

export default function SignalPanel({ icon, title, color, lastUpdate, children, defaultExpanded = false }: Props) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const hasData = lastUpdate != null;

  return (
    <div
      className="glass-card p-4 min-w-[200px] flex-1 transition-all duration-300"
      style={{
        borderColor: hasData ? `${color}30` : undefined,
      }}
    >
      <div
        className="flex items-center justify-between cursor-pointer select-none"
        onClick={() => setExpanded(!expanded)}
      >
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
        <div className="flex items-center gap-2">
          {hasData && (
            <span className="text-[10px] text-text-muted tabular-nums">
              {formatTime(lastUpdate)}
            </span>
          )}
          {expanded
            ? <ChevronUp className="w-3.5 h-3.5 text-text-muted" />
            : <ChevronDown className="w-3.5 h-3.5 text-text-muted" />
          }
        </div>
      </div>
      {expanded && <div className="animate-fade-in mt-3">{children}</div>}
    </div>
  );
}
