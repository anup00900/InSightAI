import { useMemo, useState } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceArea, ReferenceLine,
} from 'recharts';
import { type EmotionPoint, type Participant, formatTime } from '../lib/api';
import { Activity } from 'lucide-react';

interface Props {
  emotions: EmotionPoint[];
  participants: Participant[];
  isRealtime?: boolean;
}

const PARTICIPANT_COLORS = [
  '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
  '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
];

export default function EmotionTimeline({ emotions, participants, isRealtime = false }: Props) {
  const chartData = useMemo(() => {
    const timeMap = new Map<number, Record<string, number>>();

    for (const e of emotions) {
      // Use 0.5s precision to avoid collision while keeping chart readable
      const ts = Math.round(e.timestamp * 2) / 2;
      if (!timeMap.has(ts)) timeMap.set(ts, { timestamp: ts });
      const entry = timeMap.get(ts)!;
      const p = participants.find((p) => p.id === e.participant_id);
      const key = p?.name || e.participant_id;
      entry[key] = e.engagement;
    }

    return Array.from(timeMap.values()).sort((a, b) => a.timestamp - b.timestamp);
  }, [emotions, participants]);

  const [viewMode, setViewMode] = useState<'combined' | 'individual'>('combined');

  const participantNames = participants.map((p) => p.name);

  // In realtime mode, always show the chart container (even empty with loading state)
  if (chartData.length === 0) {
    return (
      <div className="glass-card p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-text-secondary uppercase tracking-wider">
            Engagement Flow
          </h3>
          {isRealtime && (
            <span className="flex items-center gap-1.5 text-[10px] text-accent">
              <Activity className="w-3 h-3 animate-pulse" />
              LIVE
            </span>
          )}
        </div>
        {isRealtime ? (
          <div className="h-[280px] flex items-center justify-center">
            <div className="flex flex-col items-center gap-3">
              <div className="flex gap-1">
                {[0, 1, 2, 3, 4].map((i) => (
                  <div
                    key={i}
                    className="w-1 bg-accent/40 rounded-full animate-pulse"
                    style={{
                      height: `${20 + Math.random() * 40}px`,
                      animationDelay: `${i * 0.15}s`,
                    }}
                  />
                ))}
              </div>
              <p className="text-text-muted text-xs">Waiting for engagement data...</p>
            </div>
          </div>
        ) : (
          <p className="text-text-muted text-sm text-center py-8">No emotion data available</p>
        )}
      </div>
    );
  }

  return (
    <div className="glass-card p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-4">
          <h3 className="text-sm font-semibold text-text-secondary uppercase tracking-wider">
            Engagement Flow
          </h3>
          {participantNames.length > 1 && (
            <div className="flex bg-bg-primary rounded-lg p-0.5">
              <button
                onClick={() => setViewMode('combined')}
                className={`px-3 py-1 text-[10px] font-medium rounded-md transition-colors ${
                  viewMode === 'combined'
                    ? 'bg-accent/20 text-accent'
                    : 'text-text-muted hover:text-text-secondary'
                }`}
              >
                Combined
              </button>
              <button
                onClick={() => setViewMode('individual')}
                className={`px-3 py-1 text-[10px] font-medium rounded-md transition-colors ${
                  viewMode === 'individual'
                    ? 'bg-accent/20 text-accent'
                    : 'text-text-muted hover:text-text-secondary'
                }`}
              >
                Individual
              </button>
            </div>
          )}
        </div>
        {isRealtime && (
          <div className="flex items-center gap-3">
            <span className="text-[10px] text-text-muted tabular-nums">
              {chartData.length} data points
            </span>
            <span className="flex items-center gap-1.5 text-[10px] text-accent">
              <Activity className="w-3 h-3 animate-pulse" />
              LIVE
            </span>
          </div>
        )}
      </div>
      {viewMode === 'combined' ? (
        <ResponsiveContainer width="100%" height={280}>
          <AreaChart data={chartData}>
            <defs>
              {participantNames.map((name, i) => (
                <linearGradient key={name} id={`grad-${i}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={PARTICIPANT_COLORS[i % PARTICIPANT_COLORS.length]} stopOpacity={0.3} />
                  <stop offset="100%" stopColor={PARTICIPANT_COLORS[i % PARTICIPANT_COLORS.length]} stopOpacity={0} />
                </linearGradient>
              ))}
            </defs>
            {/* Emotion zones: red (low) / green (high) background */}
            <ReferenceArea y1={0} y2={35} fill="#ef4444" fillOpacity={0.06} />
            <ReferenceArea y1={70} y2={100} fill="#10b981" fillOpacity={0.06} />
            <ReferenceLine y={70} stroke="#10b981" strokeDasharray="4 4" strokeOpacity={0.4} label={{ value: 'High', position: 'right', fontSize: 10, fill: '#10b981' }} />
            <ReferenceLine y={35} stroke="#ef4444" strokeDasharray="4 4" strokeOpacity={0.4} label={{ value: 'Low', position: 'right', fontSize: 10, fill: '#ef4444' }} />
            <CartesianGrid strokeDasharray="3 3" stroke="#2a3654" />
            <XAxis
              dataKey="timestamp"
              tickFormatter={formatTime}
              stroke="#64748b"
              fontSize={11}
            />
            <YAxis
              domain={[0, 100]}
              stroke="#64748b"
              fontSize={11}
              tickFormatter={(v) => `${v}%`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1a2332',
                border: '1px solid #2a3654',
                borderRadius: '8px',
                fontSize: '12px',
              }}
              labelFormatter={(label) => formatTime(Number(label))}
              formatter={(value) => [`${Math.round(Number(value))}%`, '']}
            />
            <Legend
              wrapperStyle={{ fontSize: '12px', color: '#94a3b8' }}
            />
            {participantNames.map((name, i) => (
              <Area
                key={name}
                type="monotone"
                dataKey={name}
                stroke={PARTICIPANT_COLORS[i % PARTICIPANT_COLORS.length]}
                fill={`url(#grad-${i})`}
                strokeWidth={2}
                dot={{ r: 3, fill: PARTICIPANT_COLORS[i % PARTICIPANT_COLORS.length] }}
                activeDot={{ r: 5 }}
                isAnimationActive={isRealtime}
                animationDuration={500}
                connectNulls
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      ) : (
        <div className="space-y-4">
          {participantNames.map((name, i) => {
            const color = PARTICIPANT_COLORS[i % PARTICIPANT_COLORS.length];
            return (
              <div key={name} className="bg-bg-primary/50 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
                  <span className="text-xs font-medium text-text-secondary">{name}</span>
                  {participants[i] && (
                    <span className="text-[10px] text-text-muted ml-auto">
                      Avg: {Math.round(participants[i].engagement_score)}%
                    </span>
                  )}
                </div>
                <ResponsiveContainer width="100%" height={120}>
                  <AreaChart data={chartData}>
                    <defs>
                      <linearGradient id={`grad-individual-${i}`} x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={color} stopOpacity={0.4} />
                        <stop offset="100%" stopColor={color} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <ReferenceArea y1={0} y2={35} fill="#ef4444" fillOpacity={0.08} />
                    <ReferenceArea y1={70} y2={100} fill="#10b981" fillOpacity={0.08} />
                    <CartesianGrid strokeDasharray="3 3" stroke="#2a3654" strokeOpacity={0.5} />
                    <XAxis dataKey="timestamp" tickFormatter={formatTime} stroke="#64748b" fontSize={10} />
                    <YAxis domain={[0, 100]} stroke="#64748b" fontSize={10} tickFormatter={(v) => `${v}%`} width={40} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1a2332',
                        border: '1px solid #2a3654',
                        borderRadius: '8px',
                        fontSize: '11px',
                      }}
                      labelFormatter={(label) => formatTime(Number(label))}
                      formatter={(value) => [`${Math.round(Number(value))}%`, name]}
                    />
                    <Area
                      type="monotone"
                      dataKey={name}
                      stroke={color}
                      fill={`url(#grad-individual-${i})`}
                      strokeWidth={2}
                      dot={{ r: 2, fill: color }}
                      activeDot={{ r: 4 }}
                      isAnimationActive={isRealtime}
                      animationDuration={500}
                      connectNulls
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
