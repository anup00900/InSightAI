import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { type Participant } from '../lib/api';

interface Props {
  participants: Participant[];
}

const COLORS = [
  '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
  '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
];

export default function SpeakingDonut({ participants }: Props) {
  const data = participants
    .map((p) => ({
      name: p.name,
      value: p.speaking_pct,
    }))
    .filter((d) => d.value > 0);

  // Normalize to sum to 100 and round properly
  const total = data.reduce((sum, d) => sum + d.value, 0);
  const normalized = data.map((d, i) => ({
    ...d,
    value: i === data.length - 1
      ? 100 - data.slice(0, -1).reduce((sum, x) => sum + Math.round((x.value / total) * 100), 0)
      : Math.round((d.value / total) * 100),
  }));

  const hasData = normalized.some((d) => d.value > 0);

  return (
    <div className="bg-bg-card border border-border rounded-xl p-6">
      <h3 className="text-sm font-semibold text-text-secondary mb-4 uppercase tracking-wider">
        Speaking Distribution
      </h3>
      {hasData ? (
        <>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie
                data={normalized}
                cx="50%"
                cy="50%"
                innerRadius={55}
                outerRadius={85}
                paddingAngle={3}
                dataKey="value"
                stroke="none"
                isAnimationActive
                animationDuration={500}
              >
                {normalized.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1a2332',
                  border: '1px solid #2a3654',
                  borderRadius: '8px',
                  fontSize: '12px',
                }}
                formatter={(value) => [`${value}%`, 'Speaking']}
              />
            </PieChart>
          </ResponsiveContainer>
          <div className="flex flex-wrap gap-3 justify-center mt-2">
            {normalized.map((d, i) => (
              <div key={d.name} className="flex items-center gap-1.5">
                <div
                  className="w-2.5 h-2.5 rounded-full"
                  style={{ backgroundColor: COLORS[i % COLORS.length] }}
                />
                <span className="text-xs text-text-muted">{d.name}</span>
                <span className="text-xs font-bold text-text-secondary">{d.value}%</span>
              </div>
            ))}
          </div>
        </>
      ) : (
        <div className="h-[220px] flex items-center justify-center">
          <p className="text-text-muted text-xs">Analyzing speaking patterns...</p>
        </div>
      )}
    </div>
  );
}
