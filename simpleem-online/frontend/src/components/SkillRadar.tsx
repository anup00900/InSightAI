import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer,
} from 'recharts';
import { type Participant } from '../lib/api';

interface Props {
  participant: Participant | null;
}

export default function SkillRadar({ participant }: Props) {
  if (!participant) {
    return (
      <div className="glass-card p-6">
        <h3 className="text-sm font-semibold text-text-secondary mb-4 uppercase tracking-wider">
          Skill Radar
        </h3>
        <p className="text-text-muted text-sm text-center py-8">Select a participant</p>
      </div>
    );
  }

  const data = [
    { skill: 'Engagement', value: participant.engagement_score },
    { skill: 'Clarity', value: participant.clarity_score },
    { skill: 'Rapport', value: participant.rapport_score },
    { skill: 'Energy', value: participant.energy_score },
    { skill: 'Sentiment', value: participant.sentiment_score },
  ];

  const hasData = data.some((d) => d.value > 0);

  return (
    <div className="glass-card p-6">
      <h3 className="text-sm font-semibold text-text-secondary mb-1 uppercase tracking-wider">
        Skill Radar
      </h3>
      <p className="text-xs text-accent-light mb-3">{participant.name}</p>
      {hasData ? (
        <ResponsiveContainer width="100%" height={240}>
          <RadarChart data={data} cx="50%" cy="50%" outerRadius="75%">
            <PolarGrid stroke="#2a3654" />
            <PolarAngleAxis
              dataKey="skill"
              tick={{ fill: '#94a3b8', fontSize: 11 }}
            />
            <PolarRadiusAxis
              domain={[0, 100]}
              tick={false}
              axisLine={false}
            />
            <Radar
              dataKey="value"
              stroke="#3b82f6"
              fill="#3b82f6"
              fillOpacity={0.2}
              strokeWidth={2}
              isAnimationActive
              animationDuration={500}
            />
          </RadarChart>
        </ResponsiveContainer>
      ) : (
        <div className="h-[240px] flex items-center justify-center">
          <p className="text-text-muted text-xs">Analyzing skills...</p>
        </div>
      )}
    </div>
  );
}
