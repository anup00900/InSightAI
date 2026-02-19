import { Brain } from 'lucide-react';
import SignalPanel from './SignalPanel';
import type { PersonalityData } from '../../hooks/useRealtimeAnalysis';

interface Props {
  personality: PersonalityData | null;
}

export default function PersonalitySignal({ personality }: Props) {
  if (!personality) {
    return (
      <SignalPanel icon={<Brain className="w-4 h-4" />} title="Personality" color="#6366f1">
        <p className="text-xs text-text-muted">Analyzing patterns...</p>
      </SignalPanel>
    );
  }

  return (
    <SignalPanel icon={<Brain className="w-4 h-4" />} title="Personality" color="#6366f1">
      <div className="space-y-2">
        {personality.participants.map((p) => (
          <div key={p.label} className="space-y-1">
            <span className="text-xs text-text-secondary font-medium">{p.label}</span>
            <div className="flex flex-wrap gap-1">
              {p.traits.map((trait) => (
                <span
                  key={trait}
                  className="text-[10px] px-1.5 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 font-medium"
                >
                  {trait}
                </span>
              ))}
            </div>
            <p className="text-[10px] text-text-muted">{p.communication_style}</p>
          </div>
        ))}
      </div>
    </SignalPanel>
  );
}
