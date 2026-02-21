import { Eye, Mic, MessageSquare } from 'lucide-react';

interface Props {
  visualScore: number;   // 0-100, latest visual engagement
  audioScore: number;    // 0-100, latest audio engagement
  verbalScore: number;   // 0-100, latest verbal engagement
}

export default function SignalWeightCard({ visualScore, audioScore, verbalScore }: Props) {
  // Calculate holistic score using 55/38/7 model
  const holisticScore = Math.round(0.55 * visualScore + 0.38 * audioScore + 0.07 * verbalScore);

  // Calculate fill percentages for each segment (0-1 scale)
  const visualFill = visualScore / 100;
  const audioFill = audioScore / 100;
  const verbalFill = verbalScore / 100;

  return (
    <div className="glass-card p-4">
      {/* Header */}
      <div className="mb-4">
        <h3 className="text-text-primary font-semibold text-lg">Signal Weights</h3>
        <p className="text-text-muted text-sm">55/38/7 Model</p>
      </div>

      {/* Horizontal Stacked Bar */}
      <div className="mb-4 flex h-3 w-full rounded-full overflow-hidden bg-gray-800">
        {/* Visual segment - 55% width */}
        <div className="relative" style={{ width: '55%' }}>
          <div
            className="h-full bg-purple-500 transition-all duration-300"
            style={{ width: `${visualFill * 100}%` }}
          />
        </div>

        {/* Audio segment - 38% width */}
        <div className="relative" style={{ width: '38%' }}>
          <div
            className="h-full bg-cyan-500 transition-all duration-300"
            style={{ width: `${audioFill * 100}%` }}
          />
        </div>

        {/* Verbal segment - 7% width */}
        <div className="relative" style={{ width: '7%' }}>
          <div
            className="h-full bg-amber-500 transition-all duration-300"
            style={{ width: `${verbalFill * 100}%` }}
          />
        </div>
      </div>

      {/* Signal Details */}
      <div className="space-y-3 mb-4">
        {/* Visual */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Eye className="w-4 h-4" style={{ color: '#8b5cf6' }} />
            <span className="text-text-primary text-sm font-medium">Visual</span>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-text-muted text-sm">55%</span>
            <span className="text-sm font-semibold w-8 text-right" style={{ color: '#8b5cf6' }}>
              {visualScore}
            </span>
          </div>
        </div>

        {/* Audio */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Mic className="w-4 h-4" style={{ color: '#06b6d4' }} />
            <span className="text-text-primary text-sm font-medium">Audio</span>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-text-muted text-sm">38%</span>
            <span className="text-sm font-semibold w-8 text-right" style={{ color: '#06b6d4' }}>
              {audioScore}
            </span>
          </div>
        </div>

        {/* Verbal */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <MessageSquare className="w-4 h-4" style={{ color: '#f59e0b' }} />
            <span className="text-text-primary text-sm font-medium">Verbal</span>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-text-muted text-sm">7%</span>
            <span className="text-sm font-semibold w-8 text-right" style={{ color: '#f59e0b' }}>
              {verbalScore}
            </span>
          </div>
        </div>
      </div>

      {/* Holistic Score */}
      <div className="pt-3 border-t border-border">
        <div className="flex items-center justify-between">
          <span className="text-text-primary font-semibold">Holistic</span>
          <span className="text-text-primary font-bold text-lg">{holisticScore}</span>
        </div>
      </div>
    </div>
  );
}
