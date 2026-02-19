import { Captions, CaptionsOff } from 'lucide-react';
import type { TranscriptSegment } from '../lib/api';

interface Props {
  transcript: TranscriptSegment[];
  currentTime: number;
  enabled: boolean;
  onToggle: () => void;
}

export default function CaptionOverlay({ transcript, currentTime, enabled, onToggle }: Props) {
  // Find transcript segments near current playback time (last 2-3 visible)
  const visibleSegments = transcript.filter(
    (seg) => seg.start_time <= currentTime && seg.end_time >= currentTime - 5
  ).slice(-3);

  return (
    <>
      {/* CC Toggle Button */}
      <button
        onClick={onToggle}
        className={`absolute top-3 right-3 z-10 p-2 rounded-lg transition-colors ${
          enabled
            ? 'bg-accent/80 text-white'
            : 'bg-black/40 text-white/50 hover:text-white/80'
        }`}
        title={enabled ? 'Hide captions' : 'Show captions'}
      >
        {enabled ? (
          <Captions className="w-4 h-4" />
        ) : (
          <CaptionsOff className="w-4 h-4" />
        )}
      </button>

      {/* Caption overlay */}
      {enabled && visibleSegments.length > 0 && (
        <div className="absolute bottom-14 left-4 right-4 z-10 pointer-events-none">
          <div className="bg-black/75 backdrop-blur-sm rounded-lg px-4 py-2 space-y-1">
            {visibleSegments.map((seg, i) => (
              <p
                key={`${seg.start_time}-${i}`}
                className="text-sm text-white animate-fade-in"
                style={{
                  opacity: i === visibleSegments.length - 1 ? 1 : 0.6,
                }}
              >
                {seg.speaker && (
                  <span className="text-accent font-semibold mr-2">{seg.speaker}:</span>
                )}
                {seg.text}
              </p>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
