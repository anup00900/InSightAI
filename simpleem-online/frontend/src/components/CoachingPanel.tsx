import { useState, useEffect, useRef, useCallback } from 'react';
import { type CoachingItem, type Participant, getCoaching } from '../lib/api';
import { Lightbulb, ChevronDown } from 'lucide-react';

interface Props {
  videoId: string;
  participants: Participant[];
}

const CATEGORY_COLORS: Record<string, string> = {
  communication: '#3b82f6',
  engagement: '#10b981',
  leadership: '#8b5cf6',
  listening: '#f59e0b',
  general: '#94a3b8',
};

export default function CoachingPanel({ videoId, participants }: Props) {
  const [selectedPid, setSelectedPid] = useState<string>('');
  const [items, setItems] = useState<CoachingItem[]>([]);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  const handleClickOutside = useCallback((e: MouseEvent) => {
    if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
      setDropdownOpen(false);
    }
  }, []);

  useEffect(() => {
    if (dropdownOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [dropdownOpen, handleClickOutside]);

  useEffect(() => {
    if (participants.length > 0 && !selectedPid) {
      setSelectedPid(participants[0].id);
    }
  }, [participants, selectedPid]);

  useEffect(() => {
    if (!selectedPid || !videoId) return;
    getCoaching(videoId, selectedPid).then(setItems).catch(() => setItems([]));
  }, [videoId, selectedPid]);

  const selectedName = participants.find((p) => p.id === selectedPid)?.name || 'Select';

  return (
    <div>
      {/* Participant Selector */}
      <div className="relative mb-4" ref={dropdownRef}>
        <button
          onClick={() => setDropdownOpen(!dropdownOpen)}
          className="flex items-center gap-2 px-4 py-2 bg-bg-card border border-border rounded-lg text-sm text-text-primary hover:bg-bg-card-hover transition-colors"
        >
          {selectedName}
          <ChevronDown className="w-4 h-4 text-text-muted" />
        </button>
        {dropdownOpen && (
          <div className="absolute top-full left-0 mt-1 bg-bg-card border border-border rounded-lg shadow-xl z-10 min-w-[200px]">
            {participants.map((p) => (
              <button
                key={p.id}
                onClick={() => { setSelectedPid(p.id); setDropdownOpen(false); }}
                className={`
                  w-full text-left px-4 py-2.5 text-sm transition-colors
                  ${p.id === selectedPid ? 'text-accent bg-accent/10' : 'text-text-primary hover:bg-bg-card-hover'}
                `}
              >
                {p.name}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Coaching Items */}
      {items.length === 0 ? (
        <p className="text-text-muted text-sm text-center py-8">
          No coaching recommendations available
        </p>
      ) : (
        <div className="space-y-3">
          {items.map((item) => (
            <div
              key={item.id}
              className="flex items-start gap-3 p-4 bg-bg-primary/50 border border-border rounded-lg"
            >
              <div
                className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0"
                style={{
                  backgroundColor: `${CATEGORY_COLORS[item.category] || CATEGORY_COLORS.general}15`,
                }}
              >
                <Lightbulb
                  className="w-4 h-4"
                  style={{ color: CATEGORY_COLORS[item.category] || CATEGORY_COLORS.general }}
                />
              </div>
              <div className="flex-1">
                <p className="text-sm text-text-primary">{item.recommendation}</p>
                <div className="flex items-center gap-2 mt-2">
                  <span
                    className="text-[10px] font-bold uppercase px-2 py-0.5 rounded-full"
                    style={{
                      color: CATEGORY_COLORS[item.category] || CATEGORY_COLORS.general,
                      backgroundColor: `${CATEGORY_COLORS[item.category] || CATEGORY_COLORS.general}15`,
                    }}
                  >
                    {item.category}
                  </span>
                  <span className="text-[10px] text-text-muted">
                    Priority: {item.priority}/5
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
