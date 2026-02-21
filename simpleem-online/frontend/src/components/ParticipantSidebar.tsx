import { useState, useRef, useEffect } from 'react';
import { type Participant, getScoreColor } from '../lib/api';
import { User, Pencil, Check } from 'lucide-react';

interface Props {
  participants: Participant[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}

export default function ParticipantSidebar({ participants, selectedId, onSelect }: Props) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState('');
  const [savedId, setSavedId] = useState<string | null>(null);
  const [localNames, setLocalNames] = useState<Record<string, string>>({});
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editingId && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editingId]);

  const startEditing = (e: React.MouseEvent, p: Participant) => {
    e.stopPropagation();
    setEditingId(p.id);
    setEditValue(localNames[p.id] || p.name);
  };

  const submitRename = async (p: Participant) => {
    const trimmed = editValue.trim();
    if (!trimmed || trimmed === p.name) {
      setEditingId(null);
      return;
    }

    try {
      const res = await fetch(
        `/api/videos/${p.video_id}/participants/${p.id}/rename`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: trimmed }),
        }
      );
      if (res.ok) {
        const data = await res.json();
        setLocalNames((prev) => ({ ...prev, [p.id]: data.name }));
        setSavedId(p.id);
        setTimeout(() => setSavedId(null), 2000);
      }
    } catch {
      // silently fail, keep original name
    }
    setEditingId(null);
  };

  const handleKeyDown = (e: React.KeyboardEvent, p: Participant) => {
    if (e.key === 'Enter') {
      submitRename(p);
    } else if (e.key === 'Escape') {
      setEditingId(null);
    }
  };

  return (
    <div className="glass-card p-4">
      <h3 className="text-sm font-semibold text-text-secondary mb-3 uppercase tracking-wider">
        Participants ({participants.length})
      </h3>
      <div className="space-y-2 max-h-[400px] overflow-y-auto pr-1">
        {participants.map((p) => (
          <div
            key={p.id}
            onClick={() => onSelect(selectedId === p.id ? null : p.id)}
            className={`
              flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-all
              ${selectedId === p.id
                ? 'bg-accent/10 border border-accent/30'
                : 'glass-card-hover border border-transparent'
              }
            `}
          >
            <div className="w-9 h-9 rounded-full bg-accent/20 flex items-center justify-center shrink-0">
              <User className="w-4 h-4 text-accent-light" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-1.5 group">
                {editingId === p.id ? (
                  <input
                    ref={inputRef}
                    type="text"
                    value={editValue}
                    onChange={(e) => setEditValue(e.target.value)}
                    onBlur={() => submitRename(p)}
                    onKeyDown={(e) => handleKeyDown(e, p)}
                    onClick={(e) => e.stopPropagation()}
                    maxLength={100}
                    className="text-sm font-medium text-text-primary bg-bg-primary border border-accent/40 rounded px-1.5 py-0.5 w-full outline-none focus:border-accent"
                  />
                ) : (
                  <>
                    <p className="text-sm font-medium text-text-primary truncate">
                      {localNames[p.id] || p.name}
                    </p>
                    {savedId === p.id ? (
                      <Check className="w-3.5 h-3.5 text-green-400 shrink-0" />
                    ) : (
                      <button
                        onClick={(e) => startEditing(e, p)}
                        className="opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
                        title="Rename participant"
                      >
                        <Pencil className="w-3.5 h-3.5 text-text-muted hover:text-accent-light" />
                      </button>
                    )}
                  </>
                )}
              </div>
              <div className="flex items-center gap-3 mt-1">
                <span className="text-xs text-text-muted">
                  Engagement
                </span>
                <div className="flex-1 h-1.5 bg-bg-primary rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{
                      width: `${p.engagement_score}%`,
                      backgroundColor: getScoreColor(p.engagement_score),
                    }}
                  />
                </div>
                <span
                  className="text-xs font-bold tabular-nums"
                  style={{ color: getScoreColor(p.engagement_score) }}
                >
                  {Math.round(p.engagement_score)}%
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
