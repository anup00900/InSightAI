import { type Video, formatTime, getScoreColor } from '../lib/api';
import { FileVideo, Clock, Loader2, CheckCircle2, AlertCircle, Radio, Download, Link2, Trash2 } from 'lucide-react';

interface Props {
  videos: Video[];
  onSelect: (id: string) => void;
  onDelete?: (id: string) => void;
}

function StatusBadge({ status }: { status: string }) {
  if (status === 'complete') {
    return (
      <span className="flex items-center gap-1 text-xs font-medium text-success bg-success/10 px-2 py-1 rounded-full">
        <CheckCircle2 className="w-3 h-3" /> Complete
      </span>
    );
  }
  if (status === 'url') {
    return (
      <span className="flex items-center gap-1 text-xs font-medium text-blue-400 bg-blue-400/10 px-2 py-1 rounded-full">
        <Link2 className="w-3 h-3" /> URL — View only
      </span>
    );
  }
  if (status === 'downloading') {
    return (
      <span className="flex items-center gap-1 text-xs font-medium text-blue-400 bg-blue-400/10 px-2 py-1 rounded-full">
        <Download className="w-3 h-3 animate-bounce" /> Downloading...
      </span>
    );
  }
  if (status === 'processing') {
    return (
      <span className="flex items-center gap-1 text-xs font-medium text-accent bg-accent/10 px-2 py-1 rounded-full">
        <Loader2 className="w-3 h-3 animate-spin" /> Processing
      </span>
    );
  }
  if (status.startsWith('error')) {
    return (
      <span className="flex items-center gap-1 text-xs font-medium text-danger bg-danger/10 px-2 py-1 rounded-full">
        <AlertCircle className="w-3 h-3" /> Error
      </span>
    );
  }
  if (status === 'uploaded' || status === 'ready') {
    return (
      <span className="flex items-center gap-1 text-xs font-medium text-emerald-400 bg-emerald-400/10 px-2 py-1 rounded-full">
        <Radio className="w-3 h-3" /> Ready for live analysis
      </span>
    );
  }
  return (
    <span className="flex items-center gap-1 text-xs font-medium text-text-muted bg-bg-card px-2 py-1 rounded-full">
      Uploaded
    </span>
  );
}

export default function VideoLibrary({ videos, onSelect, onDelete }: Props) {
  if (videos.length === 0) {
    return (
      <div className="text-center py-16">
        <FileVideo className="w-16 h-16 text-text-muted mx-auto mb-4 opacity-40" />
        <p className="text-text-secondary text-lg">No recordings yet</p>
        <p className="text-text-muted text-sm mt-1">Upload a video or audio file to get started</p>
      </div>
    );
  }

  return (
    <div>
      <h2 className="text-xl font-semibold text-text-primary mb-4">Recordings</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {videos.map((video) => (
          <div
            key={video.id}
            onClick={() => onSelect(video.id)}
            className="bg-bg-card border border-border rounded-xl p-5 hover:bg-bg-card-hover hover:border-accent/30 transition-all cursor-pointer group"
          >
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-accent/10 flex items-center justify-center">
                  <FileVideo className="w-5 h-5 text-accent" />
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-medium text-text-primary truncate max-w-[200px]">
                    {video.name}
                  </p>
                  <p className="text-xs text-text-muted">
                    {new Date(video.created_at).toLocaleDateString()}
                  </p>
                </div>
              </div>
              <StatusBadge status={video.status} />
            </div>

            <div className="flex items-center justify-between mt-4">
              <div className="flex items-center gap-1 text-xs text-text-muted">
                <Clock className="w-3 h-3" />
                {video.duration > 0 ? formatTime(video.duration) : '--:--'}
              </div>
              <div className="flex items-center gap-3">
                {video.status === 'complete' && (
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-text-muted">Score</span>
                    <span
                      className="text-sm font-bold"
                      style={{ color: getScoreColor(video.overall_score) }}
                    >
                      {Math.round(video.overall_score)}%
                    </span>
                  </div>
                )}
                {onDelete && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      if (confirm('Delete this recording?')) {
                        onDelete(video.id);
                      }
                    }}
                    className="opacity-0 group-hover:opacity-100 transition-opacity p-1.5 rounded-lg hover:bg-danger/10 text-text-muted hover:text-danger"
                    title="Delete recording"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
