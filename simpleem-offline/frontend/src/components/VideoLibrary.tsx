import { type Video, formatTime, getScoreColor, getThumbnailUrl } from '../lib/api';
import {
  FileVideo,
  Clock,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Trash2,
  Sparkles,
  BarChart3,
  Users,
  Zap,
  ArrowRight,
  Calendar,
  TrendingUp,
} from 'lucide-react';

interface Props {
  videos: Video[];
  onSelect: (id: string) => void;
  onDelete?: (id: string) => void;
  onNewAnalysis?: () => void;
}

function StatusBadge({ status }: { status: string }) {
  if (status === 'complete') {
    return (
      <span className="flex items-center gap-1 text-xs font-medium text-emerald-400 bg-emerald-400/10 px-2 py-1 rounded-full border border-emerald-400/20">
        <CheckCircle2 className="w-3 h-3" /> Complete
      </span>
    );
  }
  if (status === 'transcribing') {
    return (
      <span className="flex items-center gap-1 text-xs font-medium text-blue-400 bg-blue-400/10 px-2 py-1 rounded-full border border-blue-400/20">
        <Loader2 className="w-3 h-3 animate-spin" /> Transcribing...
      </span>
    );
  }
  if (status === 'analyzing') {
    return (
      <span className="flex items-center gap-1 text-xs font-medium text-purple-400 bg-purple-400/10 px-2 py-1 rounded-full border border-purple-400/20">
        <Loader2 className="w-3 h-3 animate-spin" /> Analyzing...
      </span>
    );
  }
  if (status === 'processing') {
    return (
      <span className="flex items-center gap-1 text-xs font-medium text-violet-400 bg-violet-400/10 px-2 py-1 rounded-full border border-violet-400/20">
        <Loader2 className="w-3 h-3 animate-spin" /> Processing
      </span>
    );
  }
  if (status.startsWith('error')) {
    return (
      <span className="flex items-center gap-1 text-xs font-medium text-red-400 bg-red-400/10 px-2 py-1 rounded-full border border-red-400/20">
        <AlertCircle className="w-3 h-3" /> Error
      </span>
    );
  }
  if (status === 'uploaded' || status === 'ready') {
    return (
      <span className="flex items-center gap-1 text-xs font-medium text-cyan-400 bg-cyan-400/10 px-2 py-1 rounded-full border border-cyan-400/20">
        <CheckCircle2 className="w-3 h-3" /> Ready for transcription
      </span>
    );
  }
  return (
    <span className="flex items-center gap-1 text-xs font-medium text-slate-400 bg-white/5 px-2 py-1 rounded-full border border-white/10">
      Uploaded
    </span>
  );
}

/* ── Circular Score Badge ──────────────────────────────────────────── */
function ScoreBadge({ score }: { score: number }) {
  const rounded = Math.round(score);
  const color = getScoreColor(score);
  const circumference = 2 * Math.PI * 18;
  const offset = circumference - (rounded / 100) * circumference;

  return (
    <div className="relative w-12 h-12 flex items-center justify-center flex-shrink-0">
      <svg className="w-12 h-12 -rotate-90" viewBox="0 0 40 40">
        <circle
          cx="20"
          cy="20"
          r="18"
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth="3"
        />
        <circle
          cx="20"
          cy="20"
          r="18"
          fill="none"
          stroke={color}
          strokeWidth="3"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ transition: 'stroke-dashoffset 0.6s ease' }}
        />
      </svg>
      <span
        className="absolute text-[11px] font-bold"
        style={{ color }}
      >
        {rounded}
      </span>
    </div>
  );
}

/* ── Feature Card (used in Empty State) ────────────────────────────── */
function FeatureCard({
  icon: Icon,
  title,
  description,
  delay,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  description: string;
  delay: string;
}) {
  return (
    <div
      className="glass-card p-6 flex flex-col items-center text-center animate-slide-up"
      style={{ animationDelay: delay, animationFillMode: 'both' }}
    >
      <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-indigo-500/20 to-purple-500/20 flex items-center justify-center border border-purple-500/20 mb-4">
        <Icon className="w-6 h-6 text-purple-400" />
      </div>
      <h3 className="text-sm font-semibold text-text-primary mb-1">{title}</h3>
      <p className="text-xs text-text-muted leading-relaxed">{description}</p>
    </div>
  );
}

/* ── Empty State ───────────────────────────────────────────────────── */
function EmptyState({ onNewAnalysis }: { onNewAnalysis?: () => void }) {
  return (
    <div className="flex items-center justify-center py-12 animate-fade-in">
      <div className="glass-card max-w-2xl w-full px-8 py-12 text-center relative overflow-hidden">
        {/* Subtle background glow */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-80 h-80 bg-purple-500/[0.07] rounded-full blur-[100px]" />
          <div className="absolute bottom-0 left-1/4 w-60 h-60 bg-blue-500/[0.05] rounded-full blur-[80px]" />
        </div>

        {/* Animated icon */}
        <div className="relative z-10 mb-6 flex justify-center">
          <div className="w-20 h-20 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-purple-500/30 animate-glow-pulse">
            <Sparkles className="w-10 h-10 text-white" />
          </div>
        </div>

        {/* Title */}
        <h2 className="relative z-10 text-3xl font-bold gradient-text mb-3">
          AI-Powered Meeting Intelligence
        </h2>

        {/* Subtitle */}
        <p className="relative z-10 text-slate-400 text-base max-w-lg mx-auto mb-10 leading-relaxed">
          Upload a recorded meeting to get instant insights on engagement,
          communication, and coaching opportunities.
        </p>

        {/* Feature cards */}
        <div className="relative z-10 grid grid-cols-1 sm:grid-cols-3 gap-4 mb-10">
          <FeatureCard
            icon={BarChart3}
            title="Engagement Analysis"
            description="Visual + Audio + Verbal scoring"
            delay="0.1s"
          />
          <FeatureCard
            icon={Users}
            title="Participant Insights"
            description="Per-person coaching & personality"
            delay="0.2s"
          />
          <FeatureCard
            icon={Zap}
            title="Real-time Results"
            description="See insights as they're generated"
            delay="0.3s"
          />
        </div>

        {/* CTA button */}
        <div className="relative z-10">
          <button
            onClick={onNewAnalysis}
            className="inline-flex items-center gap-2.5 px-8 py-3.5 text-base font-semibold text-white rounded-xl bg-gradient-to-r from-indigo-500 to-purple-500 hover:from-indigo-400 hover:to-purple-400 shadow-lg shadow-purple-500/25 transition-all duration-300 hover:shadow-purple-500/40 hover:scale-105"
          >
            Analyze Your First Meeting
            <ArrowRight className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
}

/* ── Main Component ────────────────────────────────────────────────── */
export default function VideoLibrary({ videos, onSelect, onDelete, onNewAnalysis }: Props) {
  if (videos.length === 0) {
    return <EmptyState onNewAnalysis={onNewAnalysis} />;
  }

  return (
    <div>
      <h2 className="text-xl font-semibold gradient-text mb-4">Recordings</h2>

      {/* Stats Bar */}
      {(() => {
        const completedVideos = videos.filter(v => v.status === 'complete');
        const avgScore = completedVideos.length > 0
          ? Math.round(completedVideos.reduce((sum, v) => sum + v.overall_score, 0) / completedVideos.length)
          : 0;
        return completedVideos.length > 0 ? (
          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className="glass-card p-4 text-center">
              <p className="text-2xl font-bold gradient-text">{videos.length}</p>
              <p className="text-xs text-slate-500 mt-1">Meetings Analyzed</p>
            </div>
            <div className="glass-card p-4 text-center">
              <p className="text-2xl font-bold gradient-text">{avgScore}%</p>
              <p className="text-xs text-slate-500 mt-1">Avg Engagement</p>
            </div>
            <div className="glass-card p-4 text-center">
              <p className="text-2xl font-bold gradient-text">{completedVideos.length * 3}+</p>
              <p className="text-xs text-slate-500 mt-1">Total Participants</p>
            </div>
          </div>
        ) : null;
      })()}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {videos.map((video) => (
          <div
            key={video.id}
            onClick={() => onSelect(video.id)}
            className="glass-card glass-card-hover overflow-hidden transition-all duration-300 cursor-pointer group relative"
          >
            {/* Gradient top-border on hover */}
            <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-indigo-500 via-purple-500 to-cyan-500 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />

            {/* Thumbnail */}
            <div className="relative w-full aspect-video bg-gradient-to-br from-indigo-500/10 via-purple-500/10 to-cyan-500/10 flex items-center justify-center border-b border-white/[0.06] overflow-hidden">
              <img
                src={getThumbnailUrl(video.id)}
                alt=""
                className="absolute inset-0 w-full h-full object-cover"
                onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
              />
              <div className="w-14 h-14 rounded-2xl bg-white/[0.06] flex items-center justify-center">
                <FileVideo className="w-7 h-7 text-purple-400/60" />
              </div>

              {/* Score badge overlay (top-right) for completed videos */}
              {video.status === 'complete' && (
                <div className="absolute top-3 right-3">
                  <ScoreBadge score={video.overall_score} />
                </div>
              )}
            </div>

            {/* Card body */}
            <div className="p-5">
              {/* Video name */}
              <p className="text-sm font-semibold text-slate-200 truncate mb-2">
                {video.name}
              </p>

              {/* Meta row: date + duration */}
              <div className="flex items-center gap-3 text-xs text-text-muted mb-3">
                <span className="flex items-center gap-1">
                  <Calendar className="w-3 h-3" />
                  {new Date(video.created_at).toLocaleDateString()}
                </span>
                <span className="flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {video.duration > 0 ? formatTime(video.duration) : '--:--'}
                </span>
              </div>

              {/* Mini engagement sparkline */}
              {video.status === 'complete' && video.overall_score > 0 && (
                <div className="mb-3">
                  <div className="flex items-center gap-2 mb-1">
                    <TrendingUp className="w-3 h-3 text-purple-400" />
                    <span className="text-[10px] text-slate-500">Engagement</span>
                  </div>
                  <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-purple-500 transition-all duration-700"
                      style={{ width: `${video.overall_score}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Status + delete row */}
              <div className="flex items-center justify-between">
                <StatusBadge status={video.status} />

                {onDelete && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      if (confirm('Delete this recording?')) {
                        onDelete(video.id);
                      }
                    }}
                    className="p-1.5 rounded-lg hover:bg-red-500/10 text-slate-600 hover:text-red-400 transition-colors duration-200"
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
