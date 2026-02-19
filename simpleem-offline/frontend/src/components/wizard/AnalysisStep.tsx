import { useState, useEffect, useCallback, useRef } from 'react';
import { Users, MessageSquare, Brain, BarChart3, Shield, Check, ChevronDown, Volume2, AlertCircle } from 'lucide-react';
import { clsx } from 'clsx';
import GlassCard from '../ui/GlassCard';
import ProgressRing from '../ui/ProgressRing';
import GlowBadge from '../ui/GlowBadge';
import { usePolling } from '../../hooks/usePolling';
import { getVideoStatus, triggerAnalysis, type VideoStatus } from '../../lib/api';

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

type StageKey = 'participants' | 'frames' | 'audio' | 'text' | 'profiles' | 'summary';
type StageStatus = 'waiting' | 'active' | 'done';

interface StageState {
  status: StageStatus;
  detail: string;
  results: string[];
}

interface StageConfig {
  key: StageKey;
  label: string;
  icon: typeof Users;
  /** Progress value at which this stage begins */
  startsAt: number;
  /** Progress value at which this stage is considered done */
  doneAt: number;
}

interface AnalysisStepProps {
  videoId: string;
  onComplete: () => void;
}

/* ------------------------------------------------------------------ */
/*  Stage definitions                                                  */
/* ------------------------------------------------------------------ */

const STAGE_CONFIGS: StageConfig[] = [
  { key: 'participants', label: 'Detecting Participants',  icon: Users,          startsAt: 0,  doneAt: 10 },
  { key: 'frames',       label: 'Analyzing Video Frames',  icon: BarChart3,      startsAt: 10, doneAt: 45 },
  { key: 'audio',        label: 'Processing Audio',        icon: Volume2,        startsAt: 45, doneAt: 52 },
  { key: 'text',         label: 'Analyzing Text Signals',  icon: MessageSquare,  startsAt: 52, doneAt: 60 },
  { key: 'profiles',     label: 'Building Profiles',       icon: Brain,          startsAt: 60, doneAt: 82 },
  { key: 'summary',      label: 'Generating Summary',      icon: Shield,         startsAt: 82, doneAt: 95 },
];

const MAX_RESULTS_PER_STAGE = 5;

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

/** Derive stage statuses from a progress value. */
function deriveStageStatuses(progress: number): Record<StageKey, StageStatus> {
  const result = {} as Record<StageKey, StageStatus>;
  for (const cfg of STAGE_CONFIGS) {
    if (progress >= cfg.doneAt) {
      result[cfg.key] = 'done';
    } else if (progress >= cfg.startsAt) {
      result[cfg.key] = 'active';
    } else {
      result[cfg.key] = 'waiting';
    }
  }
  // If progress >= 95, mark everything done
  if (progress >= 95) {
    for (const cfg of STAGE_CONFIGS) {
      result[cfg.key] = 'done';
    }
  }
  return result;
}

/** Determine which stage a detail string most likely belongs to. */
function guessStageFromDetail(detail: string): StageKey | null {
  const lower = detail.toLowerCase();
  if (lower.includes('participant') || lower.includes('detect') || lower.includes('face')) return 'participants';
  if (lower.includes('frame') || lower.includes('video') || lower.includes('visual') || lower.includes('emotion')) return 'frames';
  if (lower.includes('audio') || lower.includes('voice') || lower.includes('sound') || lower.includes('pitch')) return 'audio';
  if (lower.includes('text') || lower.includes('transcript') || lower.includes('word') || lower.includes('verbal') || lower.includes('sentiment')) return 'text';
  if (lower.includes('profile') || lower.includes('correlat') || lower.includes('personality') || lower.includes('engagement')) return 'profiles';
  if (lower.includes('summary') || lower.includes('saving') || lower.includes('finaliz') || lower.includes('report') || lower.includes('coaching') || lower.includes('notes')) return 'summary';
  return null;
}

/** Get the first currently-active stage key from statuses derived by progress. */
function firstActiveStage(statuses: Record<StageKey, StageStatus>): StageKey | null {
  for (const cfg of STAGE_CONFIGS) {
    if (statuses[cfg.key] === 'active') return cfg.key;
  }
  return null;
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function AnalysisStep({ videoId, onComplete }: AnalysisStepProps) {
  // Whether analysis has been triggered
  const [started, setStarted] = useState(false);
  // Top-level error
  const [error, setError] = useState<string | null>(null);
  // Monotonic max progress
  const [maxProgress, setMaxProgress] = useState(0);
  // Current detail text from SSE (or polling)
  const [currentDetail, setCurrentDetail] = useState('Starting analysis...');
  // Per-stage state
  const [stages, setStages] = useState<Record<StageKey, StageState>>(() => {
    const init = {} as Record<StageKey, StageState>;
    for (const cfg of STAGE_CONFIGS) {
      init[cfg.key] = { status: 'waiting', detail: '', results: [] };
    }
    return init;
  });
  // Which done-stage cards the user has expanded
  const [expandedDone, setExpandedDone] = useState<Set<StageKey>>(new Set());
  // Whether SSE is connected
  const sseConnected = useRef(false);
  // Ref for completion timeout so we can clear on unmount
  const completeTimerRef = useRef<number | null>(null);
  // Track if completion has been fired
  const completedRef = useRef(false);
  // Ref for EventSource
  const eventSourceRef = useRef<EventSource | null>(null);
  // Stable ref for onComplete callback (avoids SSE reconnection on parent re-render)
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;
  // Ref for maxProgress to avoid stale closures
  const maxProgressRef = useRef(0);

  /* ---------------------------------------------------------------- */
  /*  Trigger analysis on mount                                        */
  /* ---------------------------------------------------------------- */

  useEffect(() => {
    let cancelled = false;
    const start = async () => {
      try {
        await triggerAnalysis(videoId);
        if (!cancelled) setStarted(true);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Failed to start analysis');
      }
    };
    start();
    return () => { cancelled = true; };
  }, [videoId]);

  /* ---------------------------------------------------------------- */
  /*  Update stage state from progress + detail                        */
  /* ---------------------------------------------------------------- */

  const updateFromProgress = useCallback((progress: number, detail: string, activeStages?: string[]) => {
    setMaxProgress(prev => {
      const next = Math.max(prev, progress);
      maxProgressRef.current = next;
      return next;
    });

    const effectiveProgress = Math.max(progress, maxProgressRef.current);
    const thresholdStatuses = deriveStageStatuses(effectiveProgress);

    // If backend provides active_stages, use it to override threshold-based derivation
    const activeSet = activeStages ? new Set(activeStages) : null;

    // Determine which stage the detail belongs to
    let targetStage = guessStageFromDetail(detail);
    if (!targetStage) {
      targetStage = firstActiveStage(thresholdStatuses);
    }

    setStages(prev => {
      const next = { ...prev };
      for (const cfg of STAGE_CONFIGS) {
        const old = prev[cfg.key];

        // Derive status: prefer backend active_stages over threshold-based
        let newStatus: StageStatus;
        if (activeSet) {
          if (activeSet.has(cfg.key)) {
            // Backend says this stage is still running
            newStatus = 'active';
          } else if (old.status === 'active' || old.status === 'done') {
            // Was active/done before and backend no longer lists it — it's done
            newStatus = 'done';
          } else {
            // Not in active set and wasn't previously active — use threshold
            newStatus = thresholdStatuses[cfg.key];
          }
        } else {
          // No active_stages from backend (polling fallback) — use threshold
          newStatus = thresholdStatuses[cfg.key];
        }

        // Never go backward in status
        if (old.status === 'done' && newStatus !== 'done') {
          next[cfg.key] = old;
          continue;
        }

        const isTarget = cfg.key === targetStage;
        const newDetail = isTarget && detail ? detail : old.detail;

        // Add detail as a result entry for the target stage
        let newResults = [...old.results];
        if (isTarget && detail && detail !== old.detail) {
          newResults = [...newResults, detail];
          if (newResults.length > MAX_RESULTS_PER_STAGE) {
            newResults = newResults.slice(newResults.length - MAX_RESULTS_PER_STAGE);
          }
        }

        next[cfg.key] = {
          status: newStatus,
          detail: newDetail,
          results: newResults,
        };
      }
      return next;
    });

    if (detail) {
      setCurrentDetail(detail);
    }
  }, []);

  /* ---------------------------------------------------------------- */
  /*  SSE connection                                                    */
  /* ---------------------------------------------------------------- */

  useEffect(() => {
    if (!started) return;

    const es = new EventSource(`/api/videos/${videoId}/analysis-stream`);
    eventSourceRef.current = es;
    sseConnected.current = true;

    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === 'progress') {
          const progress = typeof data.progress === 'number' ? data.progress : 0;
          const detail = typeof data.detail === 'string' ? data.detail : '';
          const activeStages = Array.isArray(data.active_stages) ? data.active_stages : undefined;
          updateFromProgress(progress, detail, activeStages);
        }

        if (data.type === 'complete') {
          if (!completedRef.current) {
            completedRef.current = true;
            // Mark all stages as done
            setStages(prev => {
              const next = { ...prev };
              for (const cfg of STAGE_CONFIGS) {
                next[cfg.key] = {
                  ...prev[cfg.key],
                  status: 'done',
                };
              }
              return next;
            });
            setMaxProgress(100);
            setCurrentDetail('Analysis complete!');
            completeTimerRef.current = window.setTimeout(() => onCompleteRef.current(), 800);
          }
          es.close();
        }

        if (data.type === 'error') {
          setError(data.detail || 'Analysis failed');
          es.close();
        }

        // heartbeat: ignore
      } catch {
        // Malformed SSE data — ignore
      }
    };

    es.onerror = () => {
      sseConnected.current = false;
      // Don't setError — let polling fallback handle it
    };

    return () => {
      es.close();
      eventSourceRef.current = null;
      sseConnected.current = false;
    };
  }, [started, videoId, updateFromProgress]);

  /* ---------------------------------------------------------------- */
  /*  Polling fallback (3s interval)                                    */
  /* ---------------------------------------------------------------- */

  const statusFetcher = useCallback(() => getVideoStatus(videoId), [videoId]);
  const { data: polledStatus } = usePolling<VideoStatus>(statusFetcher, 3000, started);

  useEffect(() => {
    if (!polledStatus) return;

    // Always update from polling data to keep in sync
    const progress = polledStatus.progress ?? 0;
    const detail = polledStatus.detail || '';
    updateFromProgress(progress, detail);

    // Handle completion/error from polling
    if (polledStatus.step === 'complete' && !completedRef.current) {
      completedRef.current = true;
      setStages(prev => {
        const next = { ...prev };
        for (const cfg of STAGE_CONFIGS) {
          next[cfg.key] = { ...prev[cfg.key], status: 'done' };
        }
        return next;
      });
      setMaxProgress(100);
      setCurrentDetail('Analysis complete!');
      completeTimerRef.current = window.setTimeout(() => onCompleteRef.current(), 800);
    }

    if (polledStatus.step === 'error') {
      setError(polledStatus.detail || 'Analysis failed');
    }
  }, [polledStatus, updateFromProgress]);

  /* ---------------------------------------------------------------- */
  /*  Cleanup                                                          */
  /* ---------------------------------------------------------------- */

  useEffect(() => {
    return () => {
      if (completeTimerRef.current !== null) {
        clearTimeout(completeTimerRef.current);
      }
    };
  }, []);

  /* ---------------------------------------------------------------- */
  /*  Toggle done-stage expansion                                      */
  /* ---------------------------------------------------------------- */

  const toggleDoneExpanded = (key: StageKey) => {
    setExpandedDone(prev => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  /* ---------------------------------------------------------------- */
  /*  Effective progress (monotonic)                                    */
  /* ---------------------------------------------------------------- */

  const effectiveProgress = maxProgress;

  /* ---------------------------------------------------------------- */
  /*  Render                                                           */
  /* ---------------------------------------------------------------- */

  return (
    <div className="max-w-3xl mx-auto">
      <GlassCard padding={true}>
        {/* Header */}
        <div className="text-center mb-6">
          <h2 className="text-2xl font-bold gradient-text">Analyzing Meeting</h2>
          {!error && (
            <p className="text-xs text-text-muted mt-1">
              {sseConnected.current ? 'Live streaming updates' : 'Polling for updates'}
            </p>
          )}
        </div>

        {error ? (
          /* ---- Error state ---- */
          <div className="p-5 rounded-xl bg-danger/10 border border-danger/20 text-center">
            <AlertCircle className="w-8 h-8 text-red-400 mx-auto mb-3" />
            <p className="text-red-400 font-medium mb-1">Analysis Failed</p>
            <p className="text-red-400/70 text-sm">{error}</p>
          </div>
        ) : (
          <div className="space-y-5">
            {/* ---- Top progress section ---- */}
            <div className="flex items-center gap-6">
              <ProgressRing
                progress={effectiveProgress}
                size={110}
                strokeWidth={8}
                showText={true}
                label="Progress"
              />
              <div className="flex-1 min-w-0">
                <p className="text-sm text-text-primary font-medium truncate">
                  {currentDetail}
                </p>
                <div className="mt-3 h-2 rounded-full bg-white/5 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-sky-500 via-indigo-500 to-purple-500 transition-all duration-700 ease-out"
                    style={{ width: `${effectiveProgress}%` }}
                  />
                </div>
                <p className="text-[10px] text-text-muted mt-1.5">
                  {effectiveProgress < 100
                    ? `${Math.round(effectiveProgress)}% complete`
                    : 'Finalizing...'}
                </p>
              </div>
            </div>

            {/* ---- Stage cards (single column) ---- */}
            <div className="space-y-2.5">
              {STAGE_CONFIGS.map((cfg) => {
                const stage = stages[cfg.key];
                const Icon = cfg.icon;
                const isActive = stage.status === 'active';
                const isDone = stage.status === 'done';
                const isWaiting = stage.status === 'waiting';
                const isExpanded = isActive || (isDone && expandedDone.has(cfg.key));

                return (
                  <div
                    key={cfg.key}
                    onClick={isDone ? () => toggleDoneExpanded(cfg.key) : undefined}
                    className={clsx(
                      'rounded-2xl transition-all duration-500 overflow-hidden',
                      // Waiting: dimmed, compact
                      isWaiting && 'glass-depth-1 opacity-40',
                      // Active: blue glow, expanded
                      isActive && 'glass-card-expanded animate-glow-pulse',
                      // Done: subtle, clickable
                      isDone && 'glass-depth-1 cursor-pointer hover:bg-white/[0.04]',
                    )}
                  >
                    {/* Card header row */}
                    <div
                      className={clsx(
                        'flex items-center gap-3 px-4',
                        isActive ? 'py-3.5' : 'py-2.5',
                      )}
                    >
                      {/* Icon */}
                      <div
                        className={clsx(
                          'w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0 transition-all duration-300',
                          isWaiting && 'bg-white/5',
                          isActive && 'bg-sky-500/20',
                          isDone && 'bg-emerald-500/20',
                        )}
                      >
                        {isDone ? (
                          <Check className="w-4 h-4 text-emerald-400" />
                        ) : (
                          <Icon
                            className={clsx(
                              'w-4 h-4 transition-colors duration-300',
                              isWaiting && 'text-white/20',
                              isActive && 'text-sky-400',
                            )}
                          />
                        )}
                      </div>

                      {/* Label */}
                      <span
                        className={clsx(
                          'text-sm font-medium flex-1 transition-colors duration-300',
                          isWaiting && 'text-text-muted/50',
                          isActive && 'text-text-primary',
                          isDone && 'text-emerald-400',
                        )}
                      >
                        {cfg.label}
                      </span>

                      {/* Badge / chevron */}
                      {isActive && (
                        <GlowBadge variant="blue" size="sm">In Progress</GlowBadge>
                      )}
                      {isDone && (
                        <div className="flex items-center gap-2">
                          <GlowBadge variant="success" size="sm">Done</GlowBadge>
                          {stage.results.length > 0 && (
                            <ChevronDown
                              className={clsx(
                                'w-3.5 h-3.5 text-text-muted transition-transform duration-300',
                                expandedDone.has(cfg.key) && 'rotate-180',
                              )}
                            />
                          )}
                        </div>
                      )}
                    </div>

                    {/* Expanded content: detail + streaming results */}
                    {isExpanded && (stage.detail || stage.results.length > 0) && (
                      <div className="px-4 pb-3.5 animate-fade-in">
                        <div className="border-t border-white/[0.06] pt-3 ml-11">
                          {/* Current detail */}
                          {isActive && stage.detail && (
                            <p className="text-xs text-text-secondary mb-2 truncate">
                              {stage.detail}
                            </p>
                          )}

                          {/* Results list */}
                          {stage.results.length > 0 && (
                            <div className="space-y-1">
                              {stage.results.map((result, i) => (
                                <div
                                  key={i}
                                  className={clsx(
                                    'flex items-start gap-2 text-[11px]',
                                    // Latest result is brighter
                                    i === stage.results.length - 1
                                      ? 'text-text-secondary'
                                      : 'text-text-muted/60',
                                    'animate-fade-in',
                                  )}
                                  style={{ animationDelay: `${i * 50}ms` }}
                                >
                                  <span className="text-text-muted/40 select-none mt-px shrink-0">&#9656;</span>
                                  <span className="truncate">{result}</span>
                                </div>
                              ))}
                            </div>
                          )}

                          {/* Active pulsing indicator */}
                          {isActive && (
                            <div className="flex items-center gap-1.5 mt-2.5">
                              <span className="inline-block w-1.5 h-1.5 rounded-full bg-sky-400 animate-pulse" />
                              <span className="inline-block w-1.5 h-1.5 rounded-full bg-sky-400/60 animate-pulse" style={{ animationDelay: '150ms' }} />
                              <span className="inline-block w-1.5 h-1.5 rounded-full bg-sky-400/30 animate-pulse" style={{ animationDelay: '300ms' }} />
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </GlassCard>
    </div>
  );
}
