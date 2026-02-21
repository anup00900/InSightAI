import { useEffect, useState } from 'react';
import { Video, X, Download, Loader2, PhoneOff } from 'lucide-react';

interface Props {
  botId: string;
  onDismiss: () => void;
}

type BotStatus = 'joining' | 'in_call' | 'recording' | 'done' | 'error';

interface BotStatusResponse {
  status: BotStatus;
  error?: string;
}

const STATUS_CONFIG = {
  joining: { label: 'Joining', color: 'bg-blue-500/10 text-blue-500 border-blue-500/20' },
  in_call: { label: 'In Call', color: 'bg-green-500/10 text-green-500 border-green-500/20' },
  recording: { label: 'Recording', color: 'bg-red-500/10 text-red-500 border-red-500/20' },
  done: { label: 'Done', color: 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20' },
  error: { label: 'Error', color: 'bg-red-500/10 text-red-500 border-red-500/20' },
};

export default function MeetingBotTracker({ botId, onDismiss }: Props) {
  const [status, setStatus] = useState<BotStatus>('joining');
  const [error, setError] = useState<string | null>(null);
  const [isLeaving, setIsLeaving] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);

  useEffect(() => {
    let intervalId: ReturnType<typeof setInterval>;

    const fetchStatus = async () => {
      try {
        const response = await fetch(`/api/meetings/${botId}/status`);
        if (!response.ok) {
          throw new Error('Failed to fetch bot status');
        }
        const data: BotStatusResponse = await response.json();
        setStatus(data.status);
        if (data.error) {
          setError(data.error);
        }

        // Stop polling when bot is done or errored
        if (data.status === 'done' || data.status === 'error') {
          if (intervalId) {
            clearInterval(intervalId);
          }
        }
      } catch (err) {
        console.error('Error fetching bot status:', err);
        setStatus('error');
        setError(err instanceof Error ? err.message : 'Unknown error');
        if (intervalId) {
          clearInterval(intervalId);
        }
      }
    };

    // Initial fetch
    fetchStatus();

    // Start polling every 5 seconds
    intervalId = setInterval(fetchStatus, 5000);

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [botId]);

  const handleLeave = async () => {
    setIsLeaving(true);
    try {
      const response = await fetch(`/api/meetings/${botId}/leave`, {
        method: 'POST',
      });
      if (!response.ok) {
        throw new Error('Failed to leave meeting');
      }
      // Status will be updated by the polling mechanism
    } catch (err) {
      console.error('Error leaving meeting:', err);
      setError(err instanceof Error ? err.message : 'Failed to leave meeting');
    } finally {
      setIsLeaving(false);
    }
  };

  const handleDownload = async () => {
    setIsDownloading(true);
    try {
      const response = await fetch(`/api/meetings/${botId}/recording`);
      if (!response.ok) {
        throw new Error('Failed to download recording');
      }

      const data = await response.json();
      const recordingUrl = data.recording_url;
      if (!recordingUrl) {
        throw new Error('Recording URL not available');
      }

      // Open the actual recording URL in a new tab for download
      window.open(recordingUrl, '_blank');
    } catch (err) {
      console.error('Error downloading recording:', err);
      setError(err instanceof Error ? err.message : 'Failed to download recording');
    } finally {
      setIsDownloading(false);
    }
  };

  const statusConfig = STATUS_CONFIG[status];

  return (
    <div className="glass-card p-4">
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-3 flex-1">
          <div className="p-2 bg-primary/10 rounded-lg">
            <Video className="w-5 h-5 text-primary" />
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <h3 className="text-sm font-medium text-text-primary">Meeting Bot</h3>
              <span
                className={`px-2 py-0.5 text-xs font-medium rounded-full border ${statusConfig.color}`}
              >
                {statusConfig.label}
              </span>
            </div>
            <p className="text-xs text-text-secondary">Bot ID: {botId}</p>
            {error && (
              <p className="text-xs text-red-500 mt-1">{error}</p>
            )}
          </div>
        </div>
        <button
          onClick={onDismiss}
          className="p-1 hover:bg-bg-hover rounded-lg transition-colors"
          aria-label="Dismiss"
        >
          <X className="w-4 h-4 text-text-secondary" />
        </button>
      </div>

      <div className="flex items-center gap-2 mt-4">
        {status !== 'done' && status !== 'error' && (
          <button
            onClick={handleLeave}
            disabled={isLeaving}
            className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-red-500 bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLeaving ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <PhoneOff className="w-4 h-4" />
            )}
            Leave Meeting
          </button>
        )}

        {status === 'done' && (
          <button
            onClick={handleDownload}
            disabled={isDownloading}
            className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-emerald-500 bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/20 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isDownloading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Download className="w-4 h-4" />
            )}
            Download Recording
          </button>
        )}
      </div>
    </div>
  );
}
