import { useRef, useState } from 'react';
import { Upload, FileVideo, Loader2, AlertCircle, Link2, Video } from 'lucide-react';
import MeetingBotTracker from './MeetingBotTracker';

interface Props {
  onUpload: (file: File) => void;
  onImportUrl?: (url: string) => void;
  uploading: boolean;
  uploadProgress?: number;
  error?: string | null;
}

export default function VideoUpload({ onUpload, onImportUrl, uploading, uploadProgress = 0, error }: Props) {
  const [dragOver, setDragOver] = useState(false);
  const [urlInput, setUrlInput] = useState('');
  const [importing, setImporting] = useState(false);
  const [meetingUrl, setMeetingUrl] = useState('');
  const [joiningMeeting, setJoiningMeeting] = useState(false);
  const [meetingStatus, setMeetingStatus] = useState<string | null>(null);
  const [activeBotId, setActiveBotId] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) onUpload(file);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onUpload(file);
  };

  const handleImportUrl = async () => {
    const url = urlInput.trim();
    if (!url || !onImportUrl) return;
    setImporting(true);
    try {
      await onImportUrl(url);
      setUrlInput('');
    } finally {
      setImporting(false);
    }
  };

  const handleJoinMeeting = async () => {
    const url = meetingUrl.trim();
    if (!url) return;
    setJoiningMeeting(true);
    setMeetingStatus(null);
    try {
      const res = await fetch('/api/meetings/join', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ meeting_url: url }),
      });
      const data = await res.json();
      if (!res.ok) {
        setMeetingStatus(data.detail || 'Failed to join meeting');
      } else {
        setMeetingStatus(`Bot joined! ID: ${data.bot_id} — Status: ${data.status}`);
        setActiveBotId(data.bot_id);
        setMeetingUrl('');
      }
    } catch {
      setMeetingStatus('Failed to connect. Check if RECALL_API_KEY is configured.');
    } finally {
      setJoiningMeeting(false);
    }
  };

  const isProcessing = uploading || importing;

  return (
    <div className="space-y-4">
      {/* File Upload Area */}
      <div
        className={`
          relative border-2 border-dashed rounded-2xl p-12 text-center transition-all cursor-pointer
          ${dragOver
            ? 'border-accent bg-accent/5 scale-[1.01]'
            : 'border-border hover:border-accent/50 hover:bg-bg-card/50'
          }
        `}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => !isProcessing && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/*,audio/*"
          className="hidden"
          onChange={handleChange}
          disabled={isProcessing}
        />
        {uploading ? (
          <div className="flex flex-col items-center gap-4">
            <Loader2 className="w-12 h-12 text-accent animate-spin" />
            <p className="text-text-secondary font-medium">
              Uploading your recording... {uploadProgress > 0 ? `${uploadProgress}%` : ''}
            </p>
            {uploadProgress > 0 && (
              <div className="w-64 h-2 bg-border rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-accent to-blue-400 transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            )}
            <p className="text-text-muted text-sm">This will open the live analysis view automatically</p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-4">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-accent/20 to-blue-400/20 flex items-center justify-center">
              {dragOver ? (
                <FileVideo className="w-8 h-8 text-accent-light" />
              ) : (
                <Upload className="w-8 h-8 text-accent" />
              )}
            </div>
            <div>
              <p className="text-lg font-medium text-text-primary">
                Upload a meeting recording
              </p>
              <p className="text-sm text-text-muted mt-1">
                MP4, AVI, MOV, WebM, MP3, WAV, M4A — press play to start live analysis
              </p>
            </div>
            <button className="px-6 py-2.5 bg-gradient-to-r from-accent to-blue-400 hover:from-accent-light hover:to-blue-300 text-white rounded-lg font-medium text-sm transition-all shadow-lg shadow-accent-glow">
              Browse Files
            </button>
          </div>
        )}
      </div>

      {/* URL Import */}
      {onImportUrl && (
        <div className="flex items-center gap-3">
          <div className="flex-1 relative">
            <Link2 className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
            <input
              type="text"
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleImportUrl()}
              placeholder="Paste meeting recording URL (SharePoint, Teams, YouTube...)"
              disabled={isProcessing}
              className="w-full pl-10 pr-4 py-3 glass-card text-text-primary placeholder-text-muted text-sm focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent transition-colors disabled:opacity-50"
            />
          </div>
          <button
            onClick={handleImportUrl}
            disabled={!urlInput.trim() || isProcessing}
            className="px-5 py-3 bg-gradient-to-r from-accent to-blue-400 hover:from-accent-light hover:to-blue-300 text-white rounded-xl font-medium text-sm transition-all shadow-lg shadow-accent-glow disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 whitespace-nowrap"
          >
            {importing ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Importing...
              </>
            ) : (
              'Import URL'
            )}
          </button>
        </div>
      )}

      {/* Join Live Meeting (Recall.ai) */}
      <div className="flex items-center gap-3">
        <div className="flex-1 relative">
          <Video className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
          <input
            type="text"
            value={meetingUrl}
            onChange={(e) => setMeetingUrl(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleJoinMeeting()}
            placeholder="Paste Zoom/Teams/Meet link to join live with AI bot..."
            disabled={joiningMeeting}
            className="w-full pl-10 pr-4 py-3 glass-card text-text-primary placeholder-text-muted text-sm focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition-colors disabled:opacity-50"
          />
        </div>
        <button
          onClick={handleJoinMeeting}
          disabled={!meetingUrl.trim() || joiningMeeting}
          className="px-5 py-3 bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 text-white rounded-xl font-medium text-sm transition-all shadow-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 whitespace-nowrap"
        >
          {joiningMeeting ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Joining...
            </>
          ) : (
            <>
              <Video className="w-4 h-4" />
              Join Meeting
            </>
          )}
        </button>
      </div>
      {meetingStatus && !activeBotId && (
        <div className={`flex items-center gap-2 text-sm ${meetingStatus.startsWith('Bot joined') ? 'text-emerald-400' : 'text-amber-400'}`}>
          <AlertCircle className="w-4 h-4" />
          {meetingStatus}
        </div>
      )}
      {activeBotId && (
        <MeetingBotTracker botId={activeBotId} onDismiss={() => setActiveBotId(null)} />
      )}

      {error && (
        <div className="flex items-center gap-2 text-danger text-sm">
          <AlertCircle className="w-4 h-4" />
          {error}
        </div>
      )}
    </div>
  );
}
