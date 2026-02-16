import { useRef, useState } from 'react';
import { Upload, FileVideo, Loader2, AlertCircle, Link2 } from 'lucide-react';

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
            <p className="text-text-muted text-sm">Your file will be queued for batch analysis</p>
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
                MP4, AVI, MOV, WebM, MP3, WAV, M4A — for batch analysis of your meeting
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
              className="w-full pl-10 pr-4 py-3 bg-bg-card border border-border rounded-xl text-text-primary placeholder-text-muted text-sm focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent transition-colors disabled:opacity-50"
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

      {error && (
        <div className="flex items-center gap-2 text-danger text-sm">
          <AlertCircle className="w-4 h-4" />
          {error}
        </div>
      )}
    </div>
  );
}
