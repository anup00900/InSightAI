import { useState, useCallback, useRef } from 'react';
import { Upload, X } from 'lucide-react';
import GlassCard from '../ui/GlassCard';
import ProgressRing from '../ui/ProgressRing';
import { uploadVideo } from '../../lib/api';

interface UploadStepProps {
  onUploadComplete: (videoId: string) => void;
}

export default function UploadStep({ onUploadComplete }: UploadStepProps) {
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(async (file: File) => {
    setSelectedFile(file);
    setUploading(true);
    setProgress(0);
    setError(null);
    try {
      const { id } = await uploadVideo(file, (pct) => setProgress(pct));
      onUploadComplete(id);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Upload failed');
      setUploading(false);
    }
  }, [onUploadComplete]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  return (
    <div className="max-w-2xl mx-auto">
      <GlassCard className="overflow-hidden">
        <div className="text-center mb-6">
          <h2 className="text-2xl font-bold gradient-text">Upload Meeting Recording</h2>
          <p className="text-text-muted text-sm mt-2">Drag and drop your video file or click to browse</p>
        </div>

        {uploading ? (
          <div className="flex flex-col items-center py-12 gap-4">
            <ProgressRing progress={progress} size={140} label="Uploading..." />
            <p className="text-text-secondary text-sm">{selectedFile?.name}</p>
          </div>
        ) : (
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
            className={`relative border-2 border-dashed rounded-2xl p-16 text-center cursor-pointer transition-all duration-300 ${
              dragOver
                ? 'border-accent bg-accent/5 shadow-glow-purple'
                : 'border-white/10 hover:border-accent/50 hover:bg-white/[0.02]'
            }`}
          >
            <div className={`w-16 h-16 mx-auto rounded-2xl flex items-center justify-center mb-4 transition-all ${
              dragOver ? 'bg-gradient-primary shadow-glow-purple' : 'bg-white/5'
            }`}>
              <Upload className={`w-8 h-8 ${dragOver ? 'text-white' : 'text-text-muted'}`} />
            </div>
            <p className="text-text-primary font-medium">
              {dragOver ? 'Drop to upload' : 'Click or drag video file here'}
            </p>
            <p className="text-text-muted text-xs mt-2">MP4, AVI, MOV, MKV, WebM — up to 2GB</p>
            <input
              ref={inputRef}
              type="file"
              accept="video/*,audio/*"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
          </div>
        )}

        {error && (
          <div className="mt-4 p-3 rounded-xl bg-danger/10 border border-danger/20 text-red-400 text-sm flex items-center gap-2">
            <X className="w-4 h-4" />
            {error}
          </div>
        )}
      </GlassCard>
    </div>
  );
}
