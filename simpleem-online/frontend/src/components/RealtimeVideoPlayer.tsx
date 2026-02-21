import { useRef, useEffect } from 'react';
import { Wifi, WifiOff, Radio } from 'lucide-react';

interface Props {
  videoSrc: string;
  attachVideo: (el: HTMLVideoElement) => (() => void);
  statusMessage: string;
  isConnected: boolean;
  isAnalyzing: boolean;
}

export default function RealtimeVideoPlayer({
  videoSrc,
  attachVideo,
  statusMessage,
  isConnected,
  isAnalyzing,
}: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (!videoRef.current) return;
    const el = videoRef.current;
    const cleanup = attachVideo(el);
    return () => {
      cleanup();
      // Release media decode buffers on unmount
      el.pause();
      el.removeAttribute('src');
      el.load();
    };
  }, [attachVideo]);

  return (
    <div className="glass-card overflow-hidden relative group">
      <video
        ref={videoRef}
        controls
        className="w-full aspect-video bg-black"
        src={videoSrc}
      />

      {/* Live analysis status bar */}
      <div className="absolute bottom-12 left-0 right-0 flex items-center justify-between px-4 py-1.5 bg-black/60 backdrop-blur-sm">
        <div className="flex items-center gap-2">
          {isConnected ? (
            <Wifi className="w-3 h-3 text-success" />
          ) : (
            <WifiOff className="w-3 h-3 text-danger" />
          )}
          {isAnalyzing && (
            <Radio className="w-3 h-3 text-accent animate-pulse" />
          )}
          <span className="text-xs text-text-secondary">{statusMessage}</span>
        </div>
        <div className="flex items-center gap-1">
          <span
            className={`w-2 h-2 rounded-full ${isAnalyzing ? 'bg-accent animate-pulse' : isConnected ? 'bg-success' : 'bg-danger'}`}
          />
          <span className="text-[10px] text-text-muted">
            {isAnalyzing ? 'ANALYZING' : isConnected ? 'READY' : 'OFFLINE'}
          </span>
        </div>
      </div>
    </div>
  );
}
