import { useState, useEffect, useCallback } from 'react';
import { type Video, type AnalysisResults, listVideos, getVideo, getResults, uploadVideo, importVideoUrl, deleteVideo } from './lib/api';
import VideoUpload from './components/VideoUpload';
import VideoLibrary from './components/VideoLibrary';
import AnalysisDashboard from './components/AnalysisDashboard';
import CrossMeetingAnalytics from './components/CrossMeetingAnalytics';
import { Activity, ArrowLeft } from 'lucide-react';

type View = 'library' | 'dashboard';
type DashboardMode = 'realtime' | 'review' | 'url';

function App() {
  const [view, setView] = useState<View>('library');
  const [videos, setVideos] = useState<Video[]>([]);
  const [selectedVideoId, setSelectedVideoId] = useState<string | null>(null);
  const [selectedVideo, setSelectedVideo] = useState<Video | null>(null);
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [dashboardMode, setDashboardMode] = useState<DashboardMode>('review');

  const fetchVideos = useCallback(async () => {
    try {
      const data = await listVideos();
      setVideos(data);
    } catch (err) {
      console.error('Failed to fetch videos:', err);
    }
  }, []);

  useEffect(() => {
    fetchVideos();
  }, [fetchVideos]);

  const handleUpload = async (file: File) => {
    setUploading(true);
    setUploadProgress(0);
    setUploadError(null);
    try {
      const { id } = await uploadVideo(file, (pct) => setUploadProgress(pct));
      await fetchVideos();
      // Auto-open the uploaded video in realtime mode
      handleSelectVideo(id);
    } catch (err) {
      console.error('Upload failed:', err);
      setUploadError('Upload failed. Please try again.');
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  const handleImportUrl = async (url: string) => {
    setUploadError(null);
    try {
      const { id } = await importVideoUrl(url);
      await fetchVideos();

      // Poll until download completes (status changes from "downloading")
      const poll = async () => {
        for (let i = 0; i < 60; i++) { // up to 2 minutes
          await new Promise(r => setTimeout(r, 2000));
          try {
            const v = await getVideo(id);
            await fetchVideos();
            if (v.status === 'ready') {
              // Download succeeded — open in realtime analysis mode
              handleSelectVideo(id);
              return;
            }
            if (v.status === 'url') {
              // Download failed — open in URL view mode
              handleSelectVideo(id);
              return;
            }
            // Still downloading...
          } catch { break; }
        }
        // Timed out — just open whatever we have
        handleSelectVideo(id);
      };
      poll();
    } catch (err) {
      console.error('URL import failed:', err);
      setUploadError('Failed to import URL. Please check the link and try again.');
    }
  };

  const handleSelectVideo = async (videoId: string) => {
    setLoading(true);
    setSelectedVideoId(videoId);
    try {
      const video = await getVideo(videoId);
      setSelectedVideo(video);

      if (video.status === 'url') {
        // URL viewing mode — embedded player, no analysis
        setResults(null);
        setDashboardMode('url');
      } else if (video.status === 'complete') {
        const data = await getResults(videoId);
        setResults(data);
        setDashboardMode('review');
      } else {
        // Realtime mode — video is ready for live analysis
        setResults(null);
        setDashboardMode('realtime');
      }
      setView('dashboard');
    } catch (err) {
      console.error('Failed to load video:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (videoId: string) => {
    try {
      await deleteVideo(videoId);
      setVideos((prev) => prev.filter((v) => v.id !== videoId));
    } catch (err) {
      console.error('Failed to delete video:', err);
    }
  };

  const handleBack = () => {
    setView('library');
    setSelectedVideoId(null);
    setSelectedVideo(null);
    setResults(null);
    fetchVideos();
  };

  return (
    <div className="min-h-screen bg-bg-primary">
      {/* Header */}
      <header className="border-b border-border bg-bg-secondary/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-[1600px] mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3 cursor-pointer" onClick={handleBack}>
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-accent to-blue-400 flex items-center justify-center shadow-lg shadow-accent-glow">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-text-primary tracking-tight">
                InsightAI
              </h1>
              <p className="text-[11px] text-text-muted">Real-time Conversation Intelligence</p>
            </div>
          </div>
          {view === 'dashboard' && (
            <button
              onClick={handleBack}
              className="flex items-center gap-2 px-4 py-2 text-sm text-text-secondary hover:text-text-primary border border-border rounded-lg hover:bg-bg-card transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
              Library
            </button>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-[1600px] mx-auto px-6 py-8">
        {view === 'library' && (
          <div className="space-y-8">
            <VideoUpload onUpload={handleUpload} onImportUrl={handleImportUrl} uploading={uploading} uploadProgress={uploadProgress} error={uploadError} />
            <VideoLibrary videos={videos} onSelect={handleSelectVideo} onDelete={handleDelete} />
            <CrossMeetingAnalytics />
          </div>
        )}
        {view === 'dashboard' && selectedVideoId && (
          <AnalysisDashboard
            results={results}
            loading={loading}
            videoId={selectedVideoId}
            video={selectedVideo}
            mode={dashboardMode}
          />
        )}
      </main>
    </div>
  );
}

export default App;
