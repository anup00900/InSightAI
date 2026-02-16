import { useState, useEffect, useCallback } from 'react';
import { type Video, type AnalysisResults, listVideos, getVideo, getResults, deleteVideo } from './lib/api';
import VideoLibrary from './components/VideoLibrary';
import AnalysisDashboard from './components/AnalysisDashboard';
import CrossMeetingAnalytics from './components/CrossMeetingAnalytics';
import WizardShell from './components/wizard/WizardShell';
import UploadStep from './components/wizard/UploadStep';
import TranscriptStep from './components/wizard/TranscriptStep';
import AnalysisStep from './components/wizard/AnalysisStep';
import { Activity, ArrowLeft, Plus } from 'lucide-react';

type View = 'library' | 'wizard' | 'dashboard';

function App() {
  const [view, setView] = useState<View>('library');
  const [videos, setVideos] = useState<Video[]>([]);
  const [selectedVideoId, setSelectedVideoId] = useState<string | null>(null);
  const [selectedVideo, setSelectedVideo] = useState<Video | null>(null);
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [loading, setLoading] = useState(false);
  const [wizardStep, setWizardStep] = useState(1);

  // ── Fetch video list ───────────────────────────────────────────
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

  // ── Select a video from library ────────────────────────────────
  const handleSelectVideo = async (videoId: string) => {
    setLoading(true);
    setSelectedVideoId(videoId);
    try {
      const video = await getVideo(videoId);
      setSelectedVideo(video);

      // Route to the appropriate view based on video status
      switch (video.status) {
        case 'complete':
          // Completed video — go directly to dashboard
          {
            const data = await getResults(videoId);
            setResults(data);
            setView('dashboard');
          }
          break;

        case 'uploaded':
        case 'ready':
          // Needs transcription — wizard step 2
          setResults(null);
          setWizardStep(2);
          setView('wizard');
          break;

        case 'transcribing':
          // Transcription in progress — wizard step 2
          setResults(null);
          setWizardStep(2);
          setView('wizard');
          break;

        case 'transcribed':
          // Needs analysis — wizard step 3
          setResults(null);
          setWizardStep(3);
          setView('wizard');
          break;

        case 'analyzing':
          // Analysis in progress — wizard step 3
          setResults(null);
          setWizardStep(3);
          setView('wizard');
          break;

        default:
          // Fallback: open in dashboard
          setResults(null);
          setView('dashboard');
          break;
      }
    } catch (err) {
      console.error('Failed to load video:', err);
    } finally {
      setLoading(false);
    }
  };

  // ── Delete a video ─────────────────────────────────────────────
  const handleDelete = async (videoId: string) => {
    try {
      await deleteVideo(videoId);
      setVideos((prev) => prev.filter((v) => v.id !== videoId));
    } catch (err) {
      console.error('Failed to delete video:', err);
    }
  };

  // ── Back to library ────────────────────────────────────────────
  const handleBack = () => {
    setView('library');
    setSelectedVideoId(null);
    setSelectedVideo(null);
    setResults(null);
    setWizardStep(1);
    fetchVideos();
  };

  // ── Start new analysis (wizard from step 1) ────────────────────
  const handleNewAnalysis = () => {
    setSelectedVideoId(null);
    setSelectedVideo(null);
    setResults(null);
    setWizardStep(1);
    setView('wizard');
  };

  // ── Wizard step handlers ───────────────────────────────────────
  const handleUploadComplete = (videoId: string) => {
    setSelectedVideoId(videoId);
    setWizardStep(2);
    fetchVideos(); // refresh the library in background
  };

  const handleTranscriptContinue = () => {
    setWizardStep(3);
  };

  const handleAnalysisComplete = async () => {
    setWizardStep(4);
    // Fetch results for the dashboard view within the wizard
    if (selectedVideoId) {
      setLoading(true);
      try {
        const [video, data] = await Promise.all([
          getVideo(selectedVideoId),
          getResults(selectedVideoId),
        ]);
        setSelectedVideo(video);
        setResults(data);
      } catch (err) {
        console.error('Failed to load results:', err);
      } finally {
        setLoading(false);
      }
    }
  };

  // ── Render wizard step content ─────────────────────────────────
  const renderWizardContent = () => {
    switch (wizardStep) {
      case 1:
        return <UploadStep onUploadComplete={handleUploadComplete} />;
      case 2:
        return selectedVideoId ? (
          <TranscriptStep videoId={selectedVideoId} onContinue={handleTranscriptContinue} />
        ) : null;
      case 3:
        return selectedVideoId ? (
          <AnalysisStep videoId={selectedVideoId} onComplete={handleAnalysisComplete} />
        ) : null;
      case 4:
        return selectedVideoId ? (
          <AnalysisDashboard
            results={results}
            loading={loading}
            videoId={selectedVideoId}
            video={selectedVideo}
          />
        ) : null;
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-bg-primary">
      {/* ── Header ──────────────────────────────────────────────── */}
      <header className="bg-white/[0.03] backdrop-blur-xl border-b border-white/10 sticky top-0 z-50">
        <div className="max-w-[1600px] mx-auto px-6 py-4 flex items-center justify-between">
          {/* Logo & title */}
          <div className="flex items-center gap-3 cursor-pointer" onClick={handleBack}>
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center shadow-lg shadow-purple-500/25">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-text-primary tracking-tight">
                InsightAI
              </h1>
              <p className="text-[11px] text-text-muted">Offline Meeting Intelligence</p>
            </div>
          </div>

          {/* Action buttons */}
          <div className="flex items-center gap-3">
            {view === 'library' && (
              <button
                onClick={handleNewAnalysis}
                className="flex items-center gap-2 px-5 py-2.5 text-sm font-medium text-white rounded-xl bg-gradient-to-r from-indigo-500 to-purple-500 hover:from-indigo-400 hover:to-purple-400 shadow-lg shadow-purple-500/25 transition-all duration-200 hover:shadow-purple-500/40"
              >
                <Plus className="w-4 h-4" />
                New Analysis
              </button>
            )}
            {(view === 'wizard' || view === 'dashboard') && (
              <button
                onClick={handleBack}
                className="flex items-center gap-2 px-4 py-2 text-sm text-text-secondary hover:text-text-primary border border-white/10 rounded-xl hover:bg-white/[0.05] transition-all duration-200"
              >
                <ArrowLeft className="w-4 h-4" />
                Back to Library
              </button>
            )}
          </div>
        </div>
      </header>

      {/* ── Main Content ────────────────────────────────────────── */}
      <main className="max-w-[1600px] mx-auto px-6 py-8">
        {/* Library view */}
        {view === 'library' && (
          <div className="space-y-8">
            <VideoLibrary videos={videos} onSelect={handleSelectVideo} onDelete={handleDelete} onNewAnalysis={handleNewAnalysis} />
            <CrossMeetingAnalytics />
          </div>
        )}

        {/* Wizard view */}
        {view === 'wizard' && (
          <WizardShell currentStep={wizardStep}>
            {renderWizardContent()}
          </WizardShell>
        )}

        {/* Dashboard view (direct from library for completed videos) */}
        {view === 'dashboard' && selectedVideoId && (
          <AnalysisDashboard
            results={results}
            loading={loading}
            videoId={selectedVideoId}
            video={selectedVideo}
          />
        )}
      </main>
    </div>
  );
}

export default App;
