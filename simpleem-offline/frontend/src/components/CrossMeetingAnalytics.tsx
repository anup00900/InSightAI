import { useState, useEffect } from 'react';
import { BarChart3, TrendingUp, Users } from 'lucide-react';

interface MeetingAnalytics {
  video_id: string;
  video_name: string;
  meeting_date: string;
  avg_engagement: number;
  participant_count: number;
  overall_sentiment: string;
  avg_visual_engagement: number;
  avg_audio_engagement: number;
  avg_verbal_engagement: number;
  duration: number;
}

interface ComparisonData {
  meeting1: MeetingAnalytics;
  meeting2: MeetingAnalytics;
  engagement_diff: number;
  participant_diff: number;
}

export default function CrossMeetingAnalytics() {
  const [analytics, setAnalytics] = useState<MeetingAnalytics[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedMeetings, setSelectedMeetings] = useState<string[]>([]);
  const [comparisonData, setComparisonData] = useState<ComparisonData | null>(null);
  const [comparingMode, setComparingMode] = useState(false);

  useEffect(() => {
    fetchAnalytics();
  }, []);

  const fetchAnalytics = async () => {
    try {
      const response = await fetch('/api/analytics');
      if (response.ok) {
        const data = await response.json();
        setAnalytics(data);
      }
    } catch (error) {
      console.error('Failed to fetch analytics:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleMeetingClick = (vid: string) => {
    if (!comparingMode) return;

    if (selectedMeetings.includes(vid)) {
      setSelectedMeetings(selectedMeetings.filter(meetingId => meetingId !== vid));
    } else if (selectedMeetings.length < 2) {
      const newSelected = [...selectedMeetings, vid];
      setSelectedMeetings(newSelected);

      if (newSelected.length === 2) {
        fetchComparison(newSelected[0], newSelected[1]);
      }
    }
  };

  const fetchComparison = async (id1: string, id2: string) => {
    try {
      const response = await fetch(`/api/analytics/compare?ids=${id1},${id2}`);
      if (response.ok) {
        const data = await response.json();
        setComparisonData(data);
      }
    } catch (error) {
      console.error('Failed to fetch comparison:', error);
    }
  };

  const toggleCompareMode = () => {
    setComparingMode(!comparingMode);
    setSelectedMeetings([]);
    setComparisonData(null);
  };

  const getSentimentColor = (sentiment: string | undefined) => {
    const lowerSentiment = (sentiment || '').toLowerCase();
    if (lowerSentiment.includes('positive')) return 'text-green-500';
    if (lowerSentiment.includes('negative')) return 'text-red-500';
    return 'text-yellow-500';
  };

  if (loading) {
    return (
      <div className="p-6">
        <h2 className="text-2xl font-bold gradient-text mb-6 flex items-center gap-2">
          <BarChart3 className="w-6 h-6 text-purple-400" />
          Meeting Analytics
        </h2>
        <p className="text-slate-500">Loading analytics...</p>
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold gradient-text flex items-center gap-2">
          <BarChart3 className="w-6 h-6 text-purple-400" />
          Meeting Analytics
        </h2>
        {analytics.length > 1 && (
          <button
            onClick={toggleCompareMode}
            className={`px-4 py-2 rounded-lg border transition-all duration-300 ${
              comparingMode
                ? 'bg-gradient-to-r from-indigo-500 to-purple-500 text-white border-purple-500/50 shadow-lg shadow-purple-500/20'
                : 'bg-white/5 border-white/10 text-slate-200 hover:border-purple-400/50 hover:bg-white/10'
            }`}
          >
            {comparingMode ? 'Cancel Compare' : 'Compare Meetings'}
          </button>
        )}
      </div>

      {analytics.length === 0 ? (
        <p className="text-text-muted">
          Analytics appear after completing analyses
        </p>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
            {analytics.map((meeting) => (
              <div
                key={meeting.video_id}
                onClick={() => handleMeetingClick(meeting.video_id)}
                className={`glass-card p-5 transition-all duration-300 ${
                  comparingMode
                    ? selectedMeetings.includes(meeting.video_id)
                      ? 'border-purple-400/50 shadow-lg shadow-purple-500/20 cursor-pointer'
                      : 'cursor-pointer hover:border-purple-400/30 hover:bg-white/8'
                    : ''
                }`}
              >
                <h3 className="text-lg font-semibold text-slate-200 mb-2">
                  {meeting.video_name}
                </h3>
                <p className="text-sm text-slate-500 mb-4">
                  {new Date(meeting.meeting_date).toLocaleDateString()}
                </p>

                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-slate-500">
                      <TrendingUp className="w-4 h-4" />
                      <span className="text-sm">Engagement</span>
                    </div>
                    <span className="text-slate-200 font-medium">
                      {(meeting.avg_engagement ?? 0).toFixed(1)}%
                    </span>
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-slate-500">
                      <Users className="w-4 h-4" />
                      <span className="text-sm">Participants</span>
                    </div>
                    <span className="text-slate-200 font-medium">
                      {meeting.participant_count}
                    </span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-sm text-slate-500">Sentiment</span>
                    <span className={`font-medium ${getSentimentColor(meeting.overall_sentiment)}`}>
                      {meeting.overall_sentiment || 'N/A'}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {comparisonData && (
            <div className="glass-card p-6">
              <h3 className="text-xl font-bold gradient-text mb-6">
                Meeting Comparison
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold text-slate-200 mb-3">
                    {comparisonData.meeting1.video_name}
                  </h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-slate-500">Date</span>
                      <span className="text-slate-200">
                        {new Date(comparisonData.meeting1.meeting_date).toLocaleDateString()}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Engagement</span>
                      <span className="text-slate-200">
                        {(comparisonData.meeting1.avg_engagement ?? 0).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Participants</span>
                      <span className="text-slate-200">
                        {comparisonData.meeting1.participant_count}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Sentiment</span>
                      <span className={getSentimentColor(comparisonData.meeting1.overall_sentiment)}>
                        {comparisonData.meeting1.overall_sentiment || 'N/A'}
                      </span>
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="font-semibold text-slate-200 mb-3">
                    {comparisonData.meeting2.video_name}
                  </h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-slate-500">Date</span>
                      <span className="text-slate-200">
                        {new Date(comparisonData.meeting2.meeting_date).toLocaleDateString()}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Engagement</span>
                      <span className="text-slate-200">
                        {(comparisonData.meeting2.avg_engagement ?? 0).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Participants</span>
                      <span className="text-slate-200">
                        {comparisonData.meeting2.participant_count}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Sentiment</span>
                      <span className={getSentimentColor(comparisonData.meeting2.overall_sentiment)}>
                        {comparisonData.meeting2.overall_sentiment || 'N/A'}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-6 pt-6 border-t border-white/10">
                <h4 className="font-semibold text-slate-200 mb-4">Differences</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Engagement Difference</span>
                    <span
                      className={`font-medium ${
                        comparisonData.engagement_diff > 0
                          ? 'text-emerald-400'
                          : comparisonData.engagement_diff < 0
                          ? 'text-red-400'
                          : 'text-slate-200'
                      }`}
                    >
                      {comparisonData.engagement_diff > 0 ? '+' : ''}
                      {comparisonData.engagement_diff.toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Participant Difference</span>
                    <span
                      className={`font-medium ${
                        comparisonData.participant_diff > 0
                          ? 'text-emerald-400'
                          : comparisonData.participant_diff < 0
                          ? 'text-red-400'
                          : 'text-slate-200'
                      }`}
                    >
                      {comparisonData.participant_diff > 0 ? '+' : ''}
                      {comparisonData.participant_diff}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
