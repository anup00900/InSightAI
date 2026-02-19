import { useState, useEffect } from 'react';
import { BarChart3, TrendingUp, Users } from 'lucide-react';

interface MeetingAnalytics {
  id: string;
  videoName: string;
  date: string;
  avgEngagement: number;
  participantCount: number;
  sentiment: string;
}

interface ComparisonData {
  meeting1: MeetingAnalytics;
  meeting2: MeetingAnalytics;
  engagementDiff: number;
  participantDiff: number;
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

  const handleMeetingClick = (id: string) => {
    if (!comparingMode) return;

    if (selectedMeetings.includes(id)) {
      setSelectedMeetings(selectedMeetings.filter(meetingId => meetingId !== id));
    } else if (selectedMeetings.length < 2) {
      const newSelected = [...selectedMeetings, id];
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

  const getSentimentColor = (sentiment: string) => {
    const lowerSentiment = sentiment.toLowerCase();
    if (lowerSentiment.includes('positive')) return 'text-green-500';
    if (lowerSentiment.includes('negative')) return 'text-red-500';
    return 'text-yellow-500';
  };

  if (loading) {
    return (
      <div className="p-6">
        <h2 className="text-2xl font-bold text-text-primary mb-6 flex items-center gap-2">
          <BarChart3 className="w-6 h-6" />
          Meeting Analytics
        </h2>
        <p className="text-text-muted">Loading analytics...</p>
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-text-primary flex items-center gap-2">
          <BarChart3 className="w-6 h-6" />
          Meeting Analytics
        </h2>
        {analytics.length > 1 && (
          <button
            onClick={toggleCompareMode}
            className={`px-4 py-2 rounded-lg border transition-colors ${
              comparingMode
                ? 'bg-accent text-white border-accent'
                : 'bg-bg-card border-border text-text-primary hover:border-accent'
            }`}
          >
            {comparingMode ? 'Cancel Compare' : 'Compare Meetings'}
          </button>
        )}
      </div>

      {analytics.length === 0 ? (
        <p className="text-text-muted">
          Analytics appear after completing realtime analyses
        </p>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
            {analytics.map((meeting) => (
              <div
                key={meeting.id}
                onClick={() => handleMeetingClick(meeting.id)}
                className={`bg-bg-card border rounded-xl p-5 transition-all ${
                  comparingMode
                    ? selectedMeetings.includes(meeting.id)
                      ? 'border-accent shadow-lg cursor-pointer'
                      : 'border-border cursor-pointer hover:border-accent/50'
                    : 'border-border'
                }`}
              >
                <h3 className="text-lg font-semibold text-text-primary mb-2">
                  {meeting.videoName}
                </h3>
                <p className="text-sm text-text-muted mb-4">
                  {new Date(meeting.date).toLocaleDateString()}
                </p>

                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-text-muted">
                      <TrendingUp className="w-4 h-4" />
                      <span className="text-sm">Engagement</span>
                    </div>
                    <span className="text-text-primary font-medium">
                      {meeting.avgEngagement.toFixed(1)}%
                    </span>
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-text-muted">
                      <Users className="w-4 h-4" />
                      <span className="text-sm">Participants</span>
                    </div>
                    <span className="text-text-primary font-medium">
                      {meeting.participantCount}
                    </span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-sm text-text-muted">Sentiment</span>
                    <span className={`font-medium ${getSentimentColor(meeting.sentiment)}`}>
                      {meeting.sentiment}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {comparisonData && (
            <div className="bg-bg-card border border-border rounded-xl p-6">
              <h3 className="text-xl font-bold text-text-primary mb-6">
                Meeting Comparison
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold text-text-primary mb-3">
                    {comparisonData.meeting1.videoName}
                  </h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-text-muted">Date</span>
                      <span className="text-text-primary">
                        {new Date(comparisonData.meeting1.date).toLocaleDateString()}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-text-muted">Engagement</span>
                      <span className="text-text-primary">
                        {comparisonData.meeting1.avgEngagement.toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-text-muted">Participants</span>
                      <span className="text-text-primary">
                        {comparisonData.meeting1.participantCount}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-text-muted">Sentiment</span>
                      <span className={getSentimentColor(comparisonData.meeting1.sentiment)}>
                        {comparisonData.meeting1.sentiment}
                      </span>
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="font-semibold text-text-primary mb-3">
                    {comparisonData.meeting2.videoName}
                  </h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-text-muted">Date</span>
                      <span className="text-text-primary">
                        {new Date(comparisonData.meeting2.date).toLocaleDateString()}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-text-muted">Engagement</span>
                      <span className="text-text-primary">
                        {comparisonData.meeting2.avgEngagement.toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-text-muted">Participants</span>
                      <span className="text-text-primary">
                        {comparisonData.meeting2.participantCount}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-text-muted">Sentiment</span>
                      <span className={getSentimentColor(comparisonData.meeting2.sentiment)}>
                        {comparisonData.meeting2.sentiment}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-6 pt-6 border-t border-border">
                <h4 className="font-semibold text-text-primary mb-4">Differences</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-text-muted">Engagement Difference</span>
                    <span
                      className={`font-medium ${
                        comparisonData.engagementDiff > 0
                          ? 'text-green-500'
                          : comparisonData.engagementDiff < 0
                          ? 'text-red-500'
                          : 'text-text-primary'
                      }`}
                    >
                      {comparisonData.engagementDiff > 0 ? '+' : ''}
                      {comparisonData.engagementDiff.toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text-muted">Participant Difference</span>
                    <span
                      className={`font-medium ${
                        comparisonData.participantDiff > 0
                          ? 'text-green-500'
                          : comparisonData.participantDiff < 0
                          ? 'text-red-500'
                          : 'text-text-primary'
                      }`}
                    >
                      {comparisonData.participantDiff > 0 ? '+' : ''}
                      {comparisonData.participantDiff}
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
