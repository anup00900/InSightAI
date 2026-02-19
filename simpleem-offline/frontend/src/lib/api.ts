const API_BASE = '/api';

export interface Video {
  id: string;
  name: string;
  filename: string;
  duration: number;
  status: string;
  created_at: string;
  overall_score: number;
  source_url: string;
  wizard_step: string;
  transcribe_progress: number;
  analyze_progress: number;
  progress_detail: string;
}

export interface VideoStatus {
  step: string;
  progress: number;
  detail: string;
}

export interface Participant {
  id: string;
  video_id: string;
  name: string;
  engagement_score: number;
  sentiment_score: number;
  speaking_pct: number;
  clarity_score: number;
  rapport_score: number;
  energy_score: number;
}

export interface EmotionPoint {
  timestamp: number;
  emotion: string;
  confidence: number;
  engagement: number;
  participant_id: string;
}

export interface TranscriptSegment {
  start_time: number;
  end_time: number;
  text: string;
  speaker: string;
  participant_id: string | null;
}

export interface Flag {
  id: number;
  timestamp: number;
  type: string;
  description: string;
  severity: string;
}

export interface CoachingItem {
  id: number;
  recommendation: string;
  category: string;
  priority: number;
}

export interface Summary {
  summary: string;
  key_topics: string[];
  overall_sentiment: string;
}

export interface SignalSnapshot {
  timestamp: number;
  participants: Array<{
    label: string;
    emotions: { primary: string; confidence: number; engagement: number };
    body_language: {
      posture: string;
      openness: string;
      leaning: string;
      mirroring: boolean;
      description: string;
    };
    gestures: string[];
    reactions: string[];
    confidence?: number;
  }>;
}

export interface VoiceSignalData {
  timestamp: number;
  tone: string;
  pace: string;
  energy: number;
  dynamics: string;
}

export interface WordSignalData {
  timestamp: number;
  sentiment: string;
  sentiment_score: number;
  keywords: string[];
  key_phrases: string[];
}

export interface PersonalitySignalData {
  participants: Array<{
    label: string;
    traits: string[];
    communication_style: string;
    confidence_level: string;
  }>;
}

export interface CorrelationPattern {
  signal_pair: string[];
  pattern: string;
  confidence: number;
  implication: string;
}

export interface AudioFeatureData {
  timestamp: number;
  pitch_mean: number;
  volume_energy: number;
  speaking_rate: number;
  energy: number;
  pause_count: number;
}

export interface SpeakerAudioData {
  participants: Array<{
    name: string;
    features: Array<{
      timestamp: number;
      pitch_mean: number;
      pitch_std: number;
      volume_energy: number;
      speaking_rate: number;
      pause_ratio: number;
      spectral_centroid: number;
      engagement_score: number;
    }>;
  }>;
}

export interface MeetingNotes {
  action_items: Array<{ owner: string; task: string; deadline: string | null }>;
  decisions: Array<{ decision: string; context: string }>;
  follow_ups: Array<{ topic: string; owner: string | null; timeline: string | null }>;
  key_questions: Array<{ question: string; raised_by: string | null }>;
}

export interface ConfidenceSummary {
  avg_confidence: number;
  min_confidence: number;
  total_snapshots: number;
  disagreement_count: number;
  accuracy_grade: string;
}

export interface AnalysisResults {
  video: Video;
  participants: Participant[];
  emotions: EmotionPoint[];
  transcript: TranscriptSegment[];
  flags: Flag[];
  summary: Summary | null;
}

export async function uploadVideo(
  file: File,
  onProgress?: (pct: number) => void,
): Promise<{ id: string; status: string }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE}/upload`);

    if (onProgress) {
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
      };
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        reject(new Error(xhr.responseText || `Upload failed (${xhr.status})`));
      }
    };
    xhr.onerror = () => reject(new Error('Network error during upload'));
    xhr.ontimeout = () => reject(new Error('Upload timed out'));
    xhr.timeout = 600000; // 10 minutes for large files

    const formData = new FormData();
    formData.append('file', file);
    xhr.send(formData);
  });
}

export async function importVideoUrl(url: string): Promise<{ id: string; status: string }> {
  const res = await fetch(`${API_BASE}/import-url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listVideos(): Promise<Video[]> {
  const res = await fetch(`${API_BASE}/videos`);
  if (!res.ok) throw new Error('Failed to fetch videos');
  return res.json();
}

export async function deleteVideo(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/videos/${id}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to delete video');
}

export async function getVideo(id: string): Promise<Video> {
  const res = await fetch(`${API_BASE}/videos/${id}`);
  if (!res.ok) throw new Error('Video not found');
  return res.json();
}

export async function getResults(id: string): Promise<AnalysisResults> {
  const res = await fetch(`${API_BASE}/videos/${id}/results`);
  if (!res.ok) throw new Error('Results not available');
  return res.json();
}

export async function getCoaching(videoId: string, participantId: string): Promise<CoachingItem[]> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/coaching/${participantId}`);
  if (!res.ok) throw new Error('Coaching not available');
  return res.json();
}

export async function triggerTranscription(videoId: string): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/transcribe`, { method: 'POST' });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function triggerAnalysis(videoId: string): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/analyze`, { method: 'POST' });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getVideoStatus(videoId: string): Promise<VideoStatus> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/status`);
  if (!res.ok) throw new Error('Status not available');
  return res.json();
}

export async function getTranscript(videoId: string): Promise<TranscriptSegment[]> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/transcript`);
  if (!res.ok) throw new Error('Transcript not available');
  return res.json();
}

export async function getLiveTranscript(videoId: string): Promise<TranscriptSegment[]> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/transcript`);
  if (!res.ok) return [];
  return res.json();
}

export async function uploadTranscriptFile(videoId: string, file: File): Promise<{ status: string; segments: number }> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/videos/${videoId}/upload-transcript`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function uploadTranscriptText(videoId: string, text: string): Promise<{ status: string; segments: number }> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/upload-transcript-text`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getParticipants(videoId: string): Promise<Participant[]> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/participants`);
  if (!res.ok) throw new Error('Participants not available');
  return res.json();
}

export interface PartialResults {
  video: Video;
  ready_sections: string[];
  transcript?: TranscriptSegment[];
  participants?: Participant[];
  emotions?: EmotionPoint[];
  summary?: Summary;
  flags?: Flag[];
}

export async function getPartialResults(videoId: string): Promise<PartialResults> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/partial-results`);
  if (!res.ok) throw new Error('Partial results not available');
  return res.json();
}

export function getThumbnailUrl(videoId: string): string {
  return `${API_BASE}/videos/${videoId}/thumbnail`;
}

export interface MeetingAnalytics {
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

export async function getAnalytics(): Promise<MeetingAnalytics[]> {
  const res = await fetch(`${API_BASE}/analytics`);
  if (!res.ok) return [];
  return res.json();
}

export async function getSignalSnapshots(videoId: string): Promise<SignalSnapshot[]> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/signals/snapshots`);
  if (!res.ok) return [];
  return res.json();
}

export async function getVoiceSignals(videoId: string): Promise<VoiceSignalData[]> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/signals/voice`);
  if (!res.ok) return [];
  return res.json();
}

export async function getWordSignals(videoId: string): Promise<WordSignalData[]> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/signals/words`);
  if (!res.ok) return [];
  return res.json();
}

export async function getPersonalitySignals(videoId: string): Promise<PersonalitySignalData | null> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/signals/personality`);
  if (!res.ok) return null;
  return res.json();
}

export async function getCorrelations(videoId: string): Promise<CorrelationPattern[]> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/signals/correlations`);
  if (!res.ok) return [];
  return res.json();
}

export async function getAudioFeatures(videoId: string): Promise<AudioFeatureData[]> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/signals/audio-features`);
  if (!res.ok) return [];
  return res.json();
}

export async function getSpeakerAudio(videoId: string): Promise<SpeakerAudioData | null> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/signals/speaker-audio`);
  if (!res.ok) return null;
  return res.json();
}

export async function getMeetingNotes(videoId: string): Promise<MeetingNotes | null> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/meeting-notes`);
  if (!res.ok) return null;
  return res.json();
}

export async function getConfidenceSummary(videoId: string): Promise<ConfidenceSummary | null> {
  const res = await fetch(`${API_BASE}/videos/${videoId}/signals/confidence`);
  if (!res.ok) return null;
  return res.json();
}

export function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export function getEmotionColor(emotion: string): string {
  const colors: Record<string, string> = {
    happy: '#10b981',
    excited: '#10b981',
    engaged: '#3b82f6',
    neutral: '#94a3b8',
    surprised: '#f59e0b',
    confused: '#f97316',
    concerned: '#ef4444',
    disengaged: '#ef4444',
  };
  return colors[emotion] || '#94a3b8';
}

export function getScoreColor(score: number): string {
  if (score >= 80) return '#10b981';
  if (score >= 60) return '#3b82f6';
  if (score >= 40) return '#f59e0b';
  return '#ef4444';
}
