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
