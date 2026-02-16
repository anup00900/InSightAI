/**
 * React hook for managing real-time video analysis state via WebSocket.
 * True real-time: analysis runs live as video plays.
 */

import { useEffect, useReducer, useRef, useCallback } from 'react';
import { AnalysisWebSocket } from '../lib/websocket';
import type { EmotionPoint, Participant, TranscriptSegment, Flag, Summary, CoachingItem } from '../lib/api';

// ─── Signal types ────────────────────────────────────────────────

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

export interface VoiceSignal {
  timestamp: number;
  tone: string;
  pace: string;
  energy: number;
  dynamics: string;
}

export interface WordSignal {
  timestamp: number;
  sentiment: string;
  sentiment_score: number;
  keywords: string[];
  key_phrases: string[];
}

export interface PersonalityData {
  participants: Array<{
    label: string;
    traits: string[];
    communication_style: string;
    confidence_level: string;
  }>;
}

export interface CorrelationData {
  signal_pair: string[];
  pattern: string;
  confidence: number;
  implication: string;
}

export interface AudioFeatureData {
  timestamp: number;
  energy: number;
  pitch_mean: number;
  pitch_std: number;
  volume_energy: number;
  speaking_rate: number;
  pause_count: number;
}

export interface EngagementAlertData {
  participant: string;
  score: number;
  duration_below_threshold: number;
  timestamp: number;
  suggested_action: string;
}

export interface CoachingData {
  participant: string;
  items: CoachingItem[];
}

// ─── State ───────────────────────────────────────────────────────

export interface RealtimeState {
  signals: SignalSnapshot[];
  emotionTimeline: EmotionPoint[];
  participants: Map<string, Partial<Participant>>;
  transcript: TranscriptSegment[];
  flags: Flag[];
  voiceSignals: VoiceSignal[];
  wordSignals: WordSignal[];
  personalitySignals: PersonalityData | null;
  correlations: CorrelationData[];
  summary: Summary | null;
  coaching: Map<string, CoachingItem[]>;
  statusMessage: string;
  isConnected: boolean;
  isAnalyzing: boolean;
  isComplete: boolean;
  isDetecting: boolean; // True while waiting for real participant detection
  videoEnded: boolean; // True once video ends — freezes engagement chart
  hasAudio: boolean | null; // null = unknown yet, true/false = confirmed
  audioFeatures: AudioFeatureData[];
  engagementAlerts: EngagementAlertData[];
  nameMap: Record<string, string>;
}

const initialState: RealtimeState = {
  signals: [],
  emotionTimeline: [],
  participants: new Map(),
  transcript: [],
  flags: [],
  voiceSignals: [],
  wordSignals: [],
  personalitySignals: null,
  correlations: [],
  summary: null,
  coaching: new Map(),
  statusMessage: 'Connecting...',
  isConnected: false,
  isAnalyzing: false,
  isComplete: false,
  isDetecting: false,
  videoEnded: false,
  hasAudio: null,
  audioFeatures: [],
  engagementAlerts: [],
  nameMap: {},
};

// ─── Actions ─────────────────────────────────────────────────────

type Action =
  | { type: 'SIGNALS'; data: SignalSnapshot }
  | { type: 'TRANSCRIPT'; data: { start_time: number; end_time: number; text: string; speaker: string } }
  | { type: 'VOICE'; data: VoiceSignal }
  | { type: 'WORDS'; data: WordSignal }
  | { type: 'PERSONALITY'; data: PersonalityData }
  | { type: 'CORRELATION'; data: CorrelationData }
  | { type: 'FLAG'; data: Flag }
  | { type: 'SUMMARY'; data: Summary }
  | { type: 'COACHING'; data: CoachingData }
  | { type: 'STATUS'; data: { message: string } }
  | { type: 'DETECTING' }
  | { type: 'CONNECTED' }
  | { type: 'DISCONNECTED' }
  | { type: 'COMPLETE' }
  | { type: 'VIDEO_ENDED' }
  | { type: 'NO_AUDIO' }
  | { type: 'AUDIO_FEATURES'; data: AudioFeatureData }
  | { type: 'ENGAGEMENT_ALERT'; data: EngagementAlertData }
  | { type: 'NAME_MAP'; data: Record<string, string> }
  | { type: 'SPEAKER_UPDATE'; data: { start_time: number; speaker: string } }
  | { type: 'RESET' };

// Exponential moving average helper
function ema(prev: number, next: number, alpha: number = 0.3): number {
  return prev === 0 ? next : prev + alpha * (next - prev);
}

function reducer(state: RealtimeState, action: Action): RealtimeState {
  switch (action.type) {
    case 'SIGNALS': {
      // Ignore signals arriving after video ended (prevents engagement chart from updating)
      if (state.isComplete || state.videoEnded) return state;

      const snap = action.data;
      const isHeartbeat = !!(snap as unknown as Record<string, unknown>).is_heartbeat;
      const newEmotions: EmotionPoint[] = [];
      const newParticipants = new Map(state.participants);
      const participantCount = Math.max(1, snap.participants.length);

      for (const p of snap.participants) {
        const participantId = state.nameMap[p.label] || p.label;
        newEmotions.push({
          timestamp: snap.timestamp,
          emotion: p.emotions.primary,
          confidence: p.emotions.confidence,
          engagement: p.emotions.engagement,
          participant_id: participantId,
        });

        const existing = newParticipants.get(participantId) || {
          id: participantId,
          video_id: '',
          name: participantId,
          engagement_score: 0,
          sentiment_score: 0,
          speaking_pct: 100 / participantCount, // Even distribution initially
          clarity_score: 0,
          rapport_score: 0,
          energy_score: 0,
        };

        // Engagement from emotions
        existing.engagement_score = ema(existing.engagement_score || 0, p.emotions.engagement);

        // Only update detailed scores from real analyses (not heartbeats)
        if (!isHeartbeat) {
          // Rapport from body language (openness + mirroring)
          const bl = p.body_language;
          const opennessScore = bl.openness === 'open' ? 85 : bl.openness === 'mixed' ? 55 : 25;
          const mirrorBonus = bl.mirroring ? 15 : 0;
          const rapportRaw = Math.min(100, opennessScore + mirrorBonus);
          existing.rapport_score = ema(existing.rapport_score || 0, Math.max(0, Math.min(100, rapportRaw)));

          // Clarity from confidence + posture
          const postureScore = bl.posture === 'upright' ? 80 : bl.posture === 'leaning' ? 60 : 40;
          const clarityRaw = (p.emotions.confidence * 60) + (postureScore * 0.4);
          existing.clarity_score = ema(existing.clarity_score || 0, Math.min(100, clarityRaw));

          // Energy from engagement + leaning
          const leanScore = bl.leaning === 'forward' ? 85 : bl.leaning === 'neutral' ? 60 : 35;
          const energyRaw = (p.emotions.engagement * 0.6) + (leanScore * 0.4);
          existing.energy_score = ema(existing.energy_score || 0, Math.max(0, Math.min(100, energyRaw)));
        }

        newParticipants.set(participantId, existing);
      }

      // Only add REAL frame analyses to signals array
      // (gesture/reaction/body-language panels read from this)
      let newSignals = state.signals;
      if (!isHeartbeat) {
        newSignals = [...state.signals, snap];
        if (newSignals.length > 500) newSignals.splice(0, newSignals.length - 500);

        // Speaking distribution from real frame appearances only
        const frameCounts = new Map<string, number>();
        let totalFrames = 0;
        for (const s of newSignals) {
          for (const p of s.participants) {
            frameCounts.set(p.label, (frameCounts.get(p.label) || 0) + 1);
            totalFrames++;
          }
        }
        if (totalFrames > 0) {
          for (const [label, count] of frameCounts) {
            const pData = newParticipants.get(label);
            if (pData) {
              pData.speaking_pct = (count / totalFrames) * 100;
            }
          }
        }
      }

      // Cap emotionTimeline to last 2000 entries to prevent memory leak
      let newTimeline = [...state.emotionTimeline, ...newEmotions];
      if (newTimeline.length > 2000) newTimeline = newTimeline.slice(-2000);

      return {
        ...state,
        signals: newSignals,
        emotionTimeline: newTimeline,
        participants: newParticipants,
        isAnalyzing: true,
        isDetecting: false, // First signals received — detection complete
      };
    }

    case 'TRANSCRIPT': {
      const seg: TranscriptSegment = {
        start_time: action.data.start_time,
        end_time: action.data.end_time,
        text: action.data.text,
        speaker: action.data.speaker,
        participant_id: null,
      };
      const newTranscript = [...state.transcript, seg];

      // Calculate speaking distribution from all transcript segments
      const newParticipants = new Map(state.participants);
      const speakerDurations = new Map<string, number>();
      let totalDuration = 0;
      for (const s of newTranscript) {
        const speaker = s.speaker || 'Unknown';
        const dur = Math.max(0, s.end_time - s.start_time);
        speakerDurations.set(speaker, (speakerDurations.get(speaker) || 0) + dur);
        totalDuration += dur;
      }
      if (totalDuration > 0 && newParticipants.size > 0) {
        // Map speakers to participants by order (Speaker 1 → Person 1, etc.)
        // or by name match (fallback: distribute evenly)
        const speakerList = Array.from(speakerDurations.entries()).sort((a, b) => a[0].localeCompare(b[0]));
        const participantList = Array.from(newParticipants.entries()).sort((a, b) => a[0].localeCompare(b[0]));

        // Try direct name match first, then positional mapping
        const matched = new Set<string>();
        for (const [speaker, dur] of speakerList) {
          let found = false;
          // Direct match
          for (const [pid, pData] of participantList) {
            if (!matched.has(pid) && (pid === speaker || pData.name === speaker)) {
              pData.speaking_pct = (dur / totalDuration) * 100;
              matched.add(pid);
              found = true;
              break;
            }
          }
          // Number-based match: "Speaker 1" → "Person 1" (extract number)
          if (!found) {
            const speakerNum = speaker.match(/(\d+)/)?.[1];
            if (speakerNum) {
              for (const [pid, pData] of participantList) {
                const pidNum = pid.match(/(\d+)/)?.[1];
                if (!matched.has(pid) && pidNum === speakerNum) {
                  pData.speaking_pct = (dur / totalDuration) * 100;
                  matched.add(pid);
                  found = true;
                  break;
                }
              }
            }
          }
        }

        // Fallback: if no matches worked, distribute evenly among all participants
        if (matched.size === 0 && speakerList.length > 0) {
          const evenPct = 100 / Math.max(1, newParticipants.size);
          for (const [, pData] of newParticipants) {
            pData.speaking_pct = evenPct;
          }
        }
      }

      return { ...state, transcript: newTranscript, participants: newParticipants, hasAudio: true };
    }

    case 'VOICE': {
      let newVoice = [...state.voiceSignals, action.data];
      if (newVoice.length > 200) newVoice = newVoice.slice(-200);
      // Update energy scores for all participants from voice energy
      const newParticipants = new Map(state.participants);
      const clampedEnergy = Math.max(0, Math.min(100, action.data.energy));
      for (const [, pData] of newParticipants) {
        pData.energy_score = ema(pData.energy_score || 0, clampedEnergy);
      }
      return { ...state, voiceSignals: newVoice, participants: newParticipants };
    }

    case 'WORDS': {
      let newWords = [...state.wordSignals, action.data];
      if (newWords.length > 200) newWords = newWords.slice(-200);
      // Update sentiment scores for all participants
      const newParticipants = new Map(state.participants);
      // sentiment_score: convert from 0..1 to 0..100 scale, clamp for safety
      const rawScore = action.data.sentiment_score ?? 0.5;
      const sentimentPct = Math.max(0, Math.min(100, rawScore * 100));
      for (const [, pData] of newParticipants) {
        pData.sentiment_score = ema(pData.sentiment_score || 50, sentimentPct);
      }
      return { ...state, wordSignals: newWords, participants: newParticipants };
    }

    case 'PERSONALITY':
      return { ...state, personalitySignals: action.data };

    case 'CORRELATION':
      return { ...state, correlations: [...state.correlations, action.data] };

    case 'FLAG': {
      const flag: Flag = {
        id: state.flags.length + 1,
        timestamp: action.data.timestamp || 0,
        type: action.data.type || 'neutral',
        description: action.data.description || '',
        severity: action.data.severity || 'medium',
      };
      return { ...state, flags: [...state.flags, flag] };
    }

    case 'SUMMARY':
      return {
        ...state,
        summary: {
          summary: action.data.summary,
          key_topics: action.data.key_topics,
          overall_sentiment: action.data.overall_sentiment,
        },
      };

    case 'COACHING': {
      const newCoaching = new Map(state.coaching);
      newCoaching.set(action.data.participant, action.data.items);
      return { ...state, coaching: newCoaching };
    }

    case 'DETECTING':
      return { ...state, isDetecting: true, statusMessage: 'Detecting participants...' };

    case 'STATUS':
      return { ...state, statusMessage: action.data.message };

    case 'CONNECTED':
      return { ...state, isConnected: true, statusMessage: 'Connected' };

    case 'DISCONNECTED':
      return { ...state, isConnected: false, statusMessage: 'Disconnected' };

    case 'VIDEO_ENDED':
      return { ...state, videoEnded: true, statusMessage: 'Finalizing analysis...' };

    case 'COMPLETE':
      return { ...state, isComplete: true, isAnalyzing: false, statusMessage: 'Analysis complete' };

    case 'NO_AUDIO':
      return { ...state, hasAudio: false };

    case 'AUDIO_FEATURES': {
      let newFeatures = [...state.audioFeatures, action.data];
      if (newFeatures.length > 200) newFeatures = newFeatures.slice(-200);
      return { ...state, audioFeatures: newFeatures };
    }

    case 'ENGAGEMENT_ALERT': {
      return {
        ...state,
        engagementAlerts: [...state.engagementAlerts, action.data],
      };
    }

    case 'NAME_MAP': {
      const nameMap = action.data;
      // Rename existing participants in the Map
      const renamedParticipants = new Map<string, Partial<Participant>>();
      for (const [key, pData] of state.participants) {
        const newName = nameMap[key] || key;
        renamedParticipants.set(newName, { ...pData, id: newName, name: newName });
      }
      return { ...state, nameMap, participants: renamedParticipants };
    }

    case 'SPEAKER_UPDATE': {
      const { start_time, speaker } = action.data;
      const TOLERANCE = 0.5;
      const updatedTranscript = state.transcript.map((seg) => {
        if (Math.abs(seg.start_time - start_time) <= TOLERANCE) {
          return { ...seg, speaker };
        }
        return seg;
      });
      return { ...state, transcript: updatedTranscript };
    }

    case 'RESET':
      return initialState;

    default:
      return state;
  }
}

// ─── Hook ────────────────────────────────────────────────────────

export function useRealtimeAnalysis(videoId: string, enabled: boolean = true) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const wsRef = useRef<AnalysisWebSocket | null>(null);

  useEffect(() => {
    if (!enabled) return;

    const ws = new AnalysisWebSocket();
    wsRef.current = ws;

    // Subscribe to all message types
    ws.on('signals', (data) => dispatch({ type: 'SIGNALS', data: data as SignalSnapshot }));
    ws.on('transcript', (data) => dispatch({ type: 'TRANSCRIPT', data: data as { start_time: number; end_time: number; text: string; speaker: string } }));
    ws.on('voice', (data) => dispatch({ type: 'VOICE', data: data as VoiceSignal }));
    ws.on('words', (data) => dispatch({ type: 'WORDS', data: data as WordSignal }));
    ws.on('personality', (data) => dispatch({ type: 'PERSONALITY', data: data as PersonalityData }));
    ws.on('correlation', (data) => dispatch({ type: 'CORRELATION', data: data as CorrelationData }));
    ws.on('flag', (data) => dispatch({ type: 'FLAG', data: data as Flag }));
    ws.on('summary', (data) => dispatch({ type: 'SUMMARY', data: data as Summary }));
    ws.on('coaching', (data) => dispatch({ type: 'COACHING', data: data as CoachingData }));
    ws.on('status', (data) => dispatch({ type: 'STATUS', data: data as { message: string } }));
    ws.on('error', (data) => dispatch({ type: 'STATUS', data: data as { message: string } }));
    ws.on('detecting', () => dispatch({ type: 'DETECTING' }));
    ws.on('no_audio', () => dispatch({ type: 'NO_AUDIO' }));
    ws.on('video_ended_ack', () => dispatch({ type: 'VIDEO_ENDED' }));
    ws.on('audio_features', (data) => dispatch({ type: 'AUDIO_FEATURES', data: data as AudioFeatureData }));
    ws.on('engagement_alert', (data) => dispatch({ type: 'ENGAGEMENT_ALERT', data: data as EngagementAlertData }));
    ws.on('name_map', (data) => dispatch({ type: 'NAME_MAP', data: data as Record<string, string> }));
    ws.on('speaker_update', (data) => dispatch({ type: 'SPEAKER_UPDATE', data: data as { start_time: number; speaker: string } }));
    ws.on('complete', () => dispatch({ type: 'COMPLETE' }));

    ws.connect(videoId);
    dispatch({ type: 'CONNECTED' });

    return () => {
      ws.disconnect();
      wsRef.current = null;
      dispatch({ type: 'RESET' });
    };
  }, [videoId, enabled]);

  const attachVideo = useCallback((videoElement: HTMLVideoElement) => {
    const onTimeUpdate = () => wsRef.current?.sendTick(videoElement.currentTime);
    const onPlay = () => {
      wsRef.current?.sendPlay();
      // Send immediate tick at current time — timeupdate won't fire until ~1s later
      wsRef.current?.sendTick(videoElement.currentTime);
    };
    const onPause = () => wsRef.current?.sendPause();
    const onSeeked = () => wsRef.current?.sendSeek(videoElement.currentTime);
    const onEnded = () => {
      dispatch({ type: 'VIDEO_ENDED' }); // Immediately freeze chart
      wsRef.current?.sendVideoEnded();
    };

    videoElement.addEventListener('timeupdate', onTimeUpdate);
    videoElement.addEventListener('play', onPlay);
    videoElement.addEventListener('pause', onPause);
    videoElement.addEventListener('seeked', onSeeked);
    videoElement.addEventListener('ended', onEnded);

    return () => {
      videoElement.removeEventListener('timeupdate', onTimeUpdate);
      videoElement.removeEventListener('play', onPlay);
      videoElement.removeEventListener('pause', onPause);
      videoElement.removeEventListener('seeked', onSeeked);
      videoElement.removeEventListener('ended', onEnded);
    };
  }, []);

  return { state, attachVideo };
}
