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

      // Build emotion points for timeline (needed for both heartbeats and real signals)
      // Only include participants that exist in our known set (prevents phantom timeline entries)
      const hasNameMap = Object.keys(state.nameMap).length > 0;
      const newEmotions: EmotionPoint[] = snap.participants
        .filter((p: SignalSnapshot['participants'][0]) => {
          if (!hasNameMap) return true; // No name map yet — accept all
          const resolved = state.nameMap[p.label] || p.label;
          return state.participants.has(resolved) || Array.from(state.participants.keys()).some(k =>
            k.toLowerCase() === resolved.toLowerCase()
          );
        })
        .map((p: SignalSnapshot['participants'][0]) => ({
          timestamp: snap.timestamp,
          emotion: p.emotions.primary,
          confidence: p.emotions.confidence,
          engagement: p.emotions.engagement,
          participant_id: state.nameMap[p.label] || p.label,
        }));

      // FAST PATH: heartbeats — update emotionTimeline + speaking distribution
      if (isHeartbeat) {
        // Apply speaking distribution even from heartbeats (diarization sends it this way)
        const hbSpeakingDist = (snap as unknown as Record<string, unknown>).speaking_distribution as Record<string, number> | undefined;
        let newParticipants = state.participants;
        if (hbSpeakingDist && Object.keys(hbSpeakingDist).length > 0) {
          newParticipants = new Map(state.participants);
          for (const [pid, pData] of newParticipants) {
            let pct = hbSpeakingDist[pid];
            if (pct === undefined) {
              // Fuzzy match by first name
              const pidLower = pid.toLowerCase();
              for (const [speaker, val] of Object.entries(hbSpeakingDist)) {
                if (speaker.toLowerCase() === pidLower) { pct = val; break; }
              }
            }
            if (pct !== undefined) {
              pData.speaking_pct = pct;
            }
          }
        }

        if (newEmotions.length === 0 && newParticipants === state.participants) return state;
        let newTimeline: EmotionPoint[];
        if (newEmotions.length === 0) {
          newTimeline = state.emotionTimeline;
        } else if (state.emotionTimeline.length >= 100) {
          newTimeline = [...state.emotionTimeline.slice(-(100 - newEmotions.length)), ...newEmotions];
        } else {
          newTimeline = [...state.emotionTimeline, ...newEmotions];
        }
        return { ...state, emotionTimeline: newTimeline, participants: newParticipants, isAnalyzing: true, isDetecting: false };
      }

      // FULL PATH: real frame analysis — MERGE into existing participants
      // Only add NEW participants; update existing ones. Never wipe the Map.
      const newParticipants = new Map(state.participants);

      for (const p of snap.participants) {
        let participantId = state.nameMap[p.label] || p.label;

        // Fuzzy match: if participantId not in map, try case-insensitive + prefix match
        if (!newParticipants.has(participantId)) {
          const idLower = participantId.toLowerCase();
          for (const [existingId] of newParticipants) {
            const exLower = existingId.toLowerCase();
            if (exLower === idLower) {
              participantId = existingId; // Use existing key
              break;
            }
            // Word overlap match (exact OR prefix with 4+ chars)
            const idWords = idLower.split(/\s+/).filter(w => w.length >= 3);
            const exWords = exLower.split(/\s+/).filter((w: string) => w.length >= 3);
            let matched = false;
            for (const iw of idWords) {
              for (const ew of exWords) {
                if (iw === ew) { matched = true; break; }
                // Prefix match: "rangwan" matches "rangwani", "downey" matches "dowdney"
                if (iw.length >= 4 && ew.length >= 4 && (iw.startsWith(ew) || ew.startsWith(iw))) {
                  matched = true; break;
                }
              }
              if (matched) break;
            }
            if (matched) {
              participantId = existingId;
              break;
            }
            // Substring match on full name
            if (idLower.length >= 4 && exLower.length >= 4 && (idLower.includes(exLower) || exLower.includes(idLower))) {
              participantId = existingId;
              break;
            }
          }
        }

        if (newParticipants.has(participantId)) {
          // UPDATE existing
          const existing = newParticipants.get(participantId)!;
          existing.engagement_score = ema(existing.engagement_score || 0, p.emotions.engagement);

          const bl = p.body_language;
          const opennessScore = bl.openness === 'open' ? 85 : bl.openness === 'mixed' ? 55 : 25;
          const mirrorBonus = bl.mirroring ? 15 : 0;
          existing.rapport_score = ema(existing.rapport_score || 0, Math.min(100, opennessScore + mirrorBonus));

          const postureScore = bl.posture === 'upright' ? 80 : bl.posture === 'leaning' ? 60 : 40;
          existing.clarity_score = ema(existing.clarity_score || 0, Math.min(100, (p.emotions.confidence * 60) + (postureScore * 0.4)));

          const leanScore = bl.leaning === 'forward' ? 85 : bl.leaning === 'neutral' ? 60 : 35;
          existing.energy_score = ema(existing.energy_score || 0, Math.min(100, (p.emotions.engagement * 0.6) + (leanScore * 0.4)));

          // Use backend-provided speaking_pct (backend tracks from visual detection)
          if ((p as Record<string, unknown>).speaking_pct !== undefined) {
            existing.speaking_pct = (p as Record<string, unknown>).speaking_pct as number;
          }
        } else {
          // Only add new participants if we haven't received a name_map yet.
          // Once name_map is received, the participant list is LOCKED —
          // NAME_MAP handler is the sole authority for who exists.
          const hasNameMap = Object.keys(state.nameMap).length > 0;
          if (!hasNameMap) {
            newParticipants.set(participantId, {
              id: participantId,
              video_id: '',
              name: participantId,
              engagement_score: p.emotions.engagement,
              sentiment_score: 0,
              speaking_pct: 0,
              clarity_score: 0,
              rapport_score: 0,
              energy_score: 0,
            });
          }
          // else: silently ignore — unresolved label, not a known participant
        }
      }

      // Apply speaking distribution from backend to ALL participants
      // Only update when backend provides actual distribution data (not heartbeats)
      const speakingDist = (snap as unknown as Record<string, unknown>).speaking_distribution as Record<string, number> | undefined;
      if (speakingDist && Object.keys(speakingDist).length > 0) {
        for (const [pid, pData] of newParticipants) {
          // Try exact match first
          let pct = speakingDist[pid];
          // Try fuzzy match if no exact match
          if (pct === undefined) {
            const pidLower = pid.toLowerCase();
            for (const [speaker, val] of Object.entries(speakingDist)) {
              const spLower = speaker.toLowerCase();
              if (spLower === pidLower) { pct = val; break; }
              // Word overlap (exact or prefix with 4+ chars)
              const pidWords = pidLower.split(/\s+/).filter(w => w.length >= 3);
              const spWords = spLower.split(/\s+/).filter(w => w.length >= 3);
              let found = false;
              for (const pw of pidWords) {
                for (const sw of spWords) {
                  if (pw === sw || (pw.length >= 4 && sw.length >= 4 && (pw.startsWith(sw) || sw.startsWith(pw)))) {
                    found = true; break;
                  }
                }
                if (found) break;
              }
              if (found) { pct = val; break; }
            }
          }
          // Only update if we have a value; preserve existing % for non-speakers (stays at 0)
          if (pct !== undefined) {
            pData.speaking_pct = pct;
          }
          // Non-speakers not in speakingDist keep their current value (0 or previous %)
        }
      }

      // Build a STABLE snapshot that includes ALL known participants (not just current frame)
      // This prevents signal panels from flickering as different frames detect different subsets
      const stableParticipants: SignalSnapshot['participants'] = [];
      const seenLabels = new Set<string>();

      // First: add current frame's participants (fresh data)
      for (const p of snap.participants) {
        const resolvedLabel = state.nameMap[p.label] || p.label;
        if (!seenLabels.has(resolvedLabel)) {
          stableParticipants.push({ ...p, label: resolvedLabel });
          seenLabels.add(resolvedLabel);
        }
      }

      // Second: carry forward participants from previous snapshot that aren't in this frame
      // Resolve through nameMap to prevent stale labels from before name_map was received
      if (state.signals.length > 0) {
        const prevSnap = state.signals[state.signals.length - 1];
        for (const p of prevSnap.participants) {
          const resolvedPrevLabel = state.nameMap[p.label] || p.label;
          if (!seenLabels.has(resolvedPrevLabel) && !/^Person \d+$/.test(resolvedPrevLabel) && newParticipants.has(resolvedPrevLabel)) {
            stableParticipants.push({ ...p, label: resolvedPrevLabel });
            seenLabels.add(resolvedPrevLabel);
          }
        }
      }

      const stableSnap: SignalSnapshot = { ...snap, participants: stableParticipants };

      // Efficient signal cap
      let newSignals: SignalSnapshot[];
      if (state.signals.length >= 60) {
        newSignals = [...state.signals.slice(-59), stableSnap];
      } else {
        newSignals = [...state.signals, stableSnap];
      }

      // Efficient emotionTimeline cap
      let newTimeline: EmotionPoint[];
      if (state.emotionTimeline.length >= 100) {
        newTimeline = [...state.emotionTimeline.slice(-(100 - newEmotions.length)), ...newEmotions];
      } else {
        newTimeline = [...state.emotionTimeline, ...newEmotions];
      }

      return {
        ...state,
        signals: newSignals,
        emotionTimeline: newTimeline,
        participants: newParticipants,
        isAnalyzing: true,
        isDetecting: false,
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
      // Transcript cap — keep enough for accurate speaking distribution
      let newTranscript: TranscriptSegment[];
      if (state.transcript.length >= 500) {
        newTranscript = [...state.transcript.slice(-499), seg];
      } else {
        newTranscript = [...state.transcript, seg];
      }

      // Speaking distribution is now tracked by the BACKEND (via visual is_speaking detection)
      // and sent with each SIGNALS payload as speaking_pct per participant.
      // The TRANSCRIPT handler does NOT modify speaking_pct — it only adds transcript segments.

      return { ...state, transcript: newTranscript, hasAudio: true };
    }

    case 'VOICE': {
      const newVoice = state.voiceSignals.length >= 60
        ? [...state.voiceSignals.slice(-59), action.data]
        : [...state.voiceSignals, action.data];
      // Update energy scores — skip Map clone, just update values
      const clampedEnergy = Math.max(0, Math.min(100, action.data.energy));
      for (const [, pData] of state.participants) {
        pData.energy_score = ema(pData.energy_score || 0, clampedEnergy);
      }
      return { ...state, voiceSignals: newVoice };
    }

    case 'WORDS': {
      const newWords = state.wordSignals.length >= 60
        ? [...state.wordSignals.slice(-59), action.data]
        : [...state.wordSignals, action.data];
      // Update sentiment scores — skip Map clone, just update values
      const rawScore = action.data.sentiment_score ?? 0.5;
      const sentimentPct = Math.max(0, Math.min(100, rawScore * 100));
      for (const [, pData] of state.participants) {
        pData.sentiment_score = ema(pData.sentiment_score || 50, sentimentPct);
      }
      return { ...state, wordSignals: newWords };
    }

    case 'PERSONALITY':
      return { ...state, personalitySignals: action.data };

    case 'CORRELATION': {
      const newCorrs = state.correlations.length >= 30
        ? [...state.correlations.slice(-29), action.data]
        : [...state.correlations, action.data];
      return { ...state, correlations: newCorrs };
    }

    case 'FLAG': {
      const flag: Flag = {
        id: state.flags.length + 1,
        timestamp: action.data.timestamp || 0,
        type: action.data.type || 'neutral',
        description: action.data.description || '',
        severity: action.data.severity || 'medium',
      };
      const newFlags = state.flags.length >= 50
        ? [...state.flags.slice(-49), flag]
        : [...state.flags, flag];
      return { ...state, flags: newFlags };
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
      const newFeatures = state.audioFeatures.length >= 60
        ? [...state.audioFeatures.slice(-59), action.data]
        : [...state.audioFeatures, action.data];
      return { ...state, audioFeatures: newFeatures };
    }

    case 'ENGAGEMENT_ALERT': {
      const newAlerts = state.engagementAlerts.length >= 30
        ? [...state.engagementAlerts.slice(-29), action.data]
        : [...state.engagementAlerts, action.data];
      return { ...state, engagementAlerts: newAlerts };
    }

    case 'NAME_MAP': {
      const nameMap = action.data;
      // AUTHORITATIVE: rebuild participant list from name_map values ONLY.
      // name_map values are the canonical set — nothing else exists.
      const authoritative = new Set(
        Object.values(nameMap).filter((n): n is string => typeof n === 'string' && !/^Person \d+$/.test(n))
      );

      // Helper: fuzzy match a name against existing state.participants
      const findExisting = (name: string): Partial<Participant> | null => {
        // Exact match
        const exact = state.participants.get(name);
        if (exact) return exact;
        // Case-insensitive + word overlap
        const nl = name.toLowerCase();
        for (const [existKey, existData] of state.participants) {
          const el = existKey.toLowerCase();
          if (nl === el) return existData;
          const nWords = nl.split(/\s+/).filter(w => w.length >= 3);
          const eWords = el.split(/\s+/).filter(w => w.length >= 3);
          for (const nw of nWords) {
            for (const ew of eWords) {
              if (nw === ew) return existData;
              if (nw.length >= 4 && ew.length >= 4 && (nw.startsWith(ew) || ew.startsWith(nw))) return existData;
            }
          }
          if (nl.length >= 4 && el.length >= 4 && (nl.includes(el) || el.includes(nl))) return existData;
        }
        return null;
      };

      const newParticipants = new Map<string, Partial<Participant>>();
      for (const name of authoritative) {
        const existing = findExisting(name);
        if (existing) {
          // Carry over scores from existing entry
          newParticipants.set(name, { ...existing, id: name, name: name });
        } else {
          // Brand new participant (late joiner)
          newParticipants.set(name, {
            id: name,
            video_id: '',
            name: name,
            engagement_score: 0,
            sentiment_score: 0,
            speaking_pct: 0,
            clarity_score: 0,
            rapport_score: 0,
            energy_score: 0,
          });
        }
      }

      return { ...state, nameMap, participants: newParticipants };
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
