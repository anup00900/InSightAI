/**
 * WebSocket client for real-time video analysis.
 * Features: exponential backoff reconnection, max retries, error logging.
 */

export type MessageType =
  | 'signals'
  | 'transcript'
  | 'voice'
  | 'words'
  | 'personality'
  | 'correlation'
  | 'flag'
  | 'summary'
  | 'coaching'
  | 'status'
  | 'error'
  | 'no_audio'
  | 'detecting'
  | 'video_ended_ack'
  | 'complete'
  | 'audio_features'
  | 'engagement_alert'
  | 'name_map';

type MessageHandler = (data: unknown) => void;

const MAX_RECONNECT_RETRIES = 5;
const BASE_RECONNECT_DELAY = 1000; // 1s, 2s, 4s, 8s, 16s

export class AnalysisWebSocket {
  private ws: WebSocket | null = null;
  private handlers = new Map<MessageType, Set<MessageHandler>>();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private videoId: string = '';
  private _isConnected = false;
  private _reconnectAttempts = 0;
  private _intentionalClose = false;

  get isConnected() {
    return this._isConnected;
  }

  connect(videoId: string) {
    this.videoId = videoId;
    this._intentionalClose = false;
    this._reconnectAttempts = 0;
    this.cleanup();

    this._doConnect();
  }

  private _doConnect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${window.location.host}/ws/analyze/${this.videoId}`;

    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      this._isConnected = true;
      this._reconnectAttempts = 0; // Reset on successful connection
      this.emit('status', { message: 'Connected' });
    };

    this.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        const type = msg.type as MessageType;
        const data = msg.data;
        // Only log non-frequent message types (skip signals/status to avoid memory pressure)
        if (type !== 'signals' && type !== 'status' && type !== 'audio_features') {
          console.log(`[WS] ${type}`);
        }
        this.emit(type, data);
      } catch (e) {
        console.warn('[WS] Malformed message:', e);
      }
    };

    this.ws.onclose = () => {
      this._isConnected = false;

      if (this._intentionalClose) return;

      // Exponential backoff reconnection with max retries
      if (this._reconnectAttempts < MAX_RECONNECT_RETRIES && this.videoId) {
        const delay = BASE_RECONNECT_DELAY * Math.pow(2, this._reconnectAttempts);
        this._reconnectAttempts++;
        this.emit('status', { message: `Reconnecting (${this._reconnectAttempts}/${MAX_RECONNECT_RETRIES})...` });
        this.reconnectTimer = setTimeout(() => this._doConnect(), delay);
      } else {
        this.emit('status', { message: 'Disconnected' });
      }
    };

    this.ws.onerror = () => {
      this.emit('error', { message: 'WebSocket error' });
    };
  }

  sendTick(timestamp: number) {
    this.send({ action: 'tick', timestamp });
  }

  sendPlay() {
    this.send({ action: 'play' });
  }

  sendPause() {
    this.send({ action: 'pause' });
  }

  sendSeek(timestamp: number) {
    this.send({ action: 'seek', timestamp });
  }

  sendVideoEnded() {
    this.send({ action: 'video_ended' });
  }

  on(type: MessageType, handler: MessageHandler): () => void {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, new Set());
    }
    this.handlers.get(type)!.add(handler);
    return () => {
      this.handlers.get(type)?.delete(handler);
    };
  }

  disconnect() {
    this._intentionalClose = true;
    this.cleanup();
    this.videoId = '';
    this.handlers.clear(); // Clear all handlers on disconnect
  }

  private send(data: Record<string, unknown>) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify(data));
      } catch {
        // Socket closed between readyState check and send — ignore
      }
    }
  }

  private emit(type: MessageType, data: unknown) {
    const handlers = this.handlers.get(type);
    if (handlers) {
      for (const handler of handlers) {
        handler(data);
      }
    }
  }

  private cleanup() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.onerror = null;
      this.ws.onmessage = null;
      this.ws.close();
      this.ws = null;
    }
    this._isConnected = false;
  }
}
