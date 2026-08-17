/**
 * AgentStreamService — LangGraph SSE typed event dispatcher
 *
 * Fully-typed dispatcher that understands all 8 SSE event types produced by
 * POST /api/v2/chat/stream.
 *
 * SSE Event types (from backend streaming.py):
 *   token        — LLM streaming token
 *   tool_call    — Tool invocation started
 *   tool_result  — Tool invocation completed
 *   agent_switch — Graph entered a new agent sub-graph
 *   checkpoint   — A graph node completed and state was checkpointed
 *   interrupt    — HIL interrupt (graph paused, awaiting human input)
 *   done         — Graph execution completed
 *   error        — Fatal error during graph execution
 *
 * Usage:
 *   inject AgentStreamService, call sendMessage(), subscribe to the Observables,
 *   call approveHIL() / rejectHIL() from the approval banner.
 */

import { Injectable, NgZone } from '@angular/core';
import {
  BehaviorSubject,
  Observable,
  Subject,
  from,
  of,
} from 'rxjs';
import { HttpClient } from '@angular/common/http';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import { environment } from '../../../environments/environment';

// ─────────────────────────────────────────────────────────────────────────────
// Public model types
// ─────────────────────────────────────────────────────────────────────────────

/** All SSE event types emitted by the v2 backend. */
export type SSEEventType =
  | 'token'
  | 'token_reset'
  | 'tool_call'
  | 'tool_result'
  | 'agent_switch'
  | 'checkpoint'
  | 'interrupt'
  | 'done'
  | 'error';

/** Full envelope sent by backend for every SSE event. */
export interface AgentSSEEvent {
  type: SSEEventType;
  data: Record<string, unknown>;
  agent: string;
  checkpoint_id: string;
  ts: number;
}

/** Live graph node entry, used by AgentActivityComponent. */
export interface AgentActivityEntry {
  id: string;               // unique per event
  label: string;            // human-readable node label
  agent: string;
  type: SSEEventType;
  status: 'running' | 'done' | 'error';
  durationMs?: number;
  toolName?: string;        // for tool_call / tool_result
  toolInput?: unknown;
  checkpoint_id?: string;
  ts: number;
}

/** HIL interrupt payload. */
export interface HILInterruptPayload {
  reason: string;
  checkpoint_id: string;
}

/** One retrieval source backing the answer, sent on the terminal `done` event. */
export interface SourceRef {
  file_path: string;
  score: number;
  snippet: string;
}

/** Summary row returned by GET /api/v2/sessions/{id}/checkpoints */
export interface CheckpointSummary {
  checkpoint_id: string;
  parent_id: string | null;
  created_at: string | null;
  nodes_visited: string[];
  query_preview: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Service
// ─────────────────────────────────────────────────────────────────────────────

@Injectable({ providedIn: 'root' })
export class AgentStreamService {
  // ── Streaming token accumulator ───────────────────────────────────────────
  private _fullMessage = new BehaviorSubject<string>('');
  /** Full accumulated markdown response as it streams. */
  readonly fullMessage$: Observable<string> = this._fullMessage.asObservable();

  // ── Retrieval sources backing the current answer ──────────────────────────
  private _sources = new BehaviorSubject<SourceRef[]>([]);
  /** Sources from the terminal `done` event; empty until the answer completes. */
  readonly sources$: Observable<SourceRef[]> = this._sources.asObservable();

  // ── Loading / streaming state ─────────────────────────────────────────────
  private _loading = new BehaviorSubject<boolean>(false);
  readonly loading$: Observable<boolean> = this._loading.asObservable();

  // ── Per-event subject (raw) ───────────────────────────────────────────────
  private _events = new Subject<AgentSSEEvent>();
  /** All SSE events as they arrive — subscribe for custom handling. */
  readonly events$: Observable<AgentSSEEvent> = this._events.asObservable();

  // ── Agent activity log ────────────────────────────────────────────────────
  private _activity = new BehaviorSubject<AgentActivityEntry[]>([]);
  /** Ordered list of graph activity entries (nodes visited, tools called). */
  readonly activity$: Observable<AgentActivityEntry[]> = this._activity.asObservable();

  // ── HIL interrupt state ───────────────────────────────────────────────────
  private _hil = new BehaviorSubject<HILInterruptPayload | null>(null);
  /**
   * Non-null when the graph is paused for human review.
   * Drives the approval banner in ChatComponent.
   */
  readonly hilInterrupt$: Observable<HILInterruptPayload | null> = this._hil.asObservable();

  // ── Error ─────────────────────────────────────────────────────────────────
  private _error = new BehaviorSubject<string | null>(null);
  readonly error$: Observable<string | null> = this._error.asObservable();

  // ── Checkpoint list ───────────────────────────────────────────────────────
  private _checkpoints = new BehaviorSubject<CheckpointSummary[]>([]);
  readonly checkpoints$: Observable<CheckpointSummary[]> =
    this._checkpoints.asObservable();

  // ── Internal ──────────────────────────────────────────────────────────────
  private _abortController: AbortController | null = null;
  private _lastCheckpointId: string = '';
  private _currentSessionId: string = '';
  private _nodeStartMap = new Map<string, number>(); // agent → start ts

  constructor(private http: HttpClient, private ngZone: NgZone) {}

  // ─────────────────────────────────────────────────────────────────────────
  // Public: send a new message
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Send a query to POST /api/v2/chat/stream and stream the response.
   *
   * @param query       User message text
   * @param sessionId   Session identifier (namespaced server-side)
   * @param options     Optional overrides (agentHint, hilEnabled, resume)
   */
  sendMessage(
    query: string,
    sessionId: string,
    options: {
      agentHint?: string;
      hilEnabled?: boolean;
      hilThreshold?: number;
      resumeFromCheckpoint?: string;
    } = {},
  ): void {
    this._reset();
    this._currentSessionId = sessionId;
    this._loading.next(true);

    const userId =
      localStorage.getItem('chat_user_id') ??
      (() => {
        const id = `user-${crypto.randomUUID()}`;
        localStorage.setItem('chat_user_id', id);
        return id;
      })();

    const payload: Record<string, unknown> = {
      query,
      session_id: sessionId,
      user_id: userId,
      stream: true,
      hil_enabled: options.hilEnabled ?? false,
      hil_confidence_threshold: options.hilThreshold ?? 0.5,
    };
    if (options.agentHint) payload['agent_hint'] = options.agentHint;
    if (options.resumeFromCheckpoint)
      payload['resume_from_checkpoint'] = options.resumeFromCheckpoint;

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    // Reconnect support (F-46): send Last-Event-ID when we have a checkpoint
    if (this._lastCheckpointId) {
      headers['Last-Event-ID'] = this._lastCheckpointId;
    }

    const token = localStorage.getItem('auth_token');
    if (token) headers['Authorization'] = `Bearer ${token}`;

    this._abortController = new AbortController();

    fetchEventSource(
      `${environment.apiUrl}${environment.endpoints.chatStreamV2}`,
      {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
        signal: this._abortController.signal,
        openWhenHidden: true,

        onopen: async (response) => {
          const ct = response.headers.get('content-type') ?? '';
          if (!response.ok || !ct.includes('text/event-stream')) {
            throw new Error(`v2 stream failed: HTTP ${response.status}`);
          }
        },

        onmessage: (msg) => {
          this.ngZone.run(() => {
            try {
              const event: AgentSSEEvent = JSON.parse(msg.data);
              this._dispatch(event);
            } catch {
              // Malformed event — silently ignore
            }
          });
        },

        onerror: (err) => {
          this.ngZone.run(() => {
            if (err instanceof Error && err.name === 'AbortError') {
              this._loading.next(false);
              return;
            }
            this._error.next((err as Error)?.message ?? 'Stream error');
            this._loading.next(false);
          });
          throw err; // don't auto-retry (stateful graph)
        },
      },
    ).catch((err) => {
      this.ngZone.run(() => {
        if (err?.name !== 'AbortError') {
          this._error.next(err?.message ?? 'Stream failed');
          this._loading.next(false);
        }
      });
    });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public: HIL approve / reject
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Resume a paused graph after HIL review.
   * POSTs to /api/v2/sessions/{session_id}/resume and streams the continuation.
   */
  resolveHIL(approved: boolean, humanInput: string): void {
    const interrupt = this._hil.value;
    if (!interrupt) return;

    this._hil.next(null); // dismiss banner immediately
    this._addActivity({
      label: approved ? '✅ Approved by reviewer' : '❌ Rejected by reviewer',
      agent: 'Supervisor',
      type: 'checkpoint',
      status: 'done',
      checkpoint_id: interrupt.checkpoint_id,
    });

    const token = localStorage.getItem('auth_token');
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const url = `${environment.apiUrl}${environment.endpoints.sessions}/${this._currentSessionId}/resume`;
    const body = {
      human_input: humanInput,
      approved,
      checkpoint_id: interrupt.checkpoint_id || undefined,
    };

    this._loading.next(true);
    this._abortController = new AbortController();

    fetchEventSource(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: this._abortController.signal,
      openWhenHidden: true,
      onopen: async () => {},
      onmessage: (msg) => {
        this.ngZone.run(() => {
          try {
            const event: AgentSSEEvent = JSON.parse(msg.data);
            this._dispatch(event);
          } catch {}
        });
      },
      onerror: (err) => {
        this.ngZone.run(() => {
          this._error.next((err as Error)?.message ?? 'Resume failed');
          this._loading.next(false);
        });
        throw err;
      },
    }).catch(() => this.ngZone.run(() => this._loading.next(false)));
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public: cancel stream
  // ─────────────────────────────────────────────────────────────────────────

  cancelStream(): void {
    this._abortController?.abort();
    this._abortController = null;
    this._loading.next(false);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public: checkpoint list
  // ─────────────────────────────────────────────────────────────────────────

  /** Load checkpoint history for a session from the API. */
  loadCheckpoints(sessionId: string): Observable<CheckpointSummary[]> {
    const token = localStorage.getItem('auth_token');
    const headers: Record<string, string> = {};
    if (token) headers['Authorization'] = `Bearer ${token}`;

    return this.http.get<{ checkpoints: CheckpointSummary[] }>(
      `${environment.apiUrl}${environment.endpoints.sessions}/${sessionId}/checkpoints`,
      { headers },
    ) as unknown as Observable<CheckpointSummary[]>;
  }

  /** Replay graph from a historical checkpoint (opens a new stream). */
  replayFromCheckpoint(sessionId: string, checkpointId: string): void {
    this._reset();
    this._currentSessionId = sessionId;
    this._loading.next(true);

    const token = localStorage.getItem('auth_token');
    const headers: Record<string, string> = {};
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const url = `${environment.apiUrl}${environment.endpoints.sessions}/${sessionId}/replay/${checkpointId}`;
    this._abortController = new AbortController();

    fetchEventSource(url, {
      method: 'GET',
      headers,
      signal: this._abortController.signal,
      openWhenHidden: true,
      onopen: async () => {},
      onmessage: (msg) => {
        this.ngZone.run(() => {
          try { this._dispatch(JSON.parse(msg.data)); } catch {}
        });
      },
      onerror: (err) => {
        this.ngZone.run(() => this._loading.next(false));
        throw err;
      },
    }).catch(() => this.ngZone.run(() => this._loading.next(false)));
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Internal: event dispatcher
  // ─────────────────────────────────────────────────────────────────────────

  private _dispatch(event: AgentSSEEvent): void {
    this._events.next(event);

    // Track last checkpoint for reconnect
    if (event.checkpoint_id) {
      this._lastCheckpointId = event.checkpoint_id;
    }

    switch (event.type) {
      case 'token': {
        const content = (event.data['content'] as string) ?? '';
        this._fullMessage.next(this._fullMessage.value + content);
        break;
      }

      // Multi-agent runs stream each agent's draft answer before the
      // synthesiser streams the merged one. Drop the drafts so the user sees
      // the final answer once instead of the same facts two or three times.
      case 'token_reset': {
        this._fullMessage.next('');
        break;
      }

      case 'agent_switch': {
        const to = (event.data['to'] as string) ?? event.agent;
        this._nodeStartMap.set(to, Date.now());
        this._addActivity({
          label: `🔵 ${to}`,
          agent: to,
          type: 'agent_switch',
          status: 'running',
          checkpoint_id: event.checkpoint_id,
        });
        break;
      }

      case 'tool_call': {
        const toolName = (event.data['tool'] as string) ?? 'tool';
        this._addActivity({
          label: `🔧 ${toolName}`,
          agent: event.agent,
          type: 'tool_call',
          status: 'running',
          toolName,
          toolInput: event.data['input'],
          checkpoint_id: event.checkpoint_id,
        });
        break;
      }

      case 'tool_result': {
        const tName = (event.data['tool'] as string) ?? 'tool';
        this._updateLastOfType('tool_call', 'done');
        this._addActivity({
          label: `✅ ${tName} result`,
          agent: event.agent,
          type: 'tool_result',
          status: 'done',
          toolName: tName,
          checkpoint_id: event.checkpoint_id,
        });
        break;
      }

      case 'checkpoint': {
        const node = (event.data['node'] as string) ?? event.agent;
        const startTs = this._nodeStartMap.get(node);
        const durationMs = startTs ? Date.now() - startTs : undefined;
        this._nodeStartMap.delete(node);
        this._updateLastOfType('agent_switch', 'done');
        this._addActivity({
          label: `✓ ${node}`,
          agent: node,
          type: 'checkpoint',
          status: 'done',
          durationMs,
          checkpoint_id: event.checkpoint_id,
        });
        break;
      }

      case 'interrupt': {
        const reason =
          (event.data['reason'] as string) ?? 'Human review required';
        this._hil.next({
          reason,
          checkpoint_id: event.checkpoint_id,
        });
        this._addActivity({
          label: `⚠️ HIL: ${reason.slice(0, 60)}`,
          agent: 'Supervisor',
          type: 'interrupt',
          status: 'running',
          checkpoint_id: event.checkpoint_id,
        });
        break;
      }

      case 'done': {
        this._loading.next(false);
        this._markAllRunningDone();
        const sources = Array.isArray(event.data['sources'])
          ? (event.data['sources'] as SourceRef[])
          : [];
        this._sources.next(sources);
        this._addActivity({
          label: '✅ Response complete',
          agent: 'Supervisor',
          type: 'done',
          status: 'done',
          checkpoint_id: event.checkpoint_id,
        });
        break;
      }

      case 'error': {
        const msg = (event.data['message'] as string) ?? 'Unknown error';
        this._error.next(msg);
        this._loading.next(false);
        this._addActivity({
          label: `❌ ${msg.slice(0, 80)}`,
          agent: event.agent,
          type: 'error',
          status: 'error',
        });
        break;
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Internal helpers
  // ─────────────────────────────────────────────────────────────────────────

  private _reset(): void {
    this._abortController?.abort();
    this._abortController = null;
    this._fullMessage.next('');
    this._sources.next([]);
    this._activity.next([]);
    this._hil.next(null);
    this._error.next(null);
    this._lastCheckpointId = '';
    this._nodeStartMap.clear();
  }

  private _addActivity(
    partial: Omit<AgentActivityEntry, 'id' | 'ts'>,
  ): void {
    const entry: AgentActivityEntry = {
      id: crypto.randomUUID(),
      ts: Date.now(),
      ...partial,
    };
    this._activity.next([...this._activity.value, entry]);
  }

  private _updateLastOfType(type: SSEEventType, status: AgentActivityEntry['status']): void {
    const list = [...this._activity.value];
    for (let i = list.length - 1; i >= 0; i--) {
      if (list[i].type === type && list[i].status === 'running') {
        list[i] = { ...list[i], status };
        break;
      }
    }
    this._activity.next(list);
  }

  private _markAllRunningDone(): void {
    const updated = this._activity.value.map((e) =>
      e.status === 'running' ? { ...e, status: 'done' as const } : e,
    );
    this._activity.next(updated);
  }
}
