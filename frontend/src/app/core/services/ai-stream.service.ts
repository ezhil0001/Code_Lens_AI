import { Injectable, NgZone } from '@angular/core';
import { BehaviorSubject, Observable, Subject } from 'rxjs';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import { environment } from '../../../environments/environment';

/**
 * AIStreamService
 * Handles streaming responses from FastAPI backend using fetchEventSource
 * Supports both GET and POST-based Server-Sent Events (SSE)
 */
@Injectable({
  providedIn: 'root',
})
export class AIStreamService {
  // Stream state management
  private streamingSubject = new Subject<string>();
  public streaming$: Observable<string> = this.streamingSubject.asObservable();

  // Current message being streamed
  private fullMessageSubject = new BehaviorSubject<string>('');
  public fullMessage$: Observable<string> =
    this.fullMessageSubject.asObservable();

  // Loading state
  private loadingSubject = new BehaviorSubject<boolean>(false);
  public loading$: Observable<boolean> = this.loadingSubject.asObservable();

  // Error handling
  private errorSubject = new BehaviorSubject<string | null>(null);
  public error$: Observable<string | null> = this.errorSubject.asObservable();

  // Streaming controller for cleanup
  private streamController: AbortController | null = null;

  constructor(private ngZone: NgZone) {}

  /**
   * Stream a chat message from the FastAPI backend
   * Uses POST-based SSE for maximum compatibility
   *
   * Flow:
   * 1. Send query to /api/v1/stream endpoint
   * 2. Backend streams tokens as Server-Sent Events
   * 3. Client accumulates tokens in real-time
   * 4. UI updates reactively via Observable
   *
   * @param query - User's question
   * @param repositoryId - Repository context (optional)
   * @param useHybridSearch - Enable hybrid (BM25 + Vector) search
   */
  streamChatResponse(
    query: string,
    repositoryId?: string,
    useHybridSearch: boolean = true,
    sessionId?: string,
    userId?: string,
  ): Observable<void> {
    return new Observable((observer) => {
      this.loadingSubject.next(true);
      this.errorSubject.next(null);
      this.fullMessageSubject.next('');
      this.streamController = new AbortController();

      // Generate stable session/user ids if not provided
      const resolvedSessionId =
        sessionId ??
        sessionStorage.getItem('chat_session_id') ??
        (() => {
          const id = `sess-${crypto.randomUUID()}`;
          sessionStorage.setItem('chat_session_id', id);
          return id;
        })();
      const resolvedUserId =
        userId ??
        localStorage.getItem('chat_user_id') ??
        (() => {
          const id = `anon-${crypto.randomUUID()}`;
          localStorage.setItem('chat_user_id', id);
          return id;
        })();

      const payload = {
        query,
        session_id: resolvedSessionId,
        user_id: resolvedUserId,
        stream: true,
        context: repositoryId
          ? { repo_id: repositoryId, use_hybrid_search: useHybridSearch }
          : undefined,
      };

      /**
       * fetchEventSource Configuration
       * Alternative to EventSource - supports POST, custom headers, credentials
       * This is critical for RAG systems where you need:
       * - Authentication headers
       * - POST request body
       * - Cross-origin requests
       */
      fetchEventSource(
        `${environment.apiUrl}${environment.endpoints.chatStream}`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            // Add auth token here if needed
            // 'Authorization': `Bearer ${this.authService.getToken()}`
          },
          body: JSON.stringify(payload),
          signal: this.streamController.signal,
          openWhenHidden: true,

          /**
           * onopen: Connection established
           * Check HTTP status and headers
           */
          onopen: async (response) => {
            const ct = response.headers.get('content-type') ?? '';
            if (response.ok && ct.includes('text/event-stream')) {
              console.log('✅ Stream connection established');
            } else if (response.ok) {
              // Backend returned 200 but with a different content-type (e.g. JSON error body).
              // Don't throw — let onmessage handle it gracefully.
              console.warn(`⚠️ Unexpected content-type: ${ct}`);
            } else {
              throw new Error(`Stream connection failed: ${response.status}`);
            }
          },

          /**
           * onmessage: Token received
           * Each SSE message contains:
           * data: {
           *   "token": "the",
           *   "type": "content|metadata|citations",
           *   "metadata": {...}  // optional
           * }
           */
          onmessage: (event) => {
            this.ngZone.run(() => {
              try {
                const data = JSON.parse(event.data);

                if (data.type === 'heartbeat') {
                  return; // Silent — don't emit anything to UI
                }

                // Handle different token types
                // Backend sends type='token' with field 'content',
                // but also support legacy type='content' with field 'token'.
                // AFTER:
                if (data.type === 'token' || data.type === 'content') {
                  const chunk: string = data.content ?? data.token ?? '';
                  const currentMessage = this.fullMessageSubject.value;
                  const accumulated = currentMessage + chunk;

                  // Try to extract answer from JSON if complete
                  const cleaned = this.extractAnswer(accumulated);
                  this.fullMessageSubject.next(cleaned);
                  this.streamingSubject.next(chunk);
                } else if (data.type === 'citations') {
                  // Citation/metadata event
                  console.log('📚 Citations received:', data.citations);
                } else if (data.type === 'done') {
                  // Stream complete — log final response
                  const finalMessage = this.fullMessageSubject.value;
                  console.log('\n' + '='.repeat(80));
                  console.log('✅ STREAM COMPLETED - FINAL RESPONSE');
                  console.log('='.repeat(80));
                  console.log('📝 Full Message:', finalMessage);
                  console.log('📊 Metadata:', data.metadata);
                  console.log('-'.repeat(80));
                  console.log(finalMessage);
                  console.log('-'.repeat(80));
                  console.log('='.repeat(80) + '\n');

                  this.loadingSubject.next(false);
                  observer.next();
                  observer.complete();
                }
              } catch (error) {
                console.error('❌ Failed to parse token:', error);
                this.errorSubject.next('Failed to parse response');
              }
            });
          },

          /**
           * onerror: Connection or parsing error
           * Return a retry interval (ms) to keep the connection alive on transient
           * errors. Only throw to permanently close the connection on fatal errors
           * (e.g. 4xx responses).
           */
          onerror: (error) => {
            console.warn('⚠️ Stream error:', error);

            // AbortError = user cancelled — complete silently
            if (error instanceof Error && error.name === 'AbortError') {
              this.ngZone.run(() => {
                this.loadingSubject.next(false);
                observer.complete();
              });
              throw error; // re-throw to stop fetchEventSource
            }

            // Fatal HTTP errors (4xx) — don't retry
            if (error instanceof Error && error.message.includes('4')) {
              this.ngZone.run(() => {
                this.loadingSubject.next(false);
                this.errorSubject.next(error.message);
                observer.error(error);
              });
              throw error; // re-throw to stop fetchEventSource
            }

            // RAG pipeline is stateful — NEVER auto-retry (would start 2nd pipeline)
            console.warn(
              '⚠️ Stream error — not retrying (stateful RAG pipeline)',
            );
            this.ngZone.run(() => {
              this.loadingSubject.next(false);
              this.errorSubject.next(
                'Connection lost. Please send your message again.',
              );
              observer.error(error);
            });
            throw error; // ← Must throw to stop fetchEventSource completely
          },
        },
      ).catch((error) => {
        if (error.name !== 'AbortError') {
          this.ngZone.run(() => {
            console.error('❌ Stream fetch error:', error);
            this.errorSubject.next(error.message || 'Stream failed');
            observer.error(error);
          });
        }
      });
    });
  }
  /**
   * Extract answer from JSON response if applicable.
   * Backend sometimes wraps response in JSON: {"answer": "...", "sources": [...]}
   */
  private extractAnswer(content: string): string {
    try {
      // Try to parse as JSON — only if looks like JSON
      const trimmed = content.trim();
      if (trimmed.startsWith('{') || trimmed.startsWith('```json')) {
        const jsonStr = trimmed
          .replace(/^```json\s*/, '')
          .replace(/\s*```$/, '')
          .trim();
        const parsed = JSON.parse(jsonStr);
        if (parsed.answer) {
          return parsed.answer;
        }
      }
    } catch {
      // Not complete JSON yet — return as-is (still streaming)
    }
    return content;
  }
  /**
   * Cancel ongoing stream
   * Used when user clicks "Stop" or navigates away
   */
  cancelStream(): void {
    if (this.streamController) {
      this.streamController.abort();
      this.streamController = null;
      this.loadingSubject.next(false);
    }
  }

  /**
   * Get the full accumulated message
   */
  getFullMessage(): string {
    return this.fullMessageSubject.value;
  }

  /**
   * Get current loading state
   */
  isLoading(): boolean {
    return this.loadingSubject.value;
  }

  /**
   * Clear state (for new messages)
   */
  clearState(): void {
    this.fullMessageSubject.next('');
    this.errorSubject.next(null);
  }
}

/**
 * STREAMING ARCHITECTURE EXPLANATION
 *
 * Why SSE (Server-Sent Events) for RAG?
 * =====================================
 * 1. UNIDIRECTIONAL: Server → Client only (perfect for responses)
 * 2. AUTO-RECONNECT: Built-in reconnection logic
 * 3. REAL-TIME: No polling, instant token delivery
 * 4. BANDWIDTH: Text-based, smaller overhead than WebSocket
 * 5. FIREWALL: Standard HTTP, works through proxies
 *
 *
 * Why fetchEventSource instead of EventSource?
 * =============================================
 * EventSource limitations:
 *   ✗ GET requests only
 *   ✗ No custom headers
 *   ✗ No authentication support
 *   ✗ No request body
 *
 * fetchEventSource advantages:
 *   ✓ POST requests (send authentication, filters)
 *   ✓ Custom headers (Bearer tokens, API keys)
 *   ✓ Request body (send context, settings)
 *   ✓ Better error handling
 *
 *
 * Data Flow in Chat:
 * ==================
 * User Query
 *    ↓
 * streamChatResponse(query)
 *    ↓
 * POST /api/v1/stream with body
 *    ↓
 * FastAPI starts SSE stream
 *    ↓
 * Backend: Retrieve context (hybrid search)
 *    ↓
 * Backend: Call LLM with streaming
 *    ↓
 * Backend: Send tokens as SSE events
 *    ↓
 * Frontend: onmessage receives token
 *    ↓
 * Frontend: Accumulate in fullMessage$
 *    ↓
 * UI: Subscribe to fullMessage$ → display
 *    ↓
 * When done: Backend sends "done" event
 *    ↓
 * Frontend: Stream complete, UI finalized
 *
 *
 * Token Types in Events:
 * ======================
 * "content"  → LLM output tokens (visible text)
 * "citations" → Source documents (RAG context)
 * "metadata" → Timing, token counts, model info
 * "done"     → Stream finished signal
 *
 *
 * Error Handling:
 * ===============
 * Network Error   → Catch at onerror, show toast
 * Parse Error     → Log, continue (don't break stream)
 * Abort           → User clicked stop, silent cleanup
 * Server Error    → Backend sends error event, UI shows
 *
 *
 * UI Pattern:
 * ===========
 * <div>
 *   {{ fullMessage$ | async }}  <!-- Update as tokens arrive -->
 *   <span *ngIf="loading$ | async" class="animate-pulse">▌</span>
 * </div>
 *
 * Why is this pattern used?
 * - No flickering (incremental, not replacing)
 * - Natural reading speed (tokens arrive gradually)
 * - Can show citations below streaming text
 * - Users see progress immediately (better UX)
 */
