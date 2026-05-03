/**
 * Phase 4: Chat Service - Angular Integration
 *
 * Handles communication with FastAPI backend:
 * - Stream management via Server-Sent Events (SSE)
 * - Session tracking
 * - Cache interaction
 * - Error handling with reconnection
 */

import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, Subject, BehaviorSubject, throwError } from 'rxjs';
import { catchError, tap, retry, timeout } from 'rxjs/operators';
import { environment } from '../../environments/environment';

// Default environment configuration
const defaultEnvironment = {
  apiUrl: 'http://localhost:8000',
  wsUrl: 'ws://localhost:8000',
  production: false,
  cacheEnabled: true,
  streamingEnabled: true,
  healthCheckInterval: 30000,
};

// Try to import environment, fallback to defaults
const appEnvironment = { ...defaultEnvironment, ...environment };

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  metadata?: {
    retrieval_success?: boolean;
    retrieval_sources?: any[];
    tokens?: number;
    cached?: boolean;
  };
}

export interface ChatRequest {
  query: string;
  session_id: string;
  user_id: string;
  stream: boolean;
}

export interface ChatStreamToken {
  type: 'token' | 'done' | 'error' | 'heartbeat' | 'content' | 'citations';
  content?: string;
  metadata?: any;
}

export interface ChatStreamResponse {
  content: string;
  session_id: string;
  sources: any[];
  metadata: any;
  timestamp: Date;
}

@Injectable({
  providedIn: 'root',
})
export class ChatService {
  private apiUrl = `${appEnvironment.apiUrl}/api/v1`;
  private wsUrl = appEnvironment.wsUrl || 'ws://localhost:8000';
  private sessionId: string = this.generateSessionId();
  private userId: string = this.getCurrentUserId();

  // Observable streams
  private messageSubject = new Subject<ChatMessage>();
  public messages$ = this.messageSubject.asObservable();

  private streamingSubject = new BehaviorSubject<boolean>(false);
  public isStreaming$ = this.streamingSubject.asObservable();

  private errorSubject = new Subject<string>();
  public errors$ = this.errorSubject.asObservable();

  private cacheStatusSubject = new BehaviorSubject<any>(null);
  public cacheStatus$ = this.cacheStatusSubject.asObservable();

  // Local message history
  private messageHistory: ChatMessage[] = [];

  constructor(private http: HttpClient) {
    this.loadMessageHistory();
    this.loadCacheStatus();
  }

  /**
   * Stream chat response from backend using Server-Sent Events.
   *
   * Features:
   * - Real-time token streaming
   * - Automatic reconnection
   * - Graceful error handling
   * - Local message persistence
   */
  streamChat(query: string): Observable<string> {
    const chatRequest: ChatRequest = {
      query,
      session_id: this.sessionId,
      user_id: this.userId,
      stream: true,
    };

    // Add user message to history immediately
    this.addMessageToHistory({
      role: 'user',
      content: query,
      timestamp: new Date(),
    });

    this.streamingSubject.next(true);
    const fullResponse$ = new Subject<string>();
    let fullContent = '';

    // Use fetch for SSE (better than HttpClient for streaming)
    this.connectSSE(chatRequest)
      .then(async (response) => {
        const reader = response.body?.getReader();
        const decoder = new TextDecoder();

        try {
          while (true) {
            const { done, value } = (await reader?.read()) || {};
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                try {
                  const data: ChatStreamToken = JSON.parse(line.slice(6));

                  // AFTER:
                  if (data.type === 'token' && data.content) {
                    fullContent += data.content;
                    fullResponse$.next(data.content); // Raw token, service handles display
                  } else if (data.type === 'heartbeat') {
                    // Skip heartbeats — don't emit to UI
                  } else if (data.type === 'done') {
                    // Extract answer from JSON if backend wrapped it
                    let displayContent = fullContent;
                    try {
                      const cleaned = fullContent
                        .replace(/^```json\s*/, '')
                        .replace(/\s*```$/, '')
                        .trim();
                      const parsed = JSON.parse(cleaned);
                      if (parsed.answer) {
                        displayContent = parsed.answer;
                      }
                    } catch {
                      /* plain text — use as-is */
                    }

                    this.addMessageToHistory({
                      role: 'assistant',
                      content: displayContent,
                      timestamp: new Date(),
                      metadata: data.metadata,
                    });
                    fullResponse$.complete();
                  } else if (data.type === 'error') {
                    this.handleStreamError(data.content || 'Stream error');
                    fullResponse$.error(new Error(data.content));
                  }
                } catch (e) {
                  // Ignore JSON parse errors
                }
              }
            }
          }
        } catch (error) {
          this.handleStreamError(`Connection error: ${error}`);
          fullResponse$.error(error);
        } finally {
          this.streamingSubject.next(false);
        }
      })
      .catch((error) => {
        this.handleStreamError(`Connection failed: ${error}`);
        fullResponse$.error(error);
        this.streamingSubject.next(false);
      });

    return fullResponse$.asObservable();
  }

  /**
   * Non-streaming chat (compatibility mode).
   */
  sendChat(query: string): Observable<ChatStreamResponse> {
    const chatRequest: ChatRequest = {
      query,
      session_id: this.sessionId,
      user_id: this.userId,
      stream: false,
    };

    return this.http
      .post<ChatStreamResponse>(`${this.apiUrl}/chat`, chatRequest)
      .pipe(
        timeout(30000),
        retry({ count: 2, delay: 1000 }),
        tap((response) => {
          // Add both messages to history
          this.addMessageToHistory({
            role: 'user',
            content: query,
            timestamp: new Date(),
          });
          this.addMessageToHistory({
            role: 'assistant',
            content: response.content,
            timestamp: new Date(),
            metadata: response.metadata,
          });
        }),
        catchError((error) => this.handleError(error)),
      );
  }

  /**
   * Get chat history for current session.
   */
  getChatHistory(): ChatMessage[] {
    return [...this.messageHistory];
  }

  /**
   * Get semantic cache status.
   */
  getCacheStatus(): Observable<any> {
    return this.http.get(`${this.apiUrl}/chat/cache/status`).pipe(
      tap((status) => this.cacheStatusSubject.next(status)),
      catchError((error) => {
        console.warn('Cache status fetch failed:', error);
        return throwError(() => error);
      }),
    );
  }

  /**
   * Clear semantic cache (admin).
   */
  clearCache(): Observable<any> {
    return this.http.post(`${this.apiUrl}/chat/cache/clear`, {}).pipe(
      tap(() => {
        this.loadCacheStatus();
      }),
      catchError((error) => this.handleError(error)),
    );
  }

  /**
   * Get health status.
   */
  getHealthStatus(): Observable<any> {
    return this.http.get(`${this.apiUrl}/health/detailed`).pipe(
      timeout(5000),
      catchError((error) => throwError(() => error)),
    );
  }

  /**
   * Create SSE connection to streaming endpoint.
   *
   * Returns Promise<Response> for manual stream management.
   */
  private connectSSE(chatRequest: ChatRequest): Promise<Response> {
    return fetch(`${this.apiUrl}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(chatRequest),
    });
  }

  /**
   * Add message to local history and notify subscribers.
   */
  private addMessageToHistory(message: ChatMessage): void {
    const fullMessage: ChatMessage = {
      ...message,
      timestamp: message.timestamp || new Date(),
    };

    this.messageHistory.push(fullMessage);
    this.messageSubject.next(fullMessage);

    // Persist to localStorage
    this.saveMessageHistory();
  }

  /**
   * Save message history to localStorage.
   */
  private saveMessageHistory(): void {
    try {
      const serialized = this.messageHistory.map((msg) => ({
        ...msg,
        timestamp: msg.timestamp.toISOString(),
      }));
      localStorage.setItem(
        `chat_history_${this.sessionId}`,
        JSON.stringify(serialized),
      );
    } catch (error) {
      console.warn('Failed to save message history:', error);
    }
  }

  /**
   * Load message history from localStorage.
   */
  private loadMessageHistory(): void {
    try {
      const stored = localStorage.getItem(`chat_history_${this.sessionId}`);
      if (stored) {
        this.messageHistory = JSON.parse(stored).map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp),
        }));
      }
    } catch (error) {
      console.warn('Failed to load message history:', error);
    }
  }

  /**
   * Load cache status from server.
   */
  private loadCacheStatus(): void {
    this.getCacheStatus().subscribe(
      (status) => this.cacheStatusSubject.next(status),
      (error) => console.warn('Cache status load failed:', error),
    );
  }

  /**
   * Handle stream errors.
   */
  private handleStreamError(errorMessage: string): void {
    console.error('Stream error:', errorMessage);
    this.errorSubject.next(errorMessage);
    this.streamingSubject.next(false);
  }

  /**
   * Handle HTTP errors.
   */
  private handleError(error: HttpErrorResponse): Observable<never> {
    let errorMessage = 'An error occurred';

    if (error.error instanceof ErrorEvent) {
      // Client-side error
      errorMessage = `Client error: ${error.error.message}`;
    } else {
      // Server-side error
      errorMessage = `Server error: ${error.status} - ${error.message}`;
    }

    this.errorSubject.next(errorMessage);
    console.error('HTTP error:', errorMessage);

    return throwError(() => new Error(errorMessage));
  }

  /**
   * Generate or retrieve session ID.
   */
  private generateSessionId(): string {
    let sessionId = sessionStorage.getItem('session_id');
    if (!sessionId) {
      sessionId = `sess-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      sessionStorage.setItem('session_id', sessionId);
    }
    return sessionId;
  }

  /**
   * Get current user ID (from auth service or localStorage).
   */
  private getCurrentUserId(): string {
    // TODO: Get from auth service
    return localStorage.getItem('user_id') || 'anonymous';
  }

  /**
   * Clear all chat data.
   */
  clearHistory(): void {
    this.messageHistory = [];
    localStorage.removeItem(`chat_history_${this.sessionId}`);
  }
}
