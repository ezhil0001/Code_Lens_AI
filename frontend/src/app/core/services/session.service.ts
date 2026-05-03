/**
 * Session Service - Manages chat session persistence and recovery
 * 
 * Features:
 * 1. Session Storage: Persist session_id and user_id
 * 2. History Recovery: Fetch chat history from backend on hard refresh
 * 3. Session Validation: Check if session is still valid on app init
 */

import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, BehaviorSubject, throwError } from 'rxjs';
import { catchError, tap } from 'rxjs/operators';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  metadata?: any;
}

export interface HistoryResponse {
  session_id: string;
  messages: ChatMessage[];
  user_id: string;
  created_at: string;
}

@Injectable({
  providedIn: 'root',
})
export class SessionService {
  private apiUrl = 'http://localhost:8000/api/v1';
  private sessionId = new BehaviorSubject<string | null>(null);
  private userId = new BehaviorSubject<string | null>(null);
  public sessionId$ = this.sessionId.asObservable();
  public userId$ = this.userId.asObservable();

  constructor(private http: HttpClient) {
    this.initializeSession();
  }

  /**
   * Initialize session on app startup
   * 1. Check if session exists in storage
   * 2. If yes, validate with backend
   * 3. If valid, recover chat history
   */
  initializeSession(): void {
    console.log('🔧 [SessionService] Initializing session...');

    const storedSessionId = sessionStorage.getItem('chat_session_id');
    const storedUserId = localStorage.getItem('chat_user_id');

    if (storedSessionId && storedUserId) {
      console.log(`📌 Found stored session: ${storedSessionId}`);
      this.sessionId.next(storedSessionId);
      this.userId.next(storedUserId);

      // Validate session with backend
      this.validateSessionWithBackend(storedSessionId, storedUserId).subscribe({
        next: (isValid) => {
          if (isValid) {
            console.log('✅ Session validated with backend');
            // Optionally recover history here
            this.recoverChatHistory(storedSessionId, storedUserId);
          } else {
            console.warn('⚠️  Session validation failed, creating new session');
            this.createNewSession();
          }
        },
        error: (error) => {
          console.warn('⚠️  Session validation error:', error);
          // Fall back to offline mode or create new session
          this.createNewSession();
        },
      });
    } else {
      console.log('📝 No stored session found, creating new session');
      this.createNewSession();
    }
  }

  /**
   * Create a new session
   */
  createNewSession(): void {
    const sessionId = this.generateSessionId();
    const userId = this.generateUserId();

    console.log(`✨ Creating new session: ${sessionId}`);

    sessionStorage.setItem('chat_session_id', sessionId);
    localStorage.setItem('chat_user_id', userId);

    this.sessionId.next(sessionId);
    this.userId.next(userId);
  }

  /**
   * Validate session with backend
   * This ensures the session_id is legitimate and user_id matches
   */
  validateSessionWithBackend(
    sessionId: string,
    userId: string
  ): Observable<{ valid: boolean }> {
    return this.http
      .post<{ valid: boolean }>(
        `${this.apiUrl}/auth/validate-session`,
        {
          session_id: sessionId,
          user_id: userId,
        }
      )
      .pipe(
        tap((response) => {
          console.log('✅ Backend validation response:', response);
        }),
        catchError((error) => {
          console.error('❌ Backend validation failed:', error);
          return throwError(() => error);
        })
      );
  }

  /**
   * Fetch chat history from backend for a session
   * This recovers lost messages after a hard refresh
   */
  recoverChatHistory(
    sessionId: string,
    userId: string
  ): Observable<HistoryResponse> {
    console.log(
      `🔄 [SessionService] Recovering chat history for session: ${sessionId}`
    );

    return this.http
      .get<HistoryResponse>(
        `${this.apiUrl}/chat/history/${sessionId}`,
        {
          headers: {
            'X-User-ID': userId,
            'X-Session-ID': sessionId,
          },
        }
      )
      .pipe(
        tap((response) => {
          console.log(
            `✅ Recovered ${response.messages.length} messages from backend`
          );
          console.log('📊 Chat History:', response.messages);
        }),
        catchError((error: HttpErrorResponse) => {
          if (error.status === 404) {
            console.warn('⚠️  No history found for this session (first message)');
            return throwError(() => new Error('No history found'));
          } else {
            console.error('❌ Failed to fetch chat history:', error);
            return throwError(() => error);
          }
        })
      );
  }

  /**
   * Get current session ID
   */
  getSessionId(): string | null {
    return this.sessionId.value;
  }

  /**
   * Get current user ID
   */
  getUserId(): string | null {
    return this.userId.value;
  }

  /**
   * Clear session (on logout)
   */
  clearSession(): void {
    console.log('🔓 Clearing session');
    sessionStorage.removeItem('chat_session_id');
    localStorage.removeItem('chat_user_id');
    this.sessionId.next(null);
    this.userId.next(null);
  }

  /**
   * Generate a unique session ID
   */
  private generateSessionId(): string {
    return `sess-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Generate a unique user ID (for anonymous users)
   */
  private generateUserId(): string {
    const existingUserId = localStorage.getItem('chat_user_id');
    if (existingUserId) {
      return existingUserId;
    }
    const userId = `anon-${crypto.randomUUID()}`;
    localStorage.setItem('chat_user_id', userId);
    return userId;
  }
}
