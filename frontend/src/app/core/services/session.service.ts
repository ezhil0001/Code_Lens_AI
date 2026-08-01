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
import { Observable, BehaviorSubject, throwError, timeout } from 'rxjs';
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
  private apiUrl = 'http://localhost:8000/api/v1';       // auth endpoints
  private chatApiUrl = 'http://localhost:8000/api/v2/chat'; // chat endpoints (V2)
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
   * 2. If yes, use it (no backend validation - that's done at login)
   * 3. If no, create new session when user logs in
   */
  initializeSession(): void {
    console.log('🔧 [SessionService] Initializing session...');

    const storedSessionId = sessionStorage.getItem('chat_session_id');
    const storedUserId = localStorage.getItem('chat_user_id');

    if (storedSessionId && storedUserId) {
      console.log(`📌 Found stored session: ${storedSessionId}`);
      this.sessionId.next(storedSessionId);
      this.userId.next(storedUserId);
      
      // Try to recover history if backend is available (non-blocking)
      this.recoverChatHistory(storedSessionId, storedUserId).subscribe({
        next: () => console.log('✅ Chat history recovered'),
        error: () => console.warn('⚠️ Could not recover chat history (backend may be down)')
      });
    } else {
      console.log('📝 No stored session found - waiting for login');
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
   * Validate session with backend (with 3 second timeout)
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
        timeout(3000), // 3 second timeout
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
   * Fetch chat history from backend for a session (with 3 second timeout)
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
        `${this.chatApiUrl}/history/${sessionId}`,
        {
          headers: {
            'X-User-ID': userId,
            'X-Session-ID': sessionId,
          },
        }
      )
      .pipe(
        timeout(3000), // 3 second timeout
        tap((response) => {
          console.log(
            `✅ Recovered ${response.messages.length} messages from backend`
          );
          console.log('📊 Chat History:', response.messages);
        }),
        catchError((error: any) => {
          if (error.name === 'TimeoutError') {
            console.warn('⚠️  History recovery timeout (backend slow or unreachable)');
            return throwError(() => new Error('Timeout'));
          }
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
