/**
 * Legacy chat component — backs the /components/chat route used by the v1
 * ChatService + SessionService stack.  New feature work goes into
 * features/chat/components/chat.component.ts which uses the v2 agent stream.
 *
 * This component will be removed once the v2 chat is the only route.
 */

import {
  Component,
  OnInit,
  ViewChild,
  ElementRef,
  OnDestroy,
} from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { ChatService, ChatMessage } from '../services/chat.service';
import { SessionService } from '../core/services/session.service';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { MarkdownService } from 'ngx-markdown';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule } from '@angular/forms';
import { MarkdownModule } from 'ngx-markdown';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, MarkdownModule],
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.scss'],
})
export class ChatComponent implements OnInit, OnDestroy {
  // Chat state
  messages: ChatMessage[] = [];
  chatForm!: FormGroup;
  isLoading = false;
  currentStreamContent = '';
  selectedSource: any = null;

  // UI state
  cacheStatus: any = null;
  showSources = false;
  showMetadata = false;
  healthStatus: any = null;
  isRecoveringHistory = false;

  // Refs
  @ViewChild('messagesContainer') messagesContainer!: ElementRef;
  @ViewChild('userInput') userInput!: ElementRef;

  private destroy$ = new Subject<void>();

  constructor(
    private chatService: ChatService,
    private sessionService: SessionService,
    private formBuilder: FormBuilder,
    public markdownService: MarkdownService,
  ) {
    this.setupForm();
  }

  ngOnInit(): void {
    console.log(
      '🔄 [ChatComponent] ngOnInit - loading initial state and recovering history',
    );
    this.loadInitialState();
    this.recoverChatHistory();
    this.subscribeToServiceUpdates();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  /**
   * Setup reactive form for chat input.
   */
  private setupForm(): void {
    this.chatForm = this.formBuilder.group({
      query: [
        '',
        [
          Validators.required,
          Validators.minLength(1),
          Validators.maxLength(5000),
        ],
      ],
    });
  }

  /**
   * Load initial state from service and server.
   */
  private loadInitialState(): void {
    console.log('📌 [ChatComponent] Loading initial state');

    // Load chat history
    this.messages = this.chatService.getChatHistory();
    console.log(`Loaded ${this.messages.length} messages from chat service`);

    // Load cache status
    this.chatService
      .getCacheStatus()
      .pipe(takeUntil(this.destroy$))
      .subscribe(
        (status) => (this.cacheStatus = status),
        (error) => console.warn('Cache status error:', error),
      );

    // Load health status
    this.chatService
      .getHealthStatus()
      .pipe(takeUntil(this.destroy$))
      .subscribe(
        (status) => (this.healthStatus = status),
        (error) => console.warn('Health status error:', error),
      );
  }

  /**
   * Recover chat history from backend after hard refresh
   */
  private recoverChatHistory(): void {
    const sessionId = this.sessionService.getSessionId();
    const userId = this.sessionService.getUserId();

    if (!sessionId || !userId) {
      console.warn('⚠️  No session info available for history recovery');
      return;
    }

    console.log(
      `🔄 [ChatComponent] Attempting to recover history for session: ${sessionId}`,
    );
    this.isRecoveringHistory = true;

    this.sessionService
      .recoverChatHistory(sessionId, userId)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (historyResponse) => {
          console.log(
            `✅ History recovered: ${historyResponse.messages.length} messages`,
          );

          // Replace local messages with recovered history
          this.messages = historyResponse.messages.map((msg) => ({
            ...msg,
            timestamp: new Date(msg.timestamp),
          }));

          this.isRecoveringHistory = false;
          this.scrollToBottom();
        },
        error: (error) => {
          console.warn(
            '⚠️  History recovery failed (first chat or session expired):',
            error,
          );
          this.isRecoveringHistory = false;
        },
      });
  }

  /**
   * Subscribe to service updates.
   */
  private subscribeToServiceUpdates(): void {
    // New messages
    this.chatService.messages$
      .pipe(takeUntil(this.destroy$))
      .subscribe((message) => {
        this.messages.push(message);
        this.scrollToBottom();
      });

    // Streaming status
    this.chatService.isStreaming$
      .pipe(takeUntil(this.destroy$))
      .subscribe((isStreaming) => (this.isLoading = isStreaming));

    // Errors
    this.chatService.errors$
      .pipe(takeUntil(this.destroy$))
      .subscribe((error) => {
        console.error('Chat error:', error);
        this.addErrorMessage(error);
      });

    // Cache status updates
    this.chatService.cacheStatus$
      .pipe(takeUntil(this.destroy$))
      .subscribe((status) => (this.cacheStatus = status));
  }

  /**
   * Handle form submission (send chat message).
   */
  onSendMessage(): void {
    if (!this.chatForm.valid || this.isLoading) {
      return;
    }

    const query = this.chatForm.get('query')?.value?.trim();
    if (!query) {
      return;
    }

    // Clear form
    this.chatForm.reset();
    this.userInput.nativeElement.focus();

    // Stream response
    this.streamResponse(query);
  }

  /**
   * Stream response from backend.
   */
  private streamResponse(query: string): void {
    this.isLoading = true;
    this.currentStreamContent = '';

    this.chatService
      .streamChat(query)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        // AFTER:
        next: (token) => {
          this.currentStreamContent += token;
          // Extract answer if JSON wrapped
          this.currentStreamContent = this.extractAnswer(
            this.currentStreamContent,
          );
          this.updateStreamContent();
        },
        error: (error) => {
          console.error('Stream error:', error);
          this.isLoading = false;
        },
        complete: () => {
          this.isLoading = false;
          // Log final response when streaming completes
          console.log('\n' + '='.repeat(80));
          console.log('✅ CHAT COMPONENT - STREAM COMPLETED');
          console.log('='.repeat(80));
          console.log('🎯 Query:', query);
          console.log(
            '📄 Final Response Length:',
            this.currentStreamContent.length,
            'chars',
          );
          console.log(
            '📊 Word Count:',
            this.currentStreamContent.split(/\s+/).filter((w) => w.length > 0)
              .length,
            'words',
          );
          console.log('-'.repeat(80));
          console.log(this.currentStreamContent);
          console.log('-'.repeat(80));
          console.log('='.repeat(80) + '\n');

          this.currentStreamContent = '';
          this.scrollToBottom();
        },
      });
  }

  private extractAnswer(content: string): string {
    try {
      const trimmed = content.trim();
      if (trimmed.startsWith('{') || trimmed.startsWith('```json')) {
        const jsonStr = trimmed
          .replace(/^```json\s*/, '')
          .replace(/\s*```$/, '')
          .trim();
        const parsed = JSON.parse(jsonStr);
        if (parsed.answer) return parsed.answer;
      }
    } catch {
      /* still streaming */
    }
    return content;
  }

  /**
   * Update stream content in UI (re-render markdown).
   */
  private updateStreamContent(): void {
    // Force change detection
    this.messages = [...this.messages];
    this.scrollToBottom();
  }

  /**
   * Add error message to chat.
   */
  private addErrorMessage(error: string): void {
    this.messages.push({
      role: 'assistant',
      content: `❌ **Error**: ${error}`,
      timestamp: new Date(),
    });
    this.scrollToBottom();
  }

  /**
   * Get CSS class for message alignment.
   */
  getMessageClass(role: 'user' | 'assistant'): string {
    return role === 'user' ? 'message-user' : 'message-assistant';
  }

  /**
   * Copy code block to clipboard.
   */
  copyCode(code: string): void {
    navigator.clipboard
      .writeText(code)
      .then(() => {
        // Show toast notification
        alert('Code copied to clipboard!');
      })
      .catch((err) => {
        console.error('Copy failed:', err);
      });
  }

  /**
   * Show source details.
   */
  showSourceDetails(source: any): void {
    this.selectedSource = source;
    this.showSources = true;
  }

  /**
   * Get source metadata as formatted display.
   */
  getSourceMetadata(source: any): string {
    const lines = [];

    if (source.source) {
      lines.push(`📄 File: ${source.source}`);
    }

    if (source.score) {
      lines.push(`📊 Relevance: ${(source.score * 100).toFixed(1)}%`);
    }

    if (source.line_number) {
      lines.push(`📍 Line: ${source.line_number}`);
    }

    if (source.metadata?.size) {
      lines.push(`📦 Size: ${source.metadata.size} bytes`);
    }

    return lines.join('\n');
  }

  /**
   * Get all unique sources from current message.
   */
  getSources(message: ChatMessage): any[] {
    return message.metadata?.retrieval_sources || [];
  }

  /**
   * Check if message has sources.
   */
  hasSources(message: ChatMessage): boolean {
    return (message.metadata?.retrieval_sources?.length || 0) > 0;
  }

  /**
   * Get health status indicator.
   */
  getHealthColor(status: string): string {
    switch (status) {
      case 'healthy':
        return 'text-success';
      case 'degraded':
        return 'text-warning';
      case 'unhealthy':
        return 'text-danger';
      default:
        return 'text-secondary';
    }
  }

  /**
   * Toggle sources panel.
   */
  toggleSources(): void {
    this.showSources = !this.showSources;
  }

  /**
   * Toggle metadata panel.
   */
  toggleMetadata(): void {
    this.showMetadata = !this.showMetadata;
  }

  /**
   * Clear chat history.
   */
  clearHistory(): void {
    if (confirm('Clear all chat history? This cannot be undone.')) {
      this.chatService.clearHistory();
      this.messages = [];
    }
  }

  /**
   * Clear semantic cache (admin).
   */
  clearCache(): void {
    if (
      confirm('Clear semantic cache? This will clear all cached responses.')
    ) {
      this.chatService
        .clearCache()
        .pipe(takeUntil(this.destroy$))
        .subscribe(
          () => {
            console.log('Cache cleared');
            this.loadInitialState();
          },
          (error) => console.error('Cache clear failed:', error),
        );
    }
  }

  /**
   * Scroll messages container to bottom.
   */
  private scrollToBottom(): void {
    setTimeout(() => {
      if (this.messagesContainer) {
        const container = this.messagesContainer.nativeElement;
        container.scrollTop = container.scrollHeight;
      }
    }, 0);
  }

  /**
   * Get message time display.
   */
  getTimeDisplay(timestamp: Date): string {
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
  }

  onEnterKey(event: Event): void {
    const e = event as KeyboardEvent;

    if (!e.shiftKey) {
      event.preventDefault();
      this.onSendMessage();
    }
  }
}
