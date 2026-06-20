import { Component, OnInit, OnDestroy, ViewChild, ElementRef, AfterViewChecked, ChangeDetectorRef, ChangeDetectionStrategy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import { Subscription } from 'rxjs';
import { MarkdownViewerComponent } from '../../../shared/components/markdown-viewer/markdown-viewer.component';
import { CitationBadgeComponent } from '../../../shared/components/citation-badge/citation-badge.component';
import { AIStreamService } from '../../../core/services/ai-stream.service';
import { AgentStreamService, AgentActivityEntry, HILInterruptPayload } from '../../../core/services/agent-stream.service';
import { IngestService } from '../../../core/services/ingest.service';
import { Message, Citation } from '../../../data/models/message.model';
import { AgentActivityComponent } from './agent-activity.component';
import { CheckpointTimelineComponent } from './checkpoint-timeline.component';

/**
 * ChatComponent
 * Main UI for CodeLens_AI chat interface
 * 
 * Features:
 * - Message list with auto-scroll
 * - Input bar with send button
 * - Real-time streaming with visual feedback
 * - Citations display below streamed content
 * - Stop button during streaming
 */
@Component({
  selector: 'app-chat',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [
    CommonModule,
    FormsModule,
    HttpClientModule,
    MarkdownViewerComponent,
    CitationBadgeComponent,
    AgentActivityComponent,
    CheckpointTimelineComponent,
  ],
  template: `
    <div class="chat-container">
      <!-- Header -->
      <div class="chat-header">
        <h1>CodeLens AI</h1>
        <p class="subtitle">Enterprise RAG for Your Codebase</p>
        <div class="header-actions">
          <button (click)="toggleActivityPanel()" class="header-icon-btn" title="Agent Activity"
                  [class.active]="showActivityPanel">
            ⚡ Activity
          </button>
          <button (click)="toggleTimeline()" class="header-icon-btn" title="Checkpoint Timeline"
                  [class.active]="showTimeline">
            ⏱ Timeline
          </button>
          <button (click)="toggleUploadPanel()" class="upload-toggle-btn" title="Upload documents">
            📤 Upload
          </button>
        </div>
      </div>

      <!-- HIL Approval Banner -->
      <div *ngIf="hilInterrupt" class="hil-banner">
        <div class="hil-inner">
          <div class="hil-icon">⚠️</div>
          <div class="hil-body">
            <div class="hil-title">Human Review Required</div>
            <div class="hil-reason">{{ hilInterrupt.reason }}</div>
            <div class="hil-actions">
              <input
                type="text"
                [(ngModel)]="hilInput"
                placeholder="Optional: add context or guidance…"
                class="hil-input"
              />
              <button class="hil-approve-btn" (click)="onHILApprove()">✅ Approve</button>
              <button class="hil-reject-btn" (click)="onHILReject()">❌ Reject</button>
            </div>
          </div>
        </div>
      </div>

      <!-- Upload Panel -->
      <div *ngIf="showUploadPanel" class="upload-panel">
        <div class="upload-content">
          <h3>📁 Ingest Documents</h3>
          <p class="upload-help">Upload files or paste a URL to add to your knowledge base</p>
          
          <!-- File Upload -->
          <div class="upload-section">
            <label class="file-input-label">
              <input type="file" 
                     multiple 
                     accept=".pdf,.txt,.md,.py,.ts,.js,.java,.cpp,.c,.h,.go,.rs" 
                     (change)="onFilesSelected($event)"
                     [disabled]="isIngesting"
                     class="file-input">
              <span class="file-input-text">Click to select files</span>
            </label>
            <p class="file-types">Supported: PDF, TXT, MD, PY, TS, JS, JAVA, CPP, C, H, GO, RS</p>
          </div>

          <!-- OR Divider -->
          <div class="divider">or</div>

          <!-- URL Input -->
          <div class="url-section">
            <input type="url" 
                   [(ngModel)]="uploadUrl" 
                   placeholder="Paste a URL to ingest"
                   [disabled]="isIngesting"
                   class="url-input">
            <button (click)="ingestUrl()" 
                    [disabled]="!uploadUrl.trim() || isIngesting"
                    class="ingest-btn">
              {{ isIngesting ? 'Ingesting...' : 'Ingest URL' }}
            </button>
          </div>

          <!-- Status Message -->
          <div *ngIf="ingestionStatus" class="ingestion-status" [class.success]="ingestionStatus.includes('✓')">
            {{ ingestionStatus }}
          </div>

          <!-- Close Button -->
          <button (click)="showUploadPanel = false" class="close-upload-btn">✕ Close</button>
        </div>
      </div>

      <!-- Main split layout: chat + side panels -->
      <div class="main-split">
        <!-- Messages Area -->
        <div class="messages-area" #messagesContainer>
        <div class="messages-list">
          <!-- Empty State -->
          <div *ngIf="messages.length === 0" class="empty-state">
            <div class="empty-icon">💬</div>
            <h2>Start a Conversation</h2>
            <p>Ask me anything about your codebase</p>
            <div class="example-queries">
              <button *ngFor="let query of exampleQueries"
                      class="example-btn"
                      (click)="sendMessage(query)">
                {{ query }}
              </button>
            </div>
          </div>

          <!-- Messages -->
          <div *ngFor="let message of messages; let last = last; trackBy: trackByMessageId"
               [class.message-row]="true"
               [class.user]="message.role === 'user'"
               [class.assistant]="message.role === 'assistant'">

            <!-- User Message -->
            <div *ngIf="message.role === 'user'" class="message user-message">
              <div class="message-content">{{ message.content }}</div>
            </div>

            <!-- Assistant Message -->
            <div *ngIf="message.role === 'assistant'" class="message assistant-message">
              <div class="message-content">
                <!-- During streaming: plain text avoids per-token markdown re-parse -->
                <pre *ngIf="message.isStreaming" class="streaming-text">{{ message.content }}<span class="cursor">▌</span></pre>

                <!-- After streaming: full markdown with syntax highlighting -->
                <app-markdown-viewer
                  *ngIf="!message.isStreaming"
                  [content]="message.content">
                </app-markdown-viewer>

                <!-- Streaming indicator (three dots below text while loading) -->
                <div *ngIf="message.isStreaming && !message.content" class="streaming-indicator">
                  <span class="dot"></span>
                  <span class="dot"></span>
                  <span class="dot"></span>
                </div>
              </div>

              <!-- Metadata -->
              <div *ngIf="message.tokenCount" class="message-meta">
                <span>🔤 {{ message.tokenCount }} tokens</span>
                <span *ngIf="message.processingTimeMs">
                  ⏱️ {{ message.processingTimeMs }}ms
                </span>
              </div>

              <!-- Citations -->
              <div *ngIf="message.citations && message.citations.length > 0" class="citations">
                <div class="citations-label">📚 Sources:</div>
                <app-citation-badge
                  *ngFor="let citation of message.citations"
                  [citation]="citation">
                </app-citation-badge>
              </div>
            </div>

            <!-- Error Message -->
            <div *ngIf="message.error" class="message error-message">
              <div class="error-icon">⚠️</div>
              <div class="error-content">
                <strong>Error:</strong> {{ message.error }}
              </div>
            </div>
          </div>
        </div>
      </div>

        <!-- Agent Activity Side Panel -->
        <div *ngIf="showActivityPanel" class="side-panel">
          <app-agent-activity
            [activities]="agentActivities"
            [isStreaming]="isLoading"
            (checkpointSelected)="onCheckpointBadgeClick($event)">
          </app-agent-activity>
        </div>

        <!-- Checkpoint Timeline Side Panel -->
        <div *ngIf="showTimeline" class="side-panel">
          <app-checkpoint-timeline
            [sessionId]="currentSessionId"
            [visible]="showTimeline"
            (replayRequested)="onReplayCheckpoint($event)"
            (branchCreated)="onBranchCreated($event)"
            (closed)="showTimeline = false">
          </app-checkpoint-timeline>
        </div>
      </div><!-- end .main-split -->

      <!-- Input Area -->
      <div class="input-area">
        <div class="input-row">
          <textarea
            [(ngModel)]="userInput"
            (keydown.enter)="onEnterKey($event)"
            placeholder="Ask me about your code... (Ctrl+Enter to send)"
            class="message-input"
            [disabled]="isLoading"
            rows="1">
          </textarea>

          <button
            *ngIf="!isLoading"
            (click)="sendMessage()"
            [disabled]="!userInput.trim()"
            class="send-btn">
            Send
          </button>

          <button
            *ngIf="isLoading"
            (click)="stopStreaming()"
            class="stop-btn">
            Stop
          </button>
        </div>

        <!-- Settings -->
        <div class="settings-row">
          <label>
            <input type="checkbox" [(ngModel)]="useHybridSearch">
            <span>Hybrid Search (BM25 + Vector)</span>
          </label>
          <label>
            <input type="checkbox" [(ngModel)]="showCitations" checked>
            <span>Show Citations</span>
          </label>
          <label>
            <input type="checkbox" [(ngModel)]="useV2Stream">
            <span>Agent Stream (v2)</span>
          </label>
          <label *ngIf="useV2Stream">
            <input type="checkbox" [(ngModel)]="hilEnabled">
            <span>HIL Review</span>
          </label>
        </div>
      </div>
    </div>
  `,
  styles: [`
    :host {
      display: flex;
      flex-direction: column;
      height: 100vh;
      background: #0f1419;
      color: #e0e0e0;
      font-family: system-ui, -apple-system, sans-serif;
    }

    .chat-container {
      display: flex;
      flex-direction: column;
      height: 100%;
      max-width: 900px;
      margin: 0 auto;
      width: 100%;
      box-shadow: 0 0 30px rgba(0,0,0,0.8);
    }

    /* Header */
    .chat-header {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      padding: 24px;
      text-align: center;
      border-bottom: 1px solid #374151;
    }

    .chat-header h1 {
      margin: 0;
      font-size: 24px;
      font-weight: 700;
      color: white;
    }

    .subtitle {
      margin: 6px 0 0;
      font-size: 13px;
      color: rgba(255,255,255,0.8);
    }

    .upload-toggle-btn {
      position: absolute;
      top: 24px;
      right: 24px;
      background: rgba(255,255,255,0.2);
      border: 1px solid rgba(255,255,255,0.3);
      color: white;
      padding: 8px 16px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 12px;
      transition: all 0.2s;
    }

    .upload-toggle-btn:hover {
      background: rgba(255,255,255,0.3);
      border-color: rgba(255,255,255,0.5);
    }

    /* Upload Panel */
    .upload-panel {
      background: #1f2937;
      border-bottom: 2px solid #667eea;
      padding: 24px;
      animation: slideDown 0.3s ease-out;
    }

    @keyframes slideDown {
      from {
        opacity: 0;
        transform: translateY(-20px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    .upload-content {
      max-width: 600px;
      margin: 0 auto;
    }

    .upload-content h3 {
      margin: 0 0 12px;
      color: #e0e0e0;
      font-size: 16px;
    }

    .upload-help {
      margin: 0 0 20px;
      color: #9ca3af;
      font-size: 13px;
    }

    .upload-section {
      margin-bottom: 20px;
    }

    .file-input-label {
      display: block;
      padding: 20px;
      border: 2px dashed #667eea;
      border-radius: 8px;
      text-align: center;
      cursor: pointer;
      background: rgba(102, 126, 234, 0.05);
      transition: all 0.2s;
    }

    .file-input-label:hover {
      background: rgba(102, 126, 234, 0.1);
      border-color: #5568d3;
    }

    .file-input {
      display: none;
    }

    .file-input-text {
      color: #667eea;
      font-weight: 600;
      font-size: 14px;
    }

    .file-types {
      margin: 8px 0 0;
      color: #6b7280;
      font-size: 12px;
    }

    .divider {
      text-align: center;
      color: #6b7280;
      margin: 20px 0;
      position: relative;
    }

    .url-section {
      display: flex;
      gap: 8px;
      margin-bottom: 16px;
    }

    .url-input {
      flex: 1;
      background: #111827;
      border: 1px solid #374151;
      color: #e0e0e0;
      padding: 10px 12px;
      border-radius: 6px;
      font-size: 13px;
    }

    .url-input:focus {
      outline: none;
      border-color: #667eea;
      box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
    }

    .ingest-btn {
      background: #667eea;
      color: white;
      border: none;
      padding: 10px 20px;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 600;
      transition: all 0.2s;
    }

    .ingest-btn:hover:not(:disabled) {
      background: #5568d3;
    }

    .ingest-btn:disabled {
      background: #4b5563;
      cursor: not-allowed;
      opacity: 0.6;
    }

    .ingestion-status {
      padding: 12px;
      border-radius: 6px;
      background: #7f1d1d;
      color: #fecaca;
      font-size: 13px;
      margin-bottom: 12px;
    }

    .ingestion-status.success {
      background: #15803d;
      color: #86efac;
    }

    .close-upload-btn {
      width: 100%;
      background: #374151;
      color: #e0e0e0;
      border: none;
      padding: 8px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 13px;
      transition: all 0.2s;
    }

    .close-upload-btn:hover {
      background: #4b5563;
    }

    /* Messages Area */
    .messages-area {
      flex: 1;
      overflow-y: auto;
      padding: 20px;
      background: #0f1419;
    }

    .messages-list {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    /* Empty State */
    .empty-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100%;
      color: #6b7280;
      text-align: center;
    }

    .empty-icon {
      font-size: 48px;
      margin-bottom: 12px;
    }

    .empty-state h2 {
      margin: 0;
      color: #d1d5db;
      font-size: 20px;
    }

    .empty-state p {
      margin: 8px 0 20px;
      font-size: 14px;
    }

    .example-queries {
      display: flex;
      flex-direction: column;
      gap: 8px;
      width: 100%;
      max-width: 400px;
    }

    .example-btn {
      background: #1f2937;
      border: 1px solid #374151;
      color: #9ca3af;
      padding: 10px 14px;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s;
      font-size: 13px;
      text-align: left;
    }

    .example-btn:hover {
      background: #374151;
      color: #d1d5db;
      border-color: #4b5563;
    }

    /* Messages */
    .message-row {
      display: flex;
      gap: 12px;
      animation: slideIn 0.3s ease-out;
    }

    .message-row.user {
      justify-content: flex-end;
    }

    @keyframes slideIn {
      from {
        opacity: 0;
        transform: translateY(10px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    .message {
      max-width: 70%;
      padding: 12px 16px;
      border-radius: 8px;
      line-height: 1.5;
    }

    .user-message {
      background: #667eea;
      color: white;
      border-bottom-right-radius: 0;
    }

    .assistant-message {
      background: #1f2937;
      border: 1px solid #374151;
      border-bottom-left-radius: 0;
    }

    .error-message {
      background: #7f1d1d;
      border: 1px solid #991b1b;
      border-radius: 8px;
      display: flex;
      gap: 12px;
      align-items: flex-start;
    }

    .error-icon {
      font-size: 20px;
      flex-shrink: 0;
    }

    .error-content {
      color: #fecaca;
      font-size: 14px;
    }

    /* Streaming */
    .streaming-text {
      font-family: system-ui, -apple-system, sans-serif;
      font-size: 14px;
      white-space: pre-wrap;
      word-break: break-word;
      background: transparent;
      border: none;
      padding: 0;
      margin: 0;
      color: inherit;
      line-height: 1.6;
    }

    .cursor {
      display: inline-block;
      animation: blink 1s step-end infinite;
      color: #667eea;
      margin-left: 1px;
    }

    @keyframes blink {
      0%, 100% { opacity: 1; }
      50%       { opacity: 0; }
    }

    .streaming-indicator {
      display: flex;
      gap: 4px;
      margin-top: 8px;
    }

    .dot {
      display: inline-block;
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: #667eea;
      animation: pulse 1.4s infinite;
    }

    .dot:nth-child(2) { animation-delay: 0.2s; }
    .dot:nth-child(3) { animation-delay: 0.4s; }

    @keyframes pulse {
      0%, 100% { opacity: 0.3; }
      50% { opacity: 1; }
    }

    /* Metadata & Citations */
    .message-meta {
      margin-top: 8px;
      padding-top: 8px;
      border-top: 1px solid #374151;
      font-size: 12px;
      color: #6b7280;
      display: flex;
      gap: 12px;
    }

    .citations {
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid #374151;
    }

    .citations-label {
      font-size: 12px;
      color: #6b7280;
      margin-bottom: 8px;
    }

    /* Input Area */
    .input-area {
      background: #1f2937;
      border-top: 1px solid #374151;
      padding: 16px;
      gap: 12px;
    }

    .input-row {
      display: flex;
      gap: 8px;
      align-items: flex-end;
    }

    .message-input {
      flex: 1;
      background: #111827;
      border: 1px solid #374151;
      color: #e0e0e0;
      padding: 10px 12px;
      border-radius: 6px;
      font-size: 14px;
      font-family: system-ui, -apple-system, sans-serif;
      resize: none;
      max-height: 200px;
    }

    .message-input:focus {
      outline: none;
      border-color: #667eea;
      box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
    }

    .message-input:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .send-btn, .stop-btn {
      padding: 10px 20px;
      background: #667eea;
      color: white;
      border: none;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 600;
      transition: all 0.2s;
    }

    .send-btn:hover:not(:disabled) {
      background: #5568d3;
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }

    .send-btn:disabled {
      background: #4b5563;
      cursor: not-allowed;
      opacity: 0.6;
    }

    .stop-btn {
      background: #ef4444;
    }

    .stop-btn:hover {
      background: #dc2626;
    }

    /* Settings */
    .settings-row {
      display: flex;
      gap: 20px;
      margin-top: 12px;
      font-size: 13px;
      flex-wrap: wrap;
    }

    .settings-row label {
      display: flex;
      align-items: center;
      gap: 6px;
      cursor: pointer;
      color: #9ca3af;
    }

    .settings-row input[type="checkbox"] {
      width: 16px;
      height: 16px;
      cursor: pointer;
    }

    /* ── Header actions ─────────────────────────────────────────── */
    .header-actions {
      position: absolute;
      top: 16px;
      right: 16px;
      display: flex;
      gap: 6px;
    }

    .header-icon-btn {
      background: rgba(255,255,255,0.12);
      border: 1px solid rgba(255,255,255,0.2);
      color: rgba(255,255,255,0.85);
      padding: 5px 10px;
      border-radius: 5px;
      cursor: pointer;
      font-size: 11px;
      font-weight: 600;
      transition: background 0.15s;
    }

    .header-icon-btn:hover,
    .header-icon-btn.active {
      background: rgba(255,255,255,0.22);
      border-color: rgba(255,255,255,0.4);
    }

    /* ── HIL banner ─────────────────────────────────────────────── */
    .hil-banner {
      background: #451a03;
      border-bottom: 2px solid #fbbf24;
      padding: 12px 20px;
      flex-shrink: 0;
      animation: slideDown 0.25s ease-out;
    }

    .hil-inner {
      display: flex;
      gap: 12px;
      align-items: flex-start;
      max-width: 900px;
      margin: 0 auto;
    }

    .hil-icon {
      font-size: 22px;
      flex-shrink: 0;
      padding-top: 2px;
    }

    .hil-body {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .hil-title {
      font-weight: 700;
      color: #fbbf24;
      font-size: 13px;
    }

    .hil-reason {
      color: #fde68a;
      font-size: 12px;
      line-height: 1.5;
    }

    .hil-actions {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }

    .hil-input {
      flex: 1;
      min-width: 200px;
      background: #1c1008;
      border: 1px solid #92400e;
      color: #fde68a;
      padding: 6px 10px;
      border-radius: 5px;
      font-size: 12px;
    }

    .hil-input::placeholder { color: #78350f; }
    .hil-input:focus { outline: none; border-color: #fbbf24; }

    .hil-approve-btn,
    .hil-reject-btn {
      border: none;
      border-radius: 5px;
      padding: 6px 14px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.15s;
    }

    .hil-approve-btn { background: #065f46; color: #6ee7b7; }
    .hil-approve-btn:hover { opacity: 0.85; }
    .hil-reject-btn  { background: #7f1d1d; color: #fca5a5; }
    .hil-reject-btn:hover { opacity: 0.85; }

    /* ── Main split layout ──────────────────────────────────────── */
    .main-split {
      flex: 1;
      display: flex;
      overflow: hidden;
    }

    .messages-area {
      flex: 1;
      overflow-y: auto;
      padding: 20px;
      background: #0f1419;
    }

    .side-panel {
      width: 280px;
      flex-shrink: 0;
      overflow: hidden;
      border-left: 1px solid #1f2937;
    }
  `]
})
export class ChatComponent implements OnInit, OnDestroy, AfterViewChecked {
  @ViewChild('messagesContainer') private messagesContainer!: ElementRef;

  messages: Message[] = [];
  userInput: string = '';
  isLoading = false;
  useHybridSearch = true;
  showCitations = true;

  // v2 stream toggles
  useV2Stream = true;
  hilEnabled = false;

  // HIL state
  hilInterrupt: HILInterruptPayload | null = null;
  hilInput = '';

  // Side panels
  showActivityPanel = false;
  showTimeline = false;
  agentActivities: AgentActivityEntry[] = [];

  // Session identity
  currentSessionId: string = this._initSessionId();

  // Document ingestion properties
  showUploadPanel = false;
  uploadUrl: string = '';
  isIngesting = false;
  ingestionProgress = 0;
  ingestionStatus: string = '';

  exampleQueries = [
    'How do I initialize the database?',
    'Show me the authentication flow',
    'What\'s the API endpoint for search?',
    'Explain the vector search implementation'
  ];

  private shouldScroll = true;
  private subs = new Subscription();

  constructor(
    private aiStreamService: AIStreamService,
    private agentStreamService: AgentStreamService,
    private ingestService: IngestService,
    private cdr: ChangeDetectorRef,
  ) {}

  ngOnInit() {
    // ── v1 stream subscriptions (kept for backward compat) ────────
    this.subs.add(
      this.aiStreamService.streaming$.subscribe((token) => {
        if (!this.useV2Stream && this.messages.length > 0) {
          const lastIdx = this.messages.length - 1;
          const last = this.messages[lastIdx];
          if (last.role === 'assistant') {
            const updated = { ...last, content: last.content + token };
            this.messages = [
              ...this.messages.slice(0, lastIdx),
              updated,
            ];
            this.shouldScroll = true;
            this.cdr.detectChanges();
          }
        }
      })
    );

    this.subs.add(
      this.aiStreamService.loading$.subscribe((loading) => {
        if (!this.useV2Stream) {
          this.isLoading = loading;
          if (!loading && this.messages.length > 0) {
            const lastIdx = this.messages.length - 1;
            if (this.messages[lastIdx].role === 'assistant') {
              const updated = { ...this.messages[lastIdx], isStreaming: false };
              this.messages = [...this.messages.slice(0, lastIdx), updated];
            }
          }
          this.cdr.detectChanges();
        }
      })
    );

    this.subs.add(
      this.aiStreamService.error$.subscribe((error) => {
        if (!this.useV2Stream && error && this.messages.length > 0) {
          const lastIdx = this.messages.length - 1;
          const updated = { ...this.messages[lastIdx], error };
          this.messages = [...this.messages.slice(0, lastIdx), updated];
          this.cdr.detectChanges();
        }
      })
    );

    // ── v2 stream subscriptions ───────────────────────────────────
    this.subs.add(
      this.agentStreamService.fullMessage$.subscribe((text) => {
        if (this.useV2Stream && this.messages.length > 0) {
          const lastIdx = this.messages.length - 1;
          if (this.messages[lastIdx].role === 'assistant') {
            // Mutate the last message content in-place to avoid replacing the
            // array reference — trackBy keeps the same DOM node alive.
            this.messages[lastIdx] = { ...this.messages[lastIdx], content: text };
            this.shouldScroll = true;
            this.cdr.detectChanges();
          }
        }
      })
    );

    this.subs.add(
      this.agentStreamService.loading$.subscribe((loading) => {
        if (this.useV2Stream) {
          this.isLoading = loading;
          if (!loading && this.messages.length > 0) {
            const lastIdx = this.messages.length - 1;
            if (this.messages[lastIdx].role === 'assistant') {
              const updated = { ...this.messages[lastIdx], isStreaming: false };
              this.messages = [...this.messages.slice(0, lastIdx), updated];
            }
          }
          this.cdr.detectChanges();
        }
      })
    );

    this.subs.add(
      this.agentStreamService.error$.subscribe((error) => {
        if (this.useV2Stream && error && this.messages.length > 0) {
          const lastIdx = this.messages.length - 1;
          const updated = { ...this.messages[lastIdx], error };
          this.messages = [...this.messages.slice(0, lastIdx), updated];
          this.cdr.detectChanges();
        }
      })
    );

    this.subs.add(
      this.agentStreamService.activity$.subscribe((activities) => {
        this.agentActivities = activities;
        this.cdr.detectChanges();
      })
    );

    this.subs.add(
      this.agentStreamService.hilInterrupt$.subscribe((hil) => {
        this.hilInterrupt = hil;
        this.hilInput = '';
        this.cdr.detectChanges();
      })
    );
  }

  ngOnDestroy(): void {
    this.subs.unsubscribe();
    this.agentStreamService.cancelStream();
  }

  // ─── TrackBy ─────────────────────────────────────────────────────────────

  trackByMessageId(_index: number, message: Message): string {
    return message.id;
  }

  // ─── HIL handlers ────────────────────────────────────────────────────────

  onHILApprove(): void {
    this.agentStreamService.resolveHIL(true, this.hilInput);
  }

  onHILReject(): void {
    this.agentStreamService.resolveHIL(false, this.hilInput);
  }

  // ─── Panel toggles ───────────────────────────────────────────────────────

  toggleActivityPanel(): void {
    this.showActivityPanel = !this.showActivityPanel;
    if (this.showActivityPanel) this.showTimeline = false;
  }

  toggleTimeline(): void {
    this.showTimeline = !this.showTimeline;
    if (this.showTimeline) this.showActivityPanel = false;
  }

  // ─── Checkpoint / replay / branch ────────────────────────────────────────

  onCheckpointBadgeClick(_checkpointId: string): void {
    this.showTimeline = true;
    this.showActivityPanel = false;
  }

  onReplayCheckpoint(checkpointId: string): void {
    this._prepareAssistantPlaceholder();
    this.agentStreamService.replayFromCheckpoint(this.currentSessionId, checkpointId);
  }

  onBranchCreated(branchSessionId: string): void {
    // Switch to the new branch session
    this.currentSessionId = branchSessionId;
    localStorage.setItem('chat_session_id', branchSessionId);
    this.messages = [];
    this.agentActivities = [];
    this.showTimeline = false;
  }

  /**
   * Handle Enter key press in textarea
   */
  onEnterKey(event: Event) {
    const keyboardEvent = event as KeyboardEvent;
    if (keyboardEvent.ctrlKey) {
      this.sendMessage();
    }
  }

  /**
   * Send user message and trigger streaming response
   */
  sendMessage(query?: string) {
    const message = query || this.userInput.trim();
    if (!message) return;

    // Add user message
    this.messages.push({
      id: Date.now().toString(),
      role: 'user',
      content: message,
      timestamp: new Date()
    });

    // Clear input
    this.userInput = '';
    this.shouldScroll = true;

    this._prepareAssistantPlaceholder();

    if (this.useV2Stream) {
      // ── v2: LangGraph agent stream ─────────────────────────────
      this.agentStreamService.sendMessage(message, this.currentSessionId, {
        hilEnabled: this.hilEnabled,
        hilThreshold: 0.5,
      });
    } else {
      // ── v1: legacy RAG stream ──────────────────────────────────
      const assistantMessage = this.messages[this.messages.length - 1];
      this.aiStreamService.streamChatResponse(
        message,
        undefined,
        this.useHybridSearch
      ).subscribe({
        error: (err) => {
          assistantMessage.error = err.message;
        }
      });
    }
  }

  /**
   * Stop ongoing stream
   */
  stopStreaming() {
    if (this.useV2Stream) {
      this.agentStreamService.cancelStream();
    } else {
      this.aiStreamService.cancelStream();
    }
  }

  // ─── Private helpers ─────────────────────────────────────────────────────

  private _prepareAssistantPlaceholder(): void {
    const assistantMessage: Message = {
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isStreaming: true,
    };
    this.messages.push(assistantMessage);
  }

  private _initSessionId(): string {
    let sessionId = localStorage.getItem('chat_session_id');
    if (!sessionId) {
      sessionId = `session-${crypto.randomUUID()}`;
      localStorage.setItem('chat_session_id', sessionId);
    }
    return sessionId;
  }

  /**
   * Handle file upload
   */
  onFilesSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      const files = Array.from(input.files);
      this.uploadFiles(files);
    }
  }

  /**
   * Upload files to backend for ingestion
   */
  uploadFiles(files: File[]) {
    if (files.length === 0) return;

    this.isIngesting = true;
    this.ingestionStatus = `Uploading ${files.length} file(s)...`;

    this.ingestService.uploadDocuments(files).subscribe({
      next: (response) => {
        this.ingestionStatus = `✓ Successfully ingested ${files.length} file(s)`;
        this.isIngesting = false;
        setTimeout(() => {
          this.showUploadPanel = false;
          this.ingestionStatus = '';
        }, 2000);
      },
      error: (error) => {
        this.ingestionStatus = `✗ Upload failed: ${error.message}`;
        this.isIngesting = false;
      }
    });
  }

  /**
   * Ingest content from URL
   */
  ingestUrl() {
    if (!this.uploadUrl.trim()) return;

    this.isIngesting = true;
    this.ingestionStatus = 'Ingesting from URL...';

    this.ingestService.ingestFromUrl(this.uploadUrl).subscribe({
      next: (response) => {
        this.ingestionStatus = '✓ Successfully ingested URL content';
        this.uploadUrl = '';
        this.isIngesting = false;
        setTimeout(() => {
          this.showUploadPanel = false;
          this.ingestionStatus = '';
        }, 2000);
      },
      error: (error) => {
        this.ingestionStatus = `✗ Ingestion failed: ${error.message}`;
        this.isIngesting = false;
      }
    });
  }

  /**
   * Toggle upload panel
   */
  toggleUploadPanel() {
    this.showUploadPanel = !this.showUploadPanel;
  }

  /**
   * Auto-scroll messages area
   */
  ngAfterViewChecked() {
    if (this.shouldScroll) {
      this.scrollToBottom();
      this.shouldScroll = false;
    }
  }

  private scrollToBottom() {
    try {
      this.messagesContainer.nativeElement.scrollTop =
        this.messagesContainer.nativeElement.scrollHeight;
    } catch (err) { }
  }
}
