import {
  AfterViewChecked,
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  ElementRef,
  HostListener,
  OnDestroy,
  OnInit,
  ViewChild,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';
import { Subscription } from 'rxjs';

import { environment } from '../../../../environments/environment';
import { Citation, Message } from '../../../data/models/message.model';
import {
  AgentActivityEntry,
  AgentStreamService,
  HILInterruptPayload,
} from '../../../core/services/agent-stream.service';
import { ConversationStoreService } from '../../../core/services/conversation-store.service';
import { IconComponent } from '../../../shared/components/icon/icon.component';

import { AgentActivityComponent } from './agent-activity.component';
import { CheckpointTimelineComponent } from './checkpoint-timeline.component';
import { ChatComposerComponent } from './chat-composer.component';
import { ChatMessageComponent } from './chat-message.component';
import { ChatSidebarComponent } from './chat-sidebar.component';
import { KnowledgePanelComponent } from './knowledge-panel.component';

interface Suggestion {
  icon: string;
  label: string;
  prompt: string;
}

/**
 * ChatComponent — the CodeLens workspace shell.
 *
 * Owns the three-region layout (navigation, conversation, inspector panels),
 * conversation lifecycle, and every subscription to the streaming service.
 * All backend contracts are unchanged: v2 SSE stream, history, checkpoints,
 * resume/replay/branch, and ingestion.
 */
@Component({
  selector: 'app-chat',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [
    CommonModule,
    IconComponent,
    ChatSidebarComponent,
    ChatMessageComponent,
    ChatComposerComponent,
    KnowledgePanelComponent,
    AgentActivityComponent,
    CheckpointTimelineComponent,
  ],
  template: `
    <div class="shell">
      <app-chat-sidebar
        [collapsed]="sidebarCollapsed && !isNarrow"
        [drawerOpen]="drawerOpen"
        [userLabel]="userLabel"
        (toggleCollapsed)="toggleSidebar()"
        (closeDrawer)="drawerOpen = false"
        (newChat)="startNewChat()"
        (select)="openConversation($event)"
        (openKnowledge)="openKnowledge()"
        (logout)="logout()"
      ></app-chat-sidebar>

      <div class="scrim" *ngIf="drawerOpen" (click)="drawerOpen = false"></div>

      <main class="workspace">
        <!-- ── Top bar ─────────────────────────────────────────── -->
        <header class="topbar">
          <button
            class="icon-btn menu-btn"
            type="button"
            (click)="drawerOpen = true"
            aria-label="Open navigation"
          >
            <app-icon name="panel-left" [size]="18"></app-icon>
          </button>

          <h1 class="title">{{ isEmpty ? '' : conversationTitle }}</h1>

          <div class="topbar-actions">
            <button
              class="pill-btn"
              type="button"
              [class.on]="showActivityPanel"
              (click)="toggleActivityPanel()"
              title="Agent activity"
            >
              <app-icon name="activity" [size]="15"></app-icon>
              <span class="pill-label">Activity</span>
              <span class="pulse" *ngIf="isLoading"></span>
            </button>
            <button
              class="pill-btn"
              type="button"
              [class.on]="showTimeline"
              (click)="toggleTimeline()"
              title="Checkpoint timeline"
            >
              <app-icon name="clock" [size]="15"></app-icon>
              <span class="pill-label">Timeline</span>
            </button>
          </div>
        </header>

        <div class="body">
          <section class="conversation" [class.is-empty]="isEmpty">
            <!-- ── Empty state: greeting + centred composer ────── -->
            <div class="hero" *ngIf="isEmpty">
              <div class="hero-inner">
                <h2 class="greeting">{{ greeting }}</h2>
                <p class="greeting-sub">
                  Ask anything about the code and documents in your workspace.
                </p>

                <app-chat-composer
                  #composer
                  [value]="draft"
                  [streaming]="isLoading"
                  [agentHint]="agentHint"
                  [hilEnabled]="hilEnabled"
                  (valueChange)="draft = $event"
                  (send)="sendMessage($event)"
                  (stop)="stopStreaming()"
                  (attach)="openKnowledge()"
                  (agentHintChange)="agentHint = $event"
                  (hilEnabledChange)="hilEnabled = $event"
                ></app-chat-composer>

                <div class="suggestions">
                  <button
                    *ngFor="let s of suggestions"
                    type="button"
                    class="suggestion"
                    (click)="sendMessage(s.prompt)"
                  >
                    <app-icon [name]="s.icon" [size]="15"></app-icon>
                    {{ s.label }}
                  </button>
                </div>
              </div>
            </div>

            <!-- ── Transcript ─────────────────────────────────── -->
            <div
              class="scroller"
              #scroller
              (scroll)="onScroll()"
              *ngIf="!isEmpty"
            >
              <div class="transcript">
                <app-chat-message
                  *ngFor="let message of messages; trackBy: trackByMessageId"
                  [message]="message"
                  (retry)="retryLast()"
                  (edit)="editMessage($event)"
                ></app-chat-message>
                <div class="tail"></div>
              </div>
            </div>

            <!-- ── Docked composer ────────────────────────────── -->
            <div class="dock" *ngIf="!isEmpty">
              <div class="hil" *ngIf="hilInterrupt" role="alertdialog">
                <div class="hil-head">
                  <app-icon name="alert" [size]="16"></app-icon>
                  <span>Review needed</span>
                </div>
                <p class="hil-reason">{{ hilInterrupt.reason }}</p>
                <div class="hil-actions">
                  <input
                    type="text"
                    [value]="hilInput"
                    (input)="hilInput = $any($event.target).value"
                    placeholder="Add guidance for the agent (optional)"
                    aria-label="Reviewer guidance"
                  />
                  <button type="button" class="approve" (click)="onHILApprove()">
                    Approve
                  </button>
                  <button type="button" class="reject" (click)="onHILReject()">
                    Reject
                  </button>
                </div>
              </div>

              <app-chat-composer
                #composer
                [value]="draft"
                [streaming]="isLoading"
                [agentHint]="agentHint"
                [hilEnabled]="hilEnabled"
                [showScrollToBottom]="!autoScroll"
                (valueChange)="draft = $event"
                (send)="sendMessage($event)"
                (stop)="stopStreaming()"
                (attach)="openKnowledge()"
                (agentHintChange)="agentHint = $event"
                (hilEnabledChange)="hilEnabled = $event"
                (scrollToBottom)="jumpToBottom()"
              ></app-chat-composer>
            </div>
          </section>

          <!-- ── Inspector panels ───────────────────────────────── -->
          <aside class="inspector" *ngIf="showActivityPanel || showTimeline">
            <button
              class="inspector-close"
              type="button"
              (click)="closeInspector()"
              aria-label="Close panel"
            >
              <app-icon name="x" [size]="16"></app-icon>
            </button>

            <app-agent-activity
              *ngIf="showActivityPanel"
              [activities]="agentActivities"
              [isStreaming]="isLoading"
              (checkpointSelected)="onCheckpointBadgeClick($event)"
            ></app-agent-activity>

            <app-checkpoint-timeline
              *ngIf="showTimeline"
              [sessionId]="currentSessionId"
              [visible]="showTimeline"
              (replayRequested)="onReplayCheckpoint($event)"
              (branchCreated)="onBranchCreated($event)"
              (closed)="showTimeline = false"
            ></app-checkpoint-timeline>
          </aside>
        </div>
      </main>

      <app-knowledge-panel
        *ngIf="showKnowledge"
        (close)="showKnowledge = false"
      ></app-knowledge-panel>
    </div>
  `,
  styles: [
    `
      :host {
        display: block;
        height: 100dvh;
        overflow: hidden;
      }

      .shell {
        display: flex;
        height: 100%;
        background: var(--surface-app);
        color: var(--text-primary);
      }

      .scrim {
        display: none;
      }

      .workspace {
        flex: 1;
        min-width: 0;
        display: flex;
        flex-direction: column;
        height: 100%;
      }

      /* ── Top bar ─────────────────────────────────────────────── */
      .topbar {
        display: flex;
        align-items: center;
        gap: 10px;
        height: var(--header-height);
        padding: 0 12px 0 16px;
        flex-shrink: 0;
        border-bottom: 1px solid transparent;
      }

      .title {
        flex: 1;
        min-width: 0;
        margin: 0;
        font-size: 14px;
        font-weight: 550;
        color: var(--text-secondary);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .icon-btn {
        display: grid;
        place-items: center;
        width: 32px;
        height: 32px;
        border: none;
        border-radius: var(--radius-sm);
        background: transparent;
        color: var(--text-tertiary);
        cursor: pointer;
      }

      .icon-btn:hover {
        background: var(--surface-hover);
        color: var(--text-primary);
      }

      .menu-btn {
        display: none;
      }

      .topbar-actions {
        display: flex;
        align-items: center;
        gap: 4px;
      }

      .pill-btn {
        position: relative;
        display: inline-flex;
        align-items: center;
        gap: 7px;
        height: 32px;
        padding: 0 11px;
        border: 1px solid transparent;
        border-radius: var(--radius-pill);
        background: transparent;
        color: var(--text-tertiary);
        font-size: 13px;
        cursor: pointer;
      }

      .pill-btn:hover {
        background: var(--surface-hover);
        color: var(--text-primary);
      }

      .pill-btn.on {
        background: var(--surface-active);
        border-color: var(--border-subtle);
        color: var(--text-primary);
      }

      .pulse {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--accent);
        animation: pulse 1.4s ease-in-out infinite;
      }

      @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
      }

      /* ── Body ────────────────────────────────────────────────── */
      .body {
        flex: 1;
        min-height: 0;
        display: flex;
      }

      .conversation {
        flex: 1;
        min-width: 0;
        display: flex;
        flex-direction: column;
      }

      /* ── Empty state ─────────────────────────────────────────── */
      .hero {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 24px 0 8vh;
        overflow-y: auto;
      }

      .hero-inner {
        width: 100%;
        max-width: var(--content-width);
        margin: 0 auto;
      }

      .greeting {
        margin: 0;
        padding: 0 20px;
        font-family: var(--font-display);
        font-size: clamp(28px, 4.6vw, 40px);
        font-weight: 400;
        letter-spacing: -0.02em;
        line-height: 1.15;
        color: var(--text-primary);
        text-align: center;
      }

      .greeting-sub {
        margin: 10px 0 26px;
        padding: 0 20px;
        font-size: 15px;
        color: var(--text-tertiary);
        text-align: center;
      }

      .suggestions {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 8px;
        padding: 4px 20px 0;
      }

      .suggestion {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 7px 13px;
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-pill);
        background: transparent;
        color: var(--text-secondary);
        font-size: 13.5px;
        cursor: pointer;
      }

      .suggestion:hover {
        background: var(--surface-hover);
        border-color: var(--border-default);
        color: var(--text-primary);
      }

      /* ── Transcript ──────────────────────────────────────────── */
      .scroller {
        flex: 1;
        min-height: 0;
        overflow-y: auto;
        overflow-anchor: none;
      }

      .transcript {
        padding: 8px 0 0;
      }

      .tail {
        height: 40px;
      }

      /* ── Dock ────────────────────────────────────────────────── */
      .dock {
        flex-shrink: 0;
        padding-top: 6px;
        background: linear-gradient(
          to bottom,
          transparent,
          var(--surface-app) 26px
        );
      }

      /* ── HIL ─────────────────────────────────────────────────── */
      .hil {
        max-width: var(--content-width);
        margin: 0 auto 10px;
        padding: 12px 14px;
        border: 1px solid var(--warning);
        border-radius: var(--radius-md);
        background: var(--warning-soft);
      }

      .hil-head {
        display: flex;
        align-items: center;
        gap: 7px;
        font-size: 13px;
        font-weight: 600;
        color: var(--warning);
      }

      .hil-reason {
        margin: 6px 0 10px;
        font-size: 13.5px;
        color: var(--text-secondary);
      }

      .hil-actions {
        display: flex;
        gap: 8px;
      }

      .hil-actions input {
        flex: 1;
        min-width: 0;
        height: 34px;
        padding: 0 11px;
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        background: var(--surface-raised);
        font-size: 13.5px;
        outline: none;
      }

      .hil-actions button {
        height: 34px;
        padding: 0 14px;
        border: 1px solid transparent;
        border-radius: var(--radius-sm);
        font-size: 13px;
        font-weight: 550;
        cursor: pointer;
      }

      .approve {
        background: var(--success-soft);
        border-color: var(--success);
        color: var(--success);
      }

      .reject {
        background: var(--danger-soft);
        border-color: var(--danger);
        color: var(--danger);
      }

      /* ── Inspector ───────────────────────────────────────────── */
      .inspector {
        position: relative;
        width: 320px;
        flex-shrink: 0;
        border-left: 1px solid var(--border-subtle);
        background: var(--surface-sidebar);
        overflow: hidden;
      }

      .inspector-close {
        position: absolute;
        top: 8px;
        right: 8px;
        z-index: 2;
        display: grid;
        place-items: center;
        width: 26px;
        height: 26px;
        border: none;
        border-radius: var(--radius-sm);
        background: transparent;
        color: var(--text-tertiary);
        cursor: pointer;
      }

      .inspector-close:hover {
        background: var(--surface-hover);
        color: var(--text-primary);
      }

      /* ── Responsive ──────────────────────────────────────────── */
      @media (max-width: 1180px) {
        .inspector {
          width: 288px;
        }
      }

      @media (max-width: 900px) {
        .menu-btn {
          display: grid;
        }

        .scrim {
          display: block;
          position: fixed;
          inset: 0;
          z-index: 55;
          background: rgba(0, 0, 0, 0.4);
        }

        .inspector {
          position: fixed;
          top: 0;
          right: 0;
          bottom: 0;
          z-index: 50;
          width: min(88vw, 340px);
          box-shadow: var(--shadow-lg);
        }

        .pill-label {
          display: none;
        }

        .pill-btn {
          padding: 0 9px;
        }
      }

      @media (max-width: 640px) {
        .greeting-sub {
          margin-bottom: 20px;
          font-size: 14px;
        }

        .hero {
          padding-bottom: 4vh;
        }

        .hil {
          margin: 0 12px 10px;
        }

        .hil-actions {
          flex-wrap: wrap;
        }

        .hil-actions input {
          flex-basis: 100%;
        }
      }
    `,
  ],
})
export class ChatComponent implements OnInit, OnDestroy, AfterViewChecked {
  @ViewChild('scroller') private scroller?: ElementRef<HTMLElement>;
  @ViewChild('composer') private composer?: ChatComposerComponent;

  messages: Message[] = [];
  draft = '';
  isLoading = false;
  agentHint: string | null = null;
  hilEnabled = false;

  hilInterrupt: HILInterruptPayload | null = null;
  hilInput = '';

  sidebarCollapsed = false;
  drawerOpen = false;
  showActivityPanel = false;
  showTimeline = false;
  showKnowledge = false;

  agentActivities: AgentActivityEntry[] = [];
  currentSessionId = '';
  userLabel = 'You';
  autoScroll = true;
  /** Below this width the sidebar becomes an overlay drawer, never a rail. */
  isNarrow = window.innerWidth <= 900;

  readonly suggestions: Suggestion[] = [
    {
      icon: 'code',
      label: 'Explain a module',
      prompt: 'Walk me through the main modules in this codebase.',
    },
    {
      icon: 'layers',
      label: 'Map the architecture',
      prompt: 'Describe the system architecture and how data flows through it.',
    },
    {
      icon: 'bug',
      label: 'Debug an error',
      prompt: 'Help me trace the cause of a runtime error in this project.',
    },
    {
      icon: 'book',
      label: 'Find documentation',
      prompt: 'What documentation exists for the ingestion pipeline?',
    },
  ];

  private shouldScroll = false;
  private lastUserMessage = '';
  private subs = new Subscription();

  constructor(
    private agentStreamService: AgentStreamService,
    private conversations: ConversationStoreService,
    private cdr: ChangeDetectorRef,
    private http: HttpClient,
    private router: Router,
  ) {}

  // ── Derived view state ───────────────────────────────────────────────────

  get isEmpty(): boolean {
    return this.messages.length === 0;
  }

  get conversationTitle(): string {
    return this.conversations.get(this.currentSessionId)?.title ?? 'New chat';
  }

  get greeting(): string {
    const hour = new Date().getHours();
    const part = hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening';
    return `${part}, ${this.userLabel}`;
  }

  // ── Lifecycle ────────────────────────────────────────────────────────────

  ngOnInit(): void {
    this.userLabel = this._resolveUserLabel();
    this.sidebarCollapsed = localStorage.getItem('codelens.sidebarCollapsed') === '1';

    this.conversations.reloadForCurrentUser();
    const active = this.conversations.activeId ?? this.conversations.conversations[0]?.id;
    this.currentSessionId = active ?? this.conversations.createDraftId();
    this.conversations.setActive(this.currentSessionId);

    if (active) this._loadHistory(this.currentSessionId);

    this.subs.add(
      this.agentStreamService.fullMessage$.subscribe((text) => {
        const lastIdx = this.messages.length - 1;
        if (lastIdx < 0 || this.messages[lastIdx].role !== 'assistant') return;
        // Replace the object (not the array) so trackBy keeps the DOM node.
        this.messages[lastIdx] = { ...this.messages[lastIdx], content: text };
        this.shouldScroll = this.autoScroll;
        this.cdr.detectChanges();
      }),
    );

    this.subs.add(
      this.agentStreamService.loading$.subscribe((loading) => {
        this.isLoading = loading;
        const lastIdx = this.messages.length - 1;
        if (!loading && lastIdx >= 0 && this.messages[lastIdx].role === 'assistant') {
          this.messages = [
            ...this.messages.slice(0, lastIdx),
            { ...this.messages[lastIdx], isStreaming: false },
          ];
          if (this.currentSessionId) this.conversations.touch(this.currentSessionId);
        }
        this.cdr.detectChanges();
      }),
    );

    this.subs.add(
      this.agentStreamService.error$.subscribe((error) => {
        const lastIdx = this.messages.length - 1;
        if (!error || lastIdx < 0) return;
        this.messages = [
          ...this.messages.slice(0, lastIdx),
          { ...this.messages[lastIdx], error, isStreaming: false },
        ];
        this.cdr.detectChanges();
      }),
    );

    this.subs.add(
      this.agentStreamService.sources$.subscribe((sources) => {
        const lastIdx = this.messages.length - 1;
        if (!sources.length || lastIdx < 0) return;
        if (this.messages[lastIdx].role !== 'assistant') return;
        const citations: Citation[] = sources.map((s) => ({
          sourceFile: s.file_path,
          repository: '',
          lineStart: 0,
          lineEnd: 0,
          codeSnippet: s.snippet,
          relevanceScore: s.score,
        }));
        this.messages = [
          ...this.messages.slice(0, lastIdx),
          { ...this.messages[lastIdx], citations },
        ];
        this.cdr.detectChanges();
      }),
    );

    this.subs.add(
      this.agentStreamService.activity$.subscribe((activities) => {
        this.agentActivities = activities;
        this.cdr.detectChanges();
      }),
    );

    this.subs.add(
      this.agentStreamService.hilInterrupt$.subscribe((hil) => {
        this.hilInterrupt = hil;
        this.hilInput = '';
        this.cdr.detectChanges();
      }),
    );
  }

  ngOnDestroy(): void {
    this.subs.unsubscribe();
    this.agentStreamService.cancelStream();
  }

  ngAfterViewChecked(): void {
    if (this.shouldScroll) {
      this.shouldScroll = false;
      this._scrollToBottom();
    }
  }

  @HostListener('window:resize')
  onResize(): void {
    const narrow = window.innerWidth <= 900;
    if (narrow === this.isNarrow) return;
    this.isNarrow = narrow;
    if (!narrow) this.drawerOpen = false;
    this.cdr.markForCheck();
  }

  @HostListener('window:keydown', ['$event'])
  onShortcut(event: KeyboardEvent): void {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault();
      this.startNewChat();
    }
    if (event.key === 'Escape') {
      this.drawerOpen = false;
      this.showKnowledge = false;
      this.cdr.markForCheck();
    }
  }

  // ── Conversation lifecycle ───────────────────────────────────────────────

  startNewChat(): void {
    this.agentStreamService.cancelStream();
    this.currentSessionId = this.conversations.createDraftId();
    this.conversations.setActive(this.currentSessionId);
    this.messages = [];
    this.agentActivities = [];
    this.hilInterrupt = null;
    this.draft = '';
    this.autoScroll = true;
    this.drawerOpen = false;
    this.cdr.markForCheck();
  }

  openConversation(sessionId: string): void {
    if (sessionId === this.currentSessionId) {
      this.drawerOpen = false;
      return;
    }
    this.agentStreamService.cancelStream();
    this.currentSessionId = sessionId;
    this.conversations.setActive(sessionId);
    this.messages = [];
    this.agentActivities = [];
    this.hilInterrupt = null;
    this.autoScroll = true;
    this.drawerOpen = false;
    this.cdr.markForCheck();
    this._loadHistory(sessionId);
  }

  // ── Sending ──────────────────────────────────────────────────────────────

  sendMessage(query: string): void {
    const message = query.trim();
    if (!message || this.isLoading) return;

    this.lastUserMessage = message;
    this.conversations.commit(this.currentSessionId, message);

    this.messages = [
      ...this.messages,
      {
        id: `u-${Date.now()}`,
        role: 'user',
        content: message,
        timestamp: new Date(),
      },
      {
        id: `a-${Date.now() + 1}`,
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        isStreaming: true,
      },
    ];

    this.draft = '';
    this.autoScroll = true;
    this.shouldScroll = true;
    this.cdr.detectChanges();

    this.agentStreamService.sendMessage(message, this.currentSessionId, {
      agentHint: this.agentHint ?? undefined,
      hilEnabled: this.hilEnabled,
      hilThreshold: 0.5,
    });
  }

  retryLast(): void {
    if (!this.lastUserMessage || this.isLoading) return;
    // Drop the failed/previous assistant turn before re-running the same query.
    const lastIdx = this.messages.length - 1;
    if (lastIdx >= 0 && this.messages[lastIdx].role === 'assistant') {
      this.messages = this.messages.slice(0, lastIdx);
    }
    const query = this.lastUserMessage;
    if (this.messages[this.messages.length - 1]?.role === 'user') {
      this.messages = this.messages.slice(0, this.messages.length - 1);
    }
    this.sendMessage(query);
  }

  editMessage(content: string): void {
    this.draft = content;
    this.composer?.setValue(content);
    this.composer?.focus();
  }

  stopStreaming(): void {
    this.agentStreamService.cancelStream();
  }

  // ── Scroll management ────────────────────────────────────────────────────

  onScroll(): void {
    const el = this.scroller?.nativeElement;
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    const next = distance < 90;
    if (next !== this.autoScroll) {
      this.autoScroll = next;
      this.cdr.markForCheck();
    }
  }

  jumpToBottom(): void {
    this.autoScroll = true;
    const el = this.scroller?.nativeElement;
    el?.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
  }

  // ── Panels ───────────────────────────────────────────────────────────────

  toggleSidebar(): void {
    this.sidebarCollapsed = !this.sidebarCollapsed;
    localStorage.setItem('codelens.sidebarCollapsed', this.sidebarCollapsed ? '1' : '0');
  }

  toggleActivityPanel(): void {
    this.showActivityPanel = !this.showActivityPanel;
    if (this.showActivityPanel) this.showTimeline = false;
  }

  toggleTimeline(): void {
    this.showTimeline = !this.showTimeline;
    if (this.showTimeline) this.showActivityPanel = false;
  }

  closeInspector(): void {
    this.showActivityPanel = false;
    this.showTimeline = false;
  }

  openKnowledge(): void {
    this.showKnowledge = true;
    this.drawerOpen = false;
  }

  logout(): void {
    localStorage.removeItem('auth_token');
    this.router.navigate(['/login']);
  }

  // ── HIL ──────────────────────────────────────────────────────────────────

  onHILApprove(): void {
    this.agentStreamService.resolveHIL(true, this.hilInput);
  }

  onHILReject(): void {
    this.agentStreamService.resolveHIL(false, this.hilInput);
  }

  // ── Checkpoints ──────────────────────────────────────────────────────────

  onCheckpointBadgeClick(_checkpointId: string): void {
    this.showTimeline = true;
    this.showActivityPanel = false;
  }

  onReplayCheckpoint(checkpointId: string): void {
    this.messages = [
      ...this.messages,
      {
        id: `a-${Date.now()}`,
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        isStreaming: true,
      },
    ];
    this.shouldScroll = true;
    this.agentStreamService.replayFromCheckpoint(this.currentSessionId, checkpointId);
  }

  onBranchCreated(branchSessionId: string): void {
    this.conversations.commit(
      branchSessionId,
      `Branch of ${this.conversationTitle}`,
    );
    this.openConversation(branchSessionId);
    this.showTimeline = false;
  }

  trackByMessageId(_index: number, message: Message): string {
    return message.id;
  }

  // ── Private ──────────────────────────────────────────────────────────────

  /** Restores persisted turns so a refresh or switch doesn't blank the thread. */
  private _loadHistory(sessionId: string): void {
    const token = localStorage.getItem('auth_token');
    if (!token) return;

    this.subs.add(
      this.http
        .get<{ messages: Array<{ role: string; content: string }> }>(
          `${environment.apiUrl}/api/v2/chat/history/${sessionId}`,
          { headers: { Authorization: `Bearer ${token}` } },
        )
        .subscribe({
          next: (res) => {
            if (sessionId !== this.currentSessionId) return;
            if (this.messages.length || !res?.messages?.length) return;
            this.messages = res.messages.map((m, i) => ({
              id: `hist-${i}`,
              role: m.role === 'assistant' ? 'assistant' : 'user',
              content: m.content,
              timestamp: new Date(),
            })) as Message[];
            const lastUser = [...this.messages]
              .reverse()
              .find((m) => m.role === 'user');
            this.lastUserMessage = lastUser?.content ?? '';
            this.shouldScroll = true;
            this.cdr.detectChanges();
          },
          error: (e) => console.warn('history recovery failed', e?.status ?? e),
        }),
    );
  }

  private _scrollToBottom(): void {
    const el = this.scroller?.nativeElement;
    if (el) el.scrollTop = el.scrollHeight;
  }

  private _resolveUserLabel(): string {
    const token = localStorage.getItem('auth_token');
    if (!token) return 'there';
    try {
      const payload = token.split('.')[1];
      const claims = JSON.parse(
        atob(payload.replace(/-/g, '+').replace(/_/g, '/')),
      );
      const email: string = claims?.email ?? claims?.sub ?? '';
      const name = email.includes('@') ? email.split('@')[0] : email;
      if (!name) return 'there';
      return name.charAt(0).toUpperCase() + name.slice(1);
    } catch {
      return 'there';
    }
  }
}
