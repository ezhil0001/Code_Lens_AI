import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  EventEmitter,
  Input,
  Output,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { Message } from '../../../data/models/message.model';
import { IconComponent } from '../../../shared/components/icon/icon.component';
import { MarkdownViewerComponent } from '../../../shared/components/markdown-viewer/markdown-viewer.component';
import { CitationBadgeComponent } from '../../../shared/components/citation-badge/citation-badge.component';

/**
 * A single conversation turn. User turns render as a compact card; assistant
 * turns render as unadorned prose with hover actions, sources, and run stats.
 */
@Component({
  selector: 'app-chat-message',
  standalone: true,
  imports: [
    CommonModule,
    IconComponent,
    MarkdownViewerComponent,
    CitationBadgeComponent,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <!-- ── User turn ─────────────────────────────────────────── -->
    <article *ngIf="message.role === 'user'" class="turn user">
      <div class="bubble">{{ message.content }}</div>
      <div class="actions user-actions">
        <button type="button" (click)="copy()" [attr.aria-label]="copied ? 'Copied' : 'Copy message'">
          <app-icon [name]="copied ? 'check' : 'copy'" [size]="14"></app-icon>
        </button>
        <button type="button" (click)="edit.emit(message.content)" aria-label="Edit and resend">
          <app-icon name="edit" [size]="14"></app-icon>
        </button>
      </div>
    </article>

    <!-- ── Assistant turn ────────────────────────────────────── -->
    <article *ngIf="message.role === 'assistant'" class="turn assistant">
      <div class="body">
        <!-- Plain text while streaming keeps per-token markdown re-parsing out
             of the hot path; markdown takes over once the turn completes. -->
        <div *ngIf="message.isStreaming" class="stream">{{ message.content
          }}<span class="caret" aria-hidden="true"></span></div>

        <app-markdown-viewer
          *ngIf="!message.isStreaming && message.content"
          [content]="message.content"
        ></app-markdown-viewer>

        <div *ngIf="message.isStreaming && !message.content" class="thinking" role="status">
          <span class="dot"></span><span class="dot"></span><span class="dot"></span>
          <span class="thinking-text">Working on it…</span>
        </div>
      </div>

      <!-- Error -->
      <div *ngIf="message.error" class="error-card" role="alert">
        <app-icon name="alert" [size]="16"></app-icon>
        <div class="error-text">
          <strong>Something went wrong.</strong>
          <span>{{ message.error }}</span>
        </div>
        <button type="button" class="retry" (click)="retry.emit()">
          <app-icon name="refresh" [size]="14"></app-icon> Retry
        </button>
      </div>

      <!-- Sources -->
      <div
        *ngIf="showCitations && message.citations?.length"
        class="sources"
        [class.open]="sourcesOpen"
      >
        <button type="button" class="sources-toggle" (click)="sourcesOpen = !sourcesOpen">
          <app-icon [name]="sourcesOpen ? 'chevron-down' : 'chevron-right'" [size]="14"></app-icon>
          <app-icon name="file-text" [size]="14"></app-icon>
          {{ message.citations!.length }}
          {{ message.citations!.length === 1 ? 'source' : 'sources' }}
        </button>
        <div class="sources-list" *ngIf="sourcesOpen">
          <app-citation-badge
            *ngFor="let citation of message.citations"
            [citation]="citation"
          ></app-citation-badge>
        </div>
      </div>

      <!-- Actions + stats -->
      <div class="actions" *ngIf="!message.isStreaming && message.content">
        <button type="button" (click)="copy()" [attr.aria-label]="copied ? 'Copied' : 'Copy response'">
          <app-icon [name]="copied ? 'check' : 'copy'" [size]="14"></app-icon>
          <span class="label">{{ copied ? 'Copied' : 'Copy' }}</span>
        </button>
        <button type="button" (click)="retry.emit()" aria-label="Regenerate response">
          <app-icon name="refresh" [size]="14"></app-icon>
          <span class="label">Retry</span>
        </button>
        <span class="stats" *ngIf="message.tokenCount || message.processingTimeMs">
          <span *ngIf="message.tokenCount">{{ message.tokenCount }} tokens</span>
          <span *ngIf="message.processingTimeMs">{{ message.processingTimeMs }} ms</span>
        </span>
      </div>
    </article>
  `,
  styles: [
    `
      :host {
        display: block;
      }

      .turn {
        max-width: var(--content-width);
        margin: 0 auto;
        padding: 0 20px;
      }

      /* ── User ────────────────────────────────────────────────── */
      .turn.user {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        margin-top: 28px;
      }

      .bubble {
        max-width: 85%;
        padding: 11px 16px;
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        background: var(--surface-raised);
        color: var(--text-primary);
        font-size: 15.5px;
        line-height: 1.6;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
      }

      .user-actions {
        margin-top: 2px;
        justify-content: flex-end;
      }

      /* ── Assistant ───────────────────────────────────────────── */
      .turn.assistant {
        margin-top: 22px;
      }

      .body {
        font-size: 16px;
        line-height: 1.7;
        color: var(--text-primary);
      }

      .stream {
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        font-family: inherit;
      }

      .caret {
        display: inline-block;
        width: 2px;
        height: 1.05em;
        margin-left: 2px;
        vertical-align: text-bottom;
        background: var(--accent);
        animation: blink 1s steps(2, start) infinite;
      }

      @keyframes blink {
        50% {
          opacity: 0;
        }
      }

      .thinking {
        display: flex;
        align-items: center;
        gap: 5px;
        color: var(--text-tertiary);
        font-size: 14px;
      }

      .thinking .dot {
        width: 5px;
        height: 5px;
        border-radius: 50%;
        background: currentColor;
        animation: bounce 1.2s ease-in-out infinite;
      }

      .thinking .dot:nth-child(2) {
        animation-delay: 0.15s;
      }

      .thinking .dot:nth-child(3) {
        animation-delay: 0.3s;
      }

      .thinking-text {
        margin-left: 6px;
      }

      @keyframes bounce {
        0%,
        60%,
        100% {
          opacity: 0.3;
          transform: translateY(0);
        }
        30% {
          opacity: 1;
          transform: translateY(-3px);
        }
      }

      /* ── Error ───────────────────────────────────────────────── */
      .error-card {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        margin-top: 12px;
        padding: 12px 14px;
        border: 1px solid var(--danger);
        border-radius: var(--radius-md);
        background: var(--danger-soft);
        color: var(--danger);
        font-size: 13.5px;
      }

      .error-text {
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: 2px;
        color: var(--text-secondary);
      }

      .error-text strong {
        color: var(--text-primary);
        font-weight: 600;
      }

      .retry {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 10px;
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        background: var(--surface-raised);
        color: var(--text-primary);
        font-size: 12.5px;
        cursor: pointer;
      }

      .retry:hover {
        background: var(--surface-hover);
      }

      /* ── Sources ─────────────────────────────────────────────── */
      .sources {
        margin-top: 14px;
      }

      .sources-toggle {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 10px 5px 6px;
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-pill);
        background: transparent;
        color: var(--text-tertiary);
        font-size: 12.5px;
        cursor: pointer;
      }

      .sources-toggle:hover {
        background: var(--surface-hover);
        color: var(--text-primary);
      }

      .sources-list {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 10px;
      }

      /* ── Actions ─────────────────────────────────────────────── */
      .actions {
        display: flex;
        align-items: center;
        gap: 2px;
        margin-top: 10px;
        opacity: 0;
        transition: opacity 0.15s var(--ease);
      }

      .turn:hover .actions,
      .actions:focus-within {
        opacity: 1;
      }

      .actions button {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 8px;
        border: none;
        border-radius: var(--radius-sm);
        background: transparent;
        color: var(--text-tertiary);
        font-size: 12.5px;
        cursor: pointer;
      }

      .actions button:hover {
        background: var(--surface-hover);
        color: var(--text-primary);
      }

      .stats {
        display: flex;
        gap: 10px;
        margin-left: 6px;
        font-size: 11.5px;
        color: var(--text-faint);
      }

      @media (hover: none) {
        .actions {
          opacity: 1;
        }
      }

      @media (max-width: 640px) {
        .turn {
          padding: 0 14px;
        }

        .bubble {
          max-width: 92%;
          font-size: 15px;
        }

        .body {
          font-size: 15.5px;
        }

        .label {
          display: none;
        }
      }
    `,
  ],
})
export class ChatMessageComponent {
  @Input({ required: true }) message!: Message;
  @Input() showCitations = true;

  @Output() retry = new EventEmitter<void>();
  @Output() edit = new EventEmitter<string>();

  sourcesOpen = false;
  copied = false;

  constructor(private cdr: ChangeDetectorRef) {}

  copy(): void {
    navigator.clipboard?.writeText(this.message.content).then(() => {
      this.copied = true;
      this.cdr.markForCheck();
      setTimeout(() => {
        this.copied = false;
        this.cdr.markForCheck();
      }, 1600);
    });
  }
}
