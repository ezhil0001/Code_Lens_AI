import {
  AfterViewInit,
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  ElementRef,
  EventEmitter,
  HostListener,
  Input,
  Output,
  ViewChild,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { IconComponent } from '../../../shared/components/icon/icon.component';

export interface AgentOption {
  /** Value sent as `agent_hint`; null lets the supervisor route the query. */
  value: string | null;
  label: string;
  description: string;
  icon: string;
}

/** Mirrors VALID_AGENTS in the backend intent classifier. */
export const AGENT_OPTIONS: AgentOption[] = [
  {
    value: null,
    label: 'Auto',
    description: 'Let the supervisor pick the agent',
    icon: 'sparkles',
  },
  {
    value: 'CodeAgent',
    label: 'Code',
    description: 'Find and explain source code',
    icon: 'code',
  },
  {
    value: 'DocAgent',
    label: 'Docs',
    description: 'Search documentation and KT notes',
    icon: 'book',
  },
  {
    value: 'DebugAgent',
    label: 'Debug',
    description: 'Analyse errors and stack traces',
    icon: 'bug',
  },
  {
    value: 'ArchAgent',
    label: 'Architecture',
    description: 'Explain system design and data flow',
    icon: 'layers',
  },
  {
    value: 'WebAgent',
    label: 'Web',
    description: 'Look up external/upstream sources',
    icon: 'globe',
  },
];

/**
 * Anchored composer: auto-growing input, routing/review controls, and the
 * send/stop affordance.
 */
@Component({
  selector: 'app-chat-composer',
  standalone: true,
  imports: [CommonModule, FormsModule, IconComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="composer-wrap">
      <button
        *ngIf="showScrollToBottom"
        class="scroll-bottom"
        type="button"
        (click)="scrollToBottom.emit()"
        aria-label="Scroll to latest message"
      >
        <app-icon name="arrow-down" [size]="16"></app-icon>
      </button>

      <div class="composer" [class.focused]="focused">
        <textarea
          #input
          [(ngModel)]="value"
          (ngModelChange)="onValueChange()"
          (keydown)="onKeydown($event)"
          (focus)="focused = true"
          (blur)="focused = false"
          [placeholder]="placeholder"
          [attr.aria-label]="placeholder"
          rows="1"
          spellcheck="true"
        ></textarea>

        <div class="toolbar">
          <div class="tools">
            <button
              class="tool-btn"
              type="button"
              (click)="attach.emit()"
              title="Add sources to the knowledge base"
              aria-label="Add sources"
            >
              <app-icon name="paperclip" [size]="16"></app-icon>
            </button>

            <div class="select-wrap">
              <button
                class="tool-pill"
                type="button"
                (click)="toggleAgentMenu($event)"
                [attr.aria-expanded]="agentMenuOpen"
                aria-haspopup="menu"
              >
                <app-icon [name]="selectedAgent.icon" [size]="14"></app-icon>
                <span>{{ selectedAgent.label }}</span>
                <app-icon name="chevron-down" [size]="13"></app-icon>
              </button>

              <div class="menu" *ngIf="agentMenuOpen" role="menu">
                <button
                  *ngFor="let option of agentOptions"
                  type="button"
                  role="menuitem"
                  class="menu-item"
                  [class.selected]="option.value === agentHint"
                  (click)="pickAgent(option)"
                >
                  <app-icon [name]="option.icon" [size]="15"></app-icon>
                  <span class="menu-text">
                    <span class="menu-label">{{ option.label }}</span>
                    <span class="menu-desc">{{ option.description }}</span>
                  </span>
                  <app-icon
                    *ngIf="option.value === agentHint"
                    name="check"
                    [size]="14"
                  ></app-icon>
                </button>
              </div>
            </div>

            <button
              class="tool-pill"
              type="button"
              [class.on]="hilEnabled"
              (click)="hilEnabledChange.emit(!hilEnabled)"
              [attr.aria-pressed]="hilEnabled"
              title="Pause for human review on low-confidence answers"
            >
              <app-icon name="user" [size]="14"></app-icon>
              <span>Review</span>
            </button>
          </div>

          <button
            *ngIf="!streaming"
            class="send"
            type="button"
            (click)="submit()"
            [disabled]="!value.trim()"
            aria-label="Send message"
          >
            <app-icon name="arrow-up" [size]="17" [strokeWidth]="2"></app-icon>
          </button>

          <button
            *ngIf="streaming"
            class="send stop"
            type="button"
            (click)="stop.emit()"
            aria-label="Stop generating"
          >
            <app-icon name="stop" [size]="15" [strokeWidth]="2"></app-icon>
          </button>
        </div>
      </div>

      <p class="hint">
        <kbd>Enter</kbd> to send · <kbd>Shift</kbd>+<kbd>Enter</kbd> for a new line
      </p>
    </div>
  `,
  styles: [
    `
      :host {
        display: block;
      }

      .composer-wrap {
        position: relative;
        width: 100%;
        max-width: var(--content-width);
        margin: 0 auto;
        padding: 0 20px 14px;
      }

      .scroll-bottom {
        position: absolute;
        top: -46px;
        left: 50%;
        transform: translateX(-50%);
        display: grid;
        place-items: center;
        width: 32px;
        height: 32px;
        border: 1px solid var(--border-default);
        border-radius: 50%;
        background: var(--surface-raised);
        color: var(--text-secondary);
        box-shadow: var(--shadow-md);
        cursor: pointer;
        animation: rise 0.18s var(--ease);
      }

      .scroll-bottom:hover {
        color: var(--text-primary);
        border-color: var(--border-strong);
      }

      @keyframes rise {
        from {
          opacity: 0;
          transform: translate(-50%, 6px);
        }
        to {
          opacity: 1;
          transform: translate(-50%, 0);
        }
      }

      /* ── Composer shell ──────────────────────────────────────── */
      .composer {
        display: flex;
        flex-direction: column;
        border: 1px solid var(--border-default);
        border-radius: var(--radius-xl);
        background: var(--surface-raised);
        box-shadow: var(--shadow-md);
        transition: border-color 0.15s var(--ease), box-shadow 0.15s var(--ease);
      }

      .composer.focused {
        border-color: var(--accent-border);
        box-shadow: var(--shadow-md), 0 0 0 3px var(--accent-soft);
      }

      textarea {
        width: 100%;
        max-height: 220px;
        min-height: 26px;
        padding: 14px 16px 4px;
        border: none;
        background: none;
        outline: none;
        resize: none;
        overflow-y: auto;
        font-size: 15.5px;
        line-height: 1.55;
        color: var(--text-primary);
      }

      textarea::placeholder {
        color: var(--text-faint);
      }

      /* ── Toolbar ─────────────────────────────────────────────── */
      .toolbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        padding: 6px 8px 8px 10px;
      }

      .tools {
        display: flex;
        align-items: center;
        gap: 4px;
        min-width: 0;
      }

      .tool-btn {
        display: grid;
        place-items: center;
        width: 30px;
        height: 30px;
        border: none;
        border-radius: var(--radius-sm);
        background: transparent;
        color: var(--text-tertiary);
        cursor: pointer;
      }

      .tool-btn:hover {
        background: var(--surface-hover);
        color: var(--text-primary);
      }

      .tool-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        height: 30px;
        padding: 0 10px;
        border: 1px solid transparent;
        border-radius: var(--radius-pill);
        background: transparent;
        color: var(--text-tertiary);
        font-size: 13px;
        cursor: pointer;
        white-space: nowrap;
      }

      .tool-pill:hover {
        background: var(--surface-hover);
        color: var(--text-primary);
      }

      .tool-pill.on {
        background: var(--accent-soft);
        border-color: var(--accent-border);
        color: var(--accent);
      }

      /* ── Agent menu ──────────────────────────────────────────── */
      .select-wrap {
        position: relative;
      }

      .menu {
        position: absolute;
        bottom: calc(100% + 8px);
        left: 0;
        z-index: 50;
        width: 268px;
        padding: 4px;
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        background: var(--surface-overlay);
        box-shadow: var(--shadow-lg);
      }

      .menu-item {
        display: flex;
        align-items: center;
        gap: 10px;
        width: 100%;
        padding: 8px 9px;
        border: none;
        border-radius: var(--radius-sm);
        background: none;
        color: var(--text-secondary);
        text-align: left;
        cursor: pointer;
      }

      .menu-item:hover {
        background: var(--surface-hover);
        color: var(--text-primary);
      }

      .menu-item.selected {
        color: var(--text-primary);
      }

      .menu-text {
        flex: 1;
        min-width: 0;
        display: flex;
        flex-direction: column;
        line-height: 1.3;
      }

      .menu-label {
        font-size: 13.5px;
        font-weight: 550;
      }

      .menu-desc {
        font-size: 12px;
        color: var(--text-faint);
      }

      /* ── Send ────────────────────────────────────────────────── */
      .send {
        display: grid;
        place-items: center;
        width: 32px;
        height: 32px;
        flex-shrink: 0;
        border: none;
        border-radius: var(--radius-md);
        background: var(--accent);
        color: var(--accent-contrast);
        cursor: pointer;
      }

      .send:hover:not(:disabled) {
        background: var(--accent-hover);
      }

      .send:disabled {
        background: var(--surface-active);
        color: var(--text-faint);
      }

      .send.stop {
        background: var(--text-primary);
        color: var(--surface-app);
      }

      /* ── Hint ────────────────────────────────────────────────── */
      .hint {
        margin: 8px 0 0;
        text-align: center;
        font-size: 11.5px;
        color: var(--text-faint);
      }

      kbd {
        font-family: var(--font-ui);
        font-size: inherit;
        color: var(--text-tertiary);
      }

      @media (max-width: 640px) {
        .composer-wrap {
          padding: 0 12px 10px;
        }

        .hint {
          display: none;
        }

        textarea {
          font-size: 16px; /* prevents iOS zoom-on-focus */
        }
      }
    `,
  ],
})
export class ChatComposerComponent implements AfterViewInit {
  @Input() value = '';
  @Input() streaming = false;
  @Input() agentHint: string | null = null;
  @Input() hilEnabled = false;
  @Input() showScrollToBottom = false;
  @Input() placeholder = 'Ask about your codebase…';

  @Output() valueChange = new EventEmitter<string>();
  @Output() send = new EventEmitter<string>();
  @Output() stop = new EventEmitter<void>();
  @Output() attach = new EventEmitter<void>();
  @Output() agentHintChange = new EventEmitter<string | null>();
  @Output() hilEnabledChange = new EventEmitter<boolean>();
  @Output() scrollToBottom = new EventEmitter<void>();

  @ViewChild('input') private inputRef!: ElementRef<HTMLTextAreaElement>;

  readonly agentOptions = AGENT_OPTIONS;
  focused = false;
  agentMenuOpen = false;

  constructor(private cdr: ChangeDetectorRef) {}

  get selectedAgent(): AgentOption {
    return (
      this.agentOptions.find((o) => o.value === this.agentHint) ??
      this.agentOptions[0]
    );
  }

  ngAfterViewInit(): void {
    this.autoGrow();
    this.focus();
  }

  @HostListener('document:click')
  onDocumentClick(): void {
    if (this.agentMenuOpen) {
      this.agentMenuOpen = false;
      this.cdr.markForCheck();
    }
  }

  focus(): void {
    queueMicrotask(() => this.inputRef?.nativeElement.focus());
  }

  setValue(next: string): void {
    this.value = next;
    this.valueChange.emit(next);
    this.cdr.markForCheck();
    queueMicrotask(() => this.autoGrow());
  }

  onValueChange(): void {
    this.valueChange.emit(this.value);
    this.autoGrow();
  }

  onKeydown(event: KeyboardEvent): void {
    if (event.key !== 'Enter') return;
    // Shift+Enter inserts a newline; IME composition must never submit.
    if (event.shiftKey || event.isComposing) return;
    event.preventDefault();
    this.submit();
  }

  submit(): void {
    const text = this.value.trim();
    if (!text || this.streaming) return;
    this.send.emit(text);
    this.value = '';
    this.valueChange.emit('');
    this.autoGrow();
  }

  toggleAgentMenu(event: MouseEvent): void {
    event.stopPropagation();
    this.agentMenuOpen = !this.agentMenuOpen;
  }

  pickAgent(option: AgentOption): void {
    this.agentMenuOpen = false;
    this.agentHintChange.emit(option.value);
    this.focus();
  }

  private autoGrow(): void {
    const el = this.inputRef?.nativeElement;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  }
}
