/**
 * AgentActivityComponent
 *
 * Inspector panel that renders the live LangGraph traversal log:
 * agent switches, tool calls, checkpoints, HIL interrupts, errors.
 *
 * Designed as a pure display component — receives `activities` from ChatComponent
 * and emits `checkpointSelected` when the user clicks a checkpoint badge.
 */

import {
  ChangeDetectionStrategy,
  Component,
  EventEmitter,
  Input,
  OnChanges,
  Output,
  SimpleChanges,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  AgentActivityEntry,
  SSEEventType,
} from '../../../core/services/agent-stream.service';

@Component({
  selector: 'app-agent-activity',
  standalone: true,
  imports: [CommonModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="activity-panel">
      <!-- Header -->
      <div class="panel-header">
        <span class="panel-title">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2" stroke-linecap="round"
               stroke-linejoin="round" style="display:inline;vertical-align:-2px;margin-right:6px">
            <circle cx="12" cy="12" r="10"/>
            <polyline points="12 6 12 12 16 14"/>
          </svg>
          Agent Activity
        </span>
        <span class="badge-count">{{ activities.length }}</span>
      </div>

      <!-- Streaming indicator -->
      <div *ngIf="isStreaming" class="streaming-indicator">
        <span class="pulse-dot"></span>
        <span>Processing…</span>
      </div>

      <!-- Empty state -->
      <div *ngIf="!isStreaming && activities.length === 0" class="empty-state">
        No activity yet. Send a message to start.
      </div>

      <!-- Activity list -->
      <div class="activity-list" *ngIf="activities.length > 0">
        <div
          *ngFor="let entry of activities; trackBy: trackById"
          class="activity-entry"
          [class.entry-running]="entry.status === 'running'"
          [class.entry-done]="entry.status === 'done'"
          [class.entry-error]="entry.status === 'error'"
          [class.entry-agent-switch]="entry.type === 'agent_switch'"
          [class.entry-checkpoint]="entry.type === 'checkpoint'"
          [class.entry-interrupt]="entry.type === 'interrupt'"
        >
          <!-- Left: icon -->
          <div class="entry-icon">
            <span *ngIf="entry.status === 'running'" class="spinner"></span>
            <span *ngIf="entry.status === 'done' && entry.type !== 'error'" class="icon-done">✓</span>
            <span *ngIf="entry.status === 'error'" class="icon-error">✕</span>
          </div>

          <!-- Right: content -->
          <div class="entry-content">
            <span class="entry-label">{{ entry.label }}</span>

            <!-- Duration badge -->
            <span *ngIf="entry.durationMs != null" class="duration-badge">
              {{ entry.durationMs }}ms
            </span>

            <!-- Tool input preview -->
            <pre *ngIf="entry.toolInput" class="tool-input">{{ entry.toolInput | json }}</pre>

            <!-- Checkpoint badge (clickable) -->
            <button
              *ngIf="entry.checkpoint_id"
              class="checkpoint-badge"
              (click)="checkpointSelected.emit(entry.checkpoint_id!)"
              title="View checkpoint {{ entry.checkpoint_id }}"
            >
              {{ entry.checkpoint_id.slice(0, 8) }}…
            </button>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    :host {
      display: block;
      height: 100%;
      overflow: hidden;
    }

    .activity-panel {
      display: flex;
      flex-direction: column;
      height: 100%;
      background: var(--surface-sidebar);
      font-size: 12px;
      color: var(--text-secondary);
    }

    /* ── Header ─────────────────────────────────────────────────── */
    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 44px 0 16px;
      height: var(--header-height);
      border-bottom: 1px solid var(--border-subtle);
      flex-shrink: 0;
    }

    .panel-title {
      font-weight: 600;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-tertiary);
    }

    .badge-count {
      background: var(--surface-active);
      color: var(--text-tertiary);
      border-radius: var(--radius-pill);
      padding: 1px 7px;
      font-size: 10px;
    }

    /* ── Streaming indicator ─────────────────────────────────────── */
    .streaming-indicator {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 16px;
      border-bottom: 1px solid var(--border-subtle);
      flex-shrink: 0;
      color: var(--accent);
      font-size: 11.5px;
    }

    .pulse-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--accent);
      animation: pulse 1.4s ease-in-out infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50%       { opacity: 0.4; transform: scale(0.85); }
    }

    /* ── Empty state ─────────────────────────────────────────────── */
    .empty-state {
      padding: 28px 20px;
      text-align: center;
      color: var(--text-faint);
      font-size: 12px;
      line-height: 1.5;
    }

    /* ── Activity list ───────────────────────────────────── */
    .activity-list {
      flex: 1;
      overflow-y: auto;
      padding: 8px 0;
    }

    /* ── Entry ───────────────────────────────────────────────────── */
    .activity-entry {
      display: flex;
      align-items: flex-start;
      gap: 8px;
      padding: 6px 16px;
      border-left: 2px solid transparent;
      transition: background 0.15s var(--ease);
    }

    .activity-entry:hover {
      background: var(--surface-hover);
    }

    .entry-running  { border-left-color: var(--accent); }
    .entry-done     { border-left-color: var(--success); }
    .entry-error    { border-left-color: var(--danger); }
    .entry-agent-switch { background: var(--accent-soft); }
    .entry-interrupt    { background: var(--warning-soft); }

    /* ── Entry icon ──────────────────────────────────────────────── */
    .entry-icon {
      flex-shrink: 0;
      width: 16px;
      height: 16px;
      display: flex;
      align-items: center;
      justify-content: center;
      margin-top: 2px;
    }

    .spinner {
      width: 11px;
      height: 11px;
      border: 2px solid var(--border-default);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    .icon-done  { color: var(--success); font-size: 11px; font-weight: 700; }
    .icon-error { color: var(--danger); font-size: 11px; font-weight: 700; }

    /* ── Entry content ───────────────────────────────────────────── */
    .entry-content {
      flex: 1;
      min-width: 0;
      display: flex;
      flex-direction: column;
      gap: 3px;
    }

    .entry-label {
      color: var(--text-secondary);
      font-size: 12px;
      line-height: 1.45;
      word-break: break-word;
    }

    .duration-badge {
      display: inline-block;
      background: var(--surface-active);
      color: var(--text-faint);
      border-radius: 4px;
      padding: 0 5px;
      font-size: 10px;
      align-self: flex-start;
    }

    .tool-input {
      background: var(--surface-sunken);
      border: 1px solid var(--border-subtle);
      border-radius: 5px;
      padding: 5px 7px;
      font-family: var(--font-mono);
      font-size: 10px;
      color: var(--text-tertiary);
      overflow: hidden;
      max-height: 64px;
      white-space: pre-wrap;
      word-break: break-all;
      margin: 2px 0 0;
    }

    .checkpoint-badge {
      display: inline-block;
      background: transparent;
      border: 1px solid var(--border-default);
      border-radius: 5px;
      padding: 1px 6px;
      font-family: var(--font-mono);
      font-size: 10px;
      color: var(--accent);
      cursor: pointer;
      align-self: flex-start;
    }

    .checkpoint-badge:hover {
      border-color: var(--accent-border);
      background: var(--accent-soft);
    }
  `],
})
export class AgentActivityComponent implements OnChanges {
  @Input() activities: AgentActivityEntry[] = [];
  @Input() isStreaming = false;
  @Output() checkpointSelected = new EventEmitter<string>();

  ngOnChanges(_changes: SimpleChanges): void {
    // Scroll handled via CSS overflow — nothing else needed
  }

  trackById(_index: number, entry: AgentActivityEntry): string {
    return entry.id;
  }
}
