/**
 * AgentActivityComponent — Phase I
 *
 * A collapsible side-panel that renders the live LangGraph traversal log:
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
      background: #1a1f2e;
      border-left: 1px solid #2d3748;
      font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
      font-size: 12px;
      color: #e2e8f0;
    }

    /* ── Header ─────────────────────────────────────────────────── */
    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 14px;
      border-bottom: 1px solid #2d3748;
      background: #1e2534;
      flex-shrink: 0;
    }

    .panel-title {
      font-weight: 600;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #94a3b8;
    }

    .badge-count {
      background: #374151;
      color: #9ca3af;
      border-radius: 10px;
      padding: 1px 7px;
      font-size: 10px;
    }

    /* ── Streaming indicator ─────────────────────────────────────── */
    .streaming-indicator {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 6px 14px;
      background: #111827;
      border-bottom: 1px solid #1f2937;
      flex-shrink: 0;
      color: #667eea;
      font-size: 11px;
    }

    .pulse-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #667eea;
      animation: pulse 1.4s ease-in-out infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50%       { opacity: 0.4; transform: scale(0.85); }
    }

    /* ── Empty state ─────────────────────────────────────────────── */
    .empty-state {
      padding: 24px 14px;
      text-align: center;
      color: #4b5563;
      font-size: 11px;
    }

    /* ── Activity list ───────────────────────────────────────────── */
    .activity-list {
      flex: 1;
      overflow-y: auto;
      padding: 8px 0;
    }

    .activity-list::-webkit-scrollbar { width: 4px; }
    .activity-list::-webkit-scrollbar-track { background: transparent; }
    .activity-list::-webkit-scrollbar-thumb { background: #374151; border-radius: 2px; }

    /* ── Entry ───────────────────────────────────────────────────── */
    .activity-entry {
      display: flex;
      align-items: flex-start;
      gap: 8px;
      padding: 5px 14px;
      border-left: 2px solid transparent;
      transition: background 0.15s ease;
    }

    .activity-entry:hover {
      background: rgba(255, 255, 255, 0.03);
    }

    .entry-running  { border-left-color: #667eea; }
    .entry-done     { border-left-color: #10b981; }
    .entry-error    { border-left-color: #ef4444; }
    .entry-agent-switch { background: rgba(102, 126, 234, 0.06); }
    .entry-interrupt    { background: rgba(251, 191, 36, 0.07); }

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
      width: 12px;
      height: 12px;
      border: 2px solid #374151;
      border-top-color: #667eea;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    .icon-done  { color: #10b981; font-size: 11px; font-weight: 700; }
    .icon-error { color: #ef4444; font-size: 11px; font-weight: 700; }

    /* ── Entry content ───────────────────────────────────────────── */
    .entry-content {
      flex: 1;
      min-width: 0;
      display: flex;
      flex-direction: column;
      gap: 3px;
    }

    .entry-label {
      color: #e2e8f0;
      font-size: 11.5px;
      line-height: 1.4;
      word-break: break-word;
    }

    .duration-badge {
      display: inline-block;
      background: #1f2937;
      color: #6b7280;
      border-radius: 4px;
      padding: 0 5px;
      font-size: 10px;
      align-self: flex-start;
    }

    .tool-input {
      background: #111827;
      border: 1px solid #1f2937;
      border-radius: 4px;
      padding: 4px 6px;
      font-size: 10px;
      color: #9ca3af;
      overflow: hidden;
      max-height: 60px;
      text-overflow: ellipsis;
      white-space: pre-wrap;
      word-break: break-all;
      margin: 2px 0 0;
    }

    .checkpoint-badge {
      display: inline-block;
      background: transparent;
      border: 1px solid #374151;
      border-radius: 4px;
      padding: 1px 5px;
      font-size: 10px;
      color: #667eea;
      cursor: pointer;
      align-self: flex-start;
      transition: border-color 0.15s, background 0.15s;
      font-family: inherit;
    }

    .checkpoint-badge:hover {
      border-color: #667eea;
      background: rgba(102, 126, 234, 0.08);
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
