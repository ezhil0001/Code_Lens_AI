/**
 * CheckpointTimelineComponent — Phase I
 *
 * A slide-in panel that fetches and renders the checkpoint history
 * for the current session. Allows time-travel: replay from any checkpoint
 * or branch the session from a historical state.
 *
 * Inputs:
 *   sessionId     — current session id (passed by ChatComponent)
 *   visible       — drives ngIf in the host; component auto-loads on first show
 *
 * Outputs:
 *   replayRequested — emits checkpointId to replay from that point
 *   branchCreated   — emits new branchSessionId when a branch succeeds
 *   closed          — user clicked the X button
 */

import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  EventEmitter,
  Input,
  OnChanges,
  Output,
  SimpleChanges,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { environment } from '../../../../environments/environment';
import {
  AgentStreamService,
  CheckpointSummary,
} from '../../../core/services/agent-stream.service';

@Component({
  selector: 'app-checkpoint-timeline',
  standalone: true,
  imports: [CommonModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="timeline-panel" role="dialog" aria-label="Checkpoint Timeline">
      <!-- Header -->
      <div class="timeline-header">
        <span class="timeline-title">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2" stroke-linecap="round"
               stroke-linejoin="round" style="display:inline;vertical-align:-2px;margin-right:6px">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
          </svg>
          Checkpoint Timeline
        </span>
        <button class="close-btn" (click)="closed.emit()" title="Close">×</button>
      </div>

      <!-- Loading state -->
      <div *ngIf="loading" class="center-state">
        <div class="spinner-lg"></div>
        <span>Loading checkpoints…</span>
      </div>

      <!-- Error state -->
      <div *ngIf="!loading && loadError" class="center-state error-state">
        <span>⚠ {{ loadError }}</span>
        <button class="retry-btn" (click)="load()">Retry</button>
      </div>

      <!-- Empty state -->
      <div *ngIf="!loading && !loadError && checkpoints.length === 0" class="center-state">
        <span>No checkpoints saved for this session yet.</span>
      </div>

      <!-- Timeline -->
      <div class="timeline-list" *ngIf="!loading && !loadError && checkpoints.length > 0">
        <div
          *ngFor="let cp of checkpoints; let i = index; trackBy: trackById"
          class="timeline-item"
          [class.item-latest]="i === 0"
        >
          <!-- Connector line -->
          <div class="connector-area">
            <div class="dot" [class.dot-latest]="i === 0"></div>
            <div class="line" *ngIf="i < checkpoints.length - 1"></div>
          </div>

          <!-- Content -->
          <div class="item-content">
            <div class="item-row-1">
              <span class="cp-id">{{ cp.checkpoint_id.slice(0, 12) }}…</span>
              <span class="cp-time" *ngIf="cp.created_at">
                {{ formatDate(cp.created_at) }}
              </span>
              <span class="latest-badge" *ngIf="i === 0">latest</span>
            </div>

            <!-- Query preview -->
            <div class="query-preview" *ngIf="cp.query_preview">
              "{{ cp.query_preview | slice:0:100 }}{{ cp.query_preview.length > 100 ? '…' : '' }}"
            </div>

            <!-- Nodes visited -->
            <div class="nodes-row" *ngIf="cp.nodes_visited?.length">
              <span
                *ngFor="let node of cp.nodes_visited"
                class="node-pill"
              >{{ node }}</span>
            </div>

            <!-- Actions -->
            <div class="action-row">
              <button
                class="action-btn replay-btn"
                [disabled]="pendingCp === cp.checkpoint_id"
                (click)="onReplay(cp)"
              >
                <span *ngIf="pendingCp !== cp.checkpoint_id">↺ Replay</span>
                <span *ngIf="pendingCp === cp.checkpoint_id">Loading…</span>
              </button>

              <button
                class="action-btn branch-btn"
                [disabled]="pendingCp === cp.checkpoint_id"
                (click)="onBranch(cp)"
              >
                <span *ngIf="pendingCp !== cp.checkpoint_id">⎇ Branch</span>
                <span *ngIf="pendingCp === cp.checkpoint_id">Loading…</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- Branch success toast -->
      <div *ngIf="branchSuccessMsg" class="toast toast-success">
        {{ branchSuccessMsg }}
      </div>
    </div>
  `,
  styles: [`
    :host {
      display: block;
      height: 100%;
    }

    .timeline-panel {
      display: flex;
      flex-direction: column;
      height: 100%;
      background: var(--surface-sidebar);
      font-size: 12px;
      color: var(--text-secondary);
    }

    /* ── Header ──────────────────────────────────────────────────── */
    .timeline-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 44px 0 16px;
      height: var(--header-height);
      border-bottom: 1px solid var(--border-subtle);
      flex-shrink: 0;
    }

    .timeline-title {
      font-weight: 600;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-tertiary);
    }

    .close-btn {
      display: none;
    }

    /* ── Center states ───────────────────────────────────────────── */
    .center-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 12px;
      padding: 40px 20px;
      color: var(--text-faint);
      font-size: 12px;
      text-align: center;
      line-height: 1.5;
    }

    .error-state { color: var(--danger); }

    .spinner-lg {
      width: 22px;
      height: 22px;
      border: 2px solid var(--border-default);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
    }

    @keyframes spin { to { transform: rotate(360deg); } }

    .retry-btn {
      background: var(--surface-raised);
      border: 1px solid var(--border-default);
      color: var(--text-primary);
      border-radius: var(--radius-sm);
      padding: 5px 12px;
      font-size: 12px;
      cursor: pointer;
    }

    .retry-btn:hover { background: var(--surface-hover); }

    /* ── Timeline list ───────────────────────────────────────────── */
    .timeline-list {
      flex: 1;
      overflow-y: auto;
      padding: 14px 0;
    }

    /* ── Timeline item ───────────────────────────────────────────── */
    .timeline-item {
      display: flex;
      gap: 10px;
      padding: 0 16px;
    }

    .item-latest {
      background: var(--accent-soft);
    }

    /* ── Connector column ────────────────────────────────────────── */
    .connector-area {
      display: flex;
      flex-direction: column;
      align-items: center;
      flex-shrink: 0;
      width: 16px;
      padding-top: 5px;
    }

    .dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--surface-active);
      border: 2px solid var(--border-strong);
      flex-shrink: 0;
    }

    .dot-latest {
      background: var(--accent);
      border-color: var(--accent);
    }

    .line {
      flex: 1;
      width: 1px;
      background: var(--border-default);
      min-height: 12px;
      margin: 3px 0;
    }

    /* ── Item content ────────────────────────────────────────────── */
    .item-content {
      flex: 1;
      min-width: 0;
      display: flex;
      flex-direction: column;
      gap: 5px;
      padding: 4px 0 14px;
    }

    .item-row-1 {
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
    }

    .cp-id {
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--accent);
      background: var(--accent-soft);
      border-radius: 4px;
      padding: 1px 5px;
    }

    .cp-time {
      font-size: 10px;
      color: var(--text-faint);
    }

    .latest-badge {
      background: var(--accent);
      color: var(--accent-contrast);
      border-radius: var(--radius-pill);
      padding: 1px 7px;
      font-size: 10px;
      font-weight: 600;
    }

    .query-preview {
      font-size: 11.5px;
      color: var(--text-tertiary);
      line-height: 1.5;
      word-break: break-word;
    }

    /* ── Node pills ──────────────────────────────────────────────── */
    .nodes-row {
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
    }

    .node-pill {
      background: var(--surface-raised);
      border: 1px solid var(--border-subtle);
      border-radius: 4px;
      padding: 1px 6px;
      font-size: 10px;
      color: var(--text-tertiary);
      font-family: var(--font-mono);
    }

    /* ── Action buttons ──────────────────────────────────────────── */
    .action-row {
      display: flex;
      gap: 6px;
    }

    .action-btn {
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 3px 10px;
      font-size: 11.5px;
      cursor: pointer;
      font-family: inherit;
      background: var(--surface-raised);
      color: var(--text-secondary);
    }

    .action-btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .action-btn:not(:disabled):hover {
      background: var(--surface-hover);
      color: var(--text-primary);
      border-color: var(--border-default);
    }

    /* ── Toast ───────────────────────────────────────────────────── */
    .toast {
      position: sticky;
      bottom: 0;
      padding: 10px 16px;
      font-size: 12px;
      border-top: 1px solid var(--border-subtle);
    }

    .toast-success {
      background: var(--success-soft);
      color: var(--success);
    }
  `],
})
export class CheckpointTimelineComponent implements OnChanges {
  @Input() sessionId: string = '';
  @Input() visible: boolean = false;

  @Output() replayRequested = new EventEmitter<string>();
  @Output() branchCreated = new EventEmitter<string>();
  @Output() closed = new EventEmitter<void>();

  checkpoints: CheckpointSummary[] = [];
  loading = false;
  loadError: string | null = null;
  pendingCp: string | null = null;
  branchSuccessMsg: string | null = null;

  private loaded = false;

  constructor(
    private http: HttpClient,
    private cdr: ChangeDetectorRef,
  ) {}

  ngOnChanges(changes: SimpleChanges): void {
    // Auto-load when panel becomes visible and hasn't loaded yet
    if (changes['visible']?.currentValue === true && !this.loaded) {
      this.load();
    }
    // Reload when session changes
    if (changes['sessionId'] && !changes['sessionId'].firstChange) {
      this.loaded = false;
      if (this.visible) this.load();
    }
  }

  load(): void {
    if (!this.sessionId) return;
    this.loading = true;
    this.loadError = null;
    this.checkpoints = [];
    this.cdr.markForCheck();

    const token = localStorage.getItem('auth_token');
    const headers: Record<string, string> = {};
    if (token) headers['Authorization'] = `Bearer ${token}`;

    this.http
      .get<{ checkpoints: CheckpointSummary[] }>(
        `${environment.apiUrl}${environment.endpoints.sessions}/${this.sessionId}/checkpoints`,
        { headers },
      )
      .subscribe({
        next: (res) => {
          this.checkpoints = res.checkpoints ?? [];
          this.loading = false;
          this.loaded = true;
          this.cdr.markForCheck();
        },
        error: (err) => {
          this.loadError =
            err?.error?.detail ?? err?.message ?? 'Failed to load checkpoints';
          this.loading = false;
          this.cdr.markForCheck();
        },
      });
  }

  onReplay(cp: CheckpointSummary): void {
    this.replayRequested.emit(cp.checkpoint_id);
  }

  onBranch(cp: CheckpointSummary): void {
    if (!this.sessionId) return;
    this.pendingCp = cp.checkpoint_id;
    this.branchSuccessMsg = null;
    this.cdr.markForCheck();

    const token = localStorage.getItem('auth_token');
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (token) headers['Authorization'] = `Bearer ${token}`;

    this.http
      .post<{ branch_session_id: string }>(
        `${environment.apiUrl}${environment.endpoints.sessions}/${this.sessionId}/branch`,
        { graph_checkpoint_id: cp.checkpoint_id },
        { headers },
      )
      .subscribe({
        next: (res) => {
          this.pendingCp = null;
          this.branchSuccessMsg = `Branch created: ${res.branch_session_id.slice(0, 20)}…`;
          this.branchCreated.emit(res.branch_session_id);
          // Auto-hide toast after 4 s
          setTimeout(() => {
            this.branchSuccessMsg = null;
            this.cdr.markForCheck();
          }, 4000);
          this.cdr.markForCheck();
        },
        error: (err) => {
          this.pendingCp = null;
          this.loadError =
            err?.error?.detail ?? err?.message ?? 'Branch failed';
          this.cdr.markForCheck();
        },
      });
  }

  formatDate(iso: string): string {
    try {
      const d = new Date(iso);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
      return iso;
    }
  }

  trackById(_index: number, cp: CheckpointSummary): string {
    return cp.checkpoint_id;
  }
}
