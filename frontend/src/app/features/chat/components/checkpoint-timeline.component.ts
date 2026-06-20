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
        <span style="color:#4b5563">No checkpoints saved for this session yet.</span>
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
      background: #1a1f2e;
      border-left: 1px solid #2d3748;
      font-size: 12px;
      color: #e2e8f0;
    }

    /* ── Header ──────────────────────────────────────────────────── */
    .timeline-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 14px;
      border-bottom: 1px solid #2d3748;
      background: #1e2534;
      flex-shrink: 0;
    }

    .timeline-title {
      font-weight: 600;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #94a3b8;
    }

    .close-btn {
      background: none;
      border: none;
      color: #6b7280;
      font-size: 18px;
      cursor: pointer;
      padding: 0 4px;
      line-height: 1;
      transition: color 0.15s;
    }

    .close-btn:hover { color: #e2e8f0; }

    /* ── Center states ───────────────────────────────────────────── */
    .center-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 12px;
      padding: 40px 20px;
      color: #6b7280;
      font-size: 12px;
    }

    .error-state { color: #f87171; }

    .spinner-lg {
      width: 24px;
      height: 24px;
      border: 3px solid #374151;
      border-top-color: #667eea;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
    }

    @keyframes spin { to { transform: rotate(360deg); } }

    .retry-btn {
      background: #374151;
      border: none;
      color: #e2e8f0;
      border-radius: 6px;
      padding: 5px 12px;
      font-size: 12px;
      cursor: pointer;
      transition: background 0.15s;
    }

    .retry-btn:hover { background: #4b5563; }

    /* ── Timeline list ───────────────────────────────────────────── */
    .timeline-list {
      flex: 1;
      overflow-y: auto;
      padding: 16px 0 16px;
    }

    .timeline-list::-webkit-scrollbar { width: 4px; }
    .timeline-list::-webkit-scrollbar-track { background: transparent; }
    .timeline-list::-webkit-scrollbar-thumb { background: #374151; border-radius: 2px; }

    /* ── Timeline item ───────────────────────────────────────────── */
    .timeline-item {
      display: flex;
      gap: 10px;
      padding: 0 14px 0 14px;
    }

    .item-latest {
      background: rgba(102, 126, 234, 0.04);
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
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #374151;
      border: 2px solid #4b5563;
      flex-shrink: 0;
    }

    .dot-latest {
      background: #667eea;
      border-color: #667eea;
    }

    .line {
      flex: 1;
      width: 2px;
      background: #1f2937;
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
      font-family: 'JetBrains Mono', 'Fira Code', monospace;
      font-size: 11px;
      color: #667eea;
      background: rgba(102, 126, 234, 0.1);
      border-radius: 3px;
      padding: 1px 5px;
    }

    .cp-time {
      font-size: 10px;
      color: #4b5563;
    }

    .latest-badge {
      background: #667eea;
      color: #fff;
      border-radius: 4px;
      padding: 1px 6px;
      font-size: 10px;
      font-weight: 600;
    }

    .query-preview {
      font-size: 11px;
      color: #9ca3af;
      font-style: italic;
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
      background: #1f2937;
      border: 1px solid #374151;
      border-radius: 4px;
      padding: 1px 6px;
      font-size: 10px;
      color: #9ca3af;
      font-family: 'JetBrains Mono', monospace;
    }

    /* ── Action buttons ──────────────────────────────────────────── */
    .action-row {
      display: flex;
      gap: 6px;
    }

    .action-btn {
      border: none;
      border-radius: 5px;
      padding: 3px 10px;
      font-size: 11px;
      cursor: pointer;
      transition: opacity 0.15s, background 0.15s;
      font-family: inherit;
    }

    .action-btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .replay-btn {
      background: rgba(102, 126, 234, 0.15);
      color: #818cf8;
      border: 1px solid rgba(102, 126, 234, 0.3);
    }

    .replay-btn:not(:disabled):hover {
      background: rgba(102, 126, 234, 0.25);
    }

    .branch-btn {
      background: rgba(16, 185, 129, 0.1);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.25);
    }

    .branch-btn:not(:disabled):hover {
      background: rgba(16, 185, 129, 0.2);
    }

    /* ── Toast ───────────────────────────────────────────────────── */
    .toast {
      position: sticky;
      bottom: 0;
      padding: 10px 14px;
      font-size: 12px;
      border-top: 1px solid #1f2937;
    }

    .toast-success {
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
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

    const token = localStorage.getItem('access_token');
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

    const token = localStorage.getItem('access_token');
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
