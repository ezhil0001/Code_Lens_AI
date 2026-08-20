import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  EventEmitter,
  Output,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { IngestService } from '../../../core/services/ingest.service';
import { IconComponent } from '../../../shared/components/icon/icon.component';

/** Extensions accepted by the ingestion endpoint. */
const ALLOWED_EXT = new Set([
  '.md', '.txt', '.pdf',
  '.py', '.ts', '.js', '.jsx', '.tsx',
  '.java', '.cpp', '.c', '.h', '.cc', '.cxx',
  '.go', '.rs', '.rb', '.php', '.cs', '.swift',
  '.kt', '.scala', '.r', '.m',
  '.sh', '.bash',
  '.yaml', '.yml', '.json', '.toml', '.xml',
  '.html', '.css', '.scss', '.sql',
]);

/** Modal for adding sources to the retrieval index (files or a URL). */
@Component({
  selector: 'app-knowledge-panel',
  standalone: true,
  imports: [CommonModule, FormsModule, IconComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="scrim" (click)="close.emit()"></div>

    <div class="dialog" role="dialog" aria-modal="true" aria-label="Knowledge base">
      <header>
        <div>
          <h2>Knowledge base</h2>
          <p>Add code or documents for CodeLens to retrieve from.</p>
        </div>
        <button class="icon-btn" type="button" (click)="close.emit()" aria-label="Close">
          <app-icon name="x" [size]="18"></app-icon>
        </button>
      </header>

      <label
        class="dropzone"
        [class.dragging]="isDragOver"
        (dragover)="onDragOver($event)"
        (dragleave)="isDragOver = false"
        (drop)="onDrop($event)"
      >
        <input
          type="file"
          multiple
          [accept]="acceptAttr"
          (change)="onFilesSelected($event)"
          [disabled]="busy"
        />
        <app-icon name="upload" [size]="22"></app-icon>
        <span class="dz-title">Drop files here or click to browse</span>
        <span class="dz-sub">Code, markdown, PDF, config and data files</span>
      </label>

      <div class="divider"><span>or</span></div>

      <div class="url-row">
        <input
          type="url"
          [(ngModel)]="url"
          placeholder="https://example.com/docs/page"
          [disabled]="busy"
          (keydown.enter)="ingestUrl()"
          aria-label="URL to ingest"
        />
        <button type="button" (click)="ingestUrl()" [disabled]="!url.trim() || busy">
          {{ busy ? 'Working…' : 'Add URL' }}
        </button>
      </div>

      <div class="rejected" *ngIf="rejected.length">
        <strong>Skipped {{ rejected.length }} unsupported file(s)</strong>
        <ul>
          <li *ngFor="let name of rejected">{{ name }}</li>
        </ul>
      </div>

      <p class="status" *ngIf="status" [class.ok]="statusOk" [class.bad]="!statusOk">
        {{ status }}
      </p>
    </div>
  `,
  styles: [
    `
      :host {
        position: fixed;
        inset: 0;
        z-index: 80;
        display: grid;
        place-items: center;
        padding: 20px;
      }

      .scrim {
        position: absolute;
        inset: 0;
        background: rgba(0, 0, 0, 0.42);
        backdrop-filter: blur(2px);
        animation: fade 0.15s var(--ease);
      }

      @keyframes fade {
        from { opacity: 0; }
        to { opacity: 1; }
      }

      .dialog {
        position: relative;
        width: min(560px, 100%);
        max-height: 90vh;
        overflow-y: auto;
        padding: 20px;
        border: 1px solid var(--border-default);
        border-radius: var(--radius-lg);
        background: var(--surface-raised);
        box-shadow: var(--shadow-lg);
        animation: pop 0.16s var(--ease);
      }

      @keyframes pop {
        from { opacity: 0; transform: translateY(8px) scale(0.99); }
        to { opacity: 1; transform: none; }
      }

      header {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 18px;
      }

      h2 {
        margin: 0;
        font-size: 17px;
        font-weight: 600;
        color: var(--text-primary);
      }

      header p {
        margin: 3px 0 0;
        font-size: 13.5px;
        color: var(--text-tertiary);
      }

      .icon-btn {
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

      .icon-btn:hover {
        background: var(--surface-hover);
        color: var(--text-primary);
      }

      .dropzone {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 6px;
        padding: 26px 20px;
        border: 1px dashed var(--border-default);
        border-radius: var(--radius-md);
        background: var(--surface-hover);
        color: var(--text-tertiary);
        text-align: center;
        cursor: pointer;
        transition: border-color 0.15s var(--ease), background 0.15s var(--ease);
      }

      .dropzone:hover,
      .dropzone.dragging {
        border-color: var(--accent-border);
        background: var(--accent-soft);
      }

      .dropzone input {
        display: none;
      }

      .dz-title {
        font-size: 14px;
        font-weight: 550;
        color: var(--text-primary);
      }

      .dz-sub {
        font-size: 12.5px;
      }

      .divider {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 16px 0;
        color: var(--text-faint);
        font-size: 12px;
      }

      .divider::before,
      .divider::after {
        content: '';
        flex: 1;
        height: 1px;
        background: var(--border-subtle);
      }

      .url-row {
        display: flex;
        gap: 8px;
      }

      .url-row input {
        flex: 1;
        min-width: 0;
        height: 38px;
        padding: 0 12px;
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        background: var(--surface-app);
        font-size: 14px;
        outline: none;
      }

      .url-row input:focus {
        border-color: var(--accent-border);
        box-shadow: 0 0 0 3px var(--accent-soft);
      }

      .url-row button {
        height: 38px;
        padding: 0 16px;
        border: none;
        border-radius: var(--radius-md);
        background: var(--accent);
        color: var(--accent-contrast);
        font-size: 14px;
        font-weight: 550;
        cursor: pointer;
      }

      .url-row button:disabled {
        background: var(--surface-active);
        color: var(--text-faint);
      }

      .rejected {
        margin-top: 16px;
        padding: 10px 12px;
        border-radius: var(--radius-md);
        background: var(--warning-soft);
        color: var(--text-secondary);
        font-size: 12.5px;
      }

      .rejected ul {
        margin: 6px 0 0;
        padding-left: 18px;
      }

      .status {
        margin: 16px 0 0;
        font-size: 13.5px;
      }

      .status.ok {
        color: var(--success);
      }

      .status.bad {
        color: var(--danger);
      }

      @media (max-width: 640px) {
        :host { padding: 0; align-items: flex-end; }
        .dialog {
          width: 100%;
          max-height: 92vh;
          border-radius: var(--radius-lg) var(--radius-lg) 0 0;
        }
      }
    `,
  ],
})
export class KnowledgePanelComponent {
  @Output() close = new EventEmitter<void>();

  readonly acceptAttr = [...ALLOWED_EXT].join(',');

  url = '';
  busy = false;
  isDragOver = false;
  status = '';
  statusOk = true;
  rejected: string[] = [];

  constructor(
    private ingest: IngestService,
    private cdr: ChangeDetectorRef,
  ) {}

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragOver = true;
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragOver = false;
    const files = event.dataTransfer ? Array.from(event.dataTransfer.files) : [];
    if (files.length) this.process(files);
  }

  onFilesSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files?.length) this.process(Array.from(input.files));
    input.value = '';
  }

  ingestUrl(): void {
    const target = this.url.trim();
    if (!target || this.busy) return;

    this.busy = true;
    this.status = 'Fetching and indexing…';
    this.statusOk = true;
    this.cdr.markForCheck();

    this.ingest.ingestFromUrl(target).subscribe({
      next: () => {
        this.url = '';
        this.finish('URL indexed successfully.', true);
      },
      error: (err) => this.finish(this.describe(err, 'Ingestion failed'), false),
    });
  }

  private process(files: File[]): void {
    const valid: File[] = [];
    const rejected: string[] = [];

    for (const file of files) {
      const ext = `.${file.name.split('.').pop()?.toLowerCase() ?? ''}`;
      if (ALLOWED_EXT.has(ext)) valid.push(file);
      else rejected.push(file.name);
    }

    this.rejected = rejected;

    if (!valid.length) {
      this.status = 'No supported files selected.';
      this.statusOk = false;
      this.cdr.markForCheck();
      return;
    }

    this.busy = true;
    this.status = `Uploading ${valid.length} file(s)…`;
    this.statusOk = true;
    this.cdr.markForCheck();

    this.ingest.uploadDocuments(valid).subscribe({
      next: (response) => {
        const backendErrors: string[] = response?.errors ?? [];
        if (backendErrors.length) this.rejected = [...this.rejected, ...backendErrors];
        const accepted = response?.files_ingested ?? valid.length;
        this.finish(`Indexed ${accepted} file(s).`, true);
      },
      error: (err) => this.finish(this.describe(err, 'Upload failed'), false),
    });
  }

  private finish(message: string, ok: boolean): void {
    this.busy = false;
    this.status = message;
    this.statusOk = ok;
    this.cdr.markForCheck();
  }

  private describe(err: unknown, fallback: string): string {
    const detail = (err as { error?: { detail?: string }; message?: string });
    return detail?.error?.detail ?? detail?.message ?? fallback;
  }
}
