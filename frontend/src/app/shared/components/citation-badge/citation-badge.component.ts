import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Citation } from '../../../data/models/message.model';
import { ShortPathPipe } from '../../pipes/short-path.pipe';

/**
 * CitationBadgeComponent
 * Displays source references for RAG responses
 * Shows file path, line numbers, and language
 */
@Component({
  selector: 'app-citation-badge',
  standalone: true,
  imports: [CommonModule, ShortPathPipe],
  template: `
    <div class="citation-badge"
         [title]="'Source: ' + citation.sourceFile + (citation.lineStart ? ':' + citation.lineStart : '')">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <path d="M14 2v6h6"/>
      </svg>
      <span class="citation-file">{{ citation.sourceFile | shortPath }}</span>
      <span class="citation-lines" *ngIf="citation.lineStart">
        {{ citation.lineStart }}–{{ citation.lineEnd }}
      </span>
      <span class="citation-score" *ngIf="citation.relevanceScore">
        {{ (citation.relevanceScore * 100).toFixed(0) }}%
      </span>
    </div>
  `,
  styles: [`
    :host { display: inline-flex; max-width: 100%; }

    .citation-badge {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      max-width: 100%;
      padding: 5px 10px;
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-pill);
      background: var(--surface-raised);
      color: var(--text-tertiary);
      font-size: 12.5px;
      line-height: 1.4;
      cursor: default;
      transition: border-color 0.15s var(--ease), color 0.15s var(--ease);
    }

    .citation-badge:hover {
      border-color: var(--border-strong);
      color: var(--text-secondary);
    }

    .citation-file {
      font-family: var(--font-mono);
      font-size: 12px;
      color: var(--text-primary);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .citation-lines {
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text-faint);
    }

    .citation-score {
      padding: 1px 6px;
      border-radius: var(--radius-pill);
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 11px;
      font-weight: 600;
    }
  `]
})
export class CitationBadgeComponent {
  @Input() citation!: Citation;
}
