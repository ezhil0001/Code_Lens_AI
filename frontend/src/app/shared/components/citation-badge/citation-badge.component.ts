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
         [title]="'Source: ' + citation.sourceFile + ':' + citation.lineStart">
      <span class="citation-icon">📄</span>
      <span class="citation-file">{{ citation.sourceFile | shortPath }}</span>
      <span class="citation-lines" *ngIf="citation.lineStart">
        ({{ citation.lineStart }}:{{ citation.lineEnd }})
      </span>
      <span class="citation-score" *ngIf="citation.relevanceScore">
        {{ (citation.relevanceScore * 100).toFixed(0) }}%
      </span>
    </div>
  `,
  styles: [`
    .citation-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: #1f2937;
      border: 1px solid #374151;
      border-radius: 6px;
      padding: 6px 10px;
      margin: 4px;
      font-size: 0.85em;
      color: #9ca3af;
      cursor: pointer;
      transition: all 0.2s;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .citation-badge:hover {
      background: #374151;
      border-color: #4b5563;
      color: #d1d5db;
    }

    .citation-icon {
      font-size: 1em;
    }

    .citation-file {
      font-family: monospace;
      color: #60a5fa;
      font-weight: 500;
    }

    .citation-lines {
      color: #9ca3af;
      font-size: 0.9em;
    }

    .citation-score {
      background: #1e3a8a;
      color: #60a5fa;
      padding: 2px 6px;
      border-radius: 3px;
      font-size: 0.8em;
      font-weight: 600;
    }
  `]
})
export class CitationBadgeComponent {
  @Input() citation!: Citation;
}
