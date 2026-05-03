import { Component, Input, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MarkdownModule } from 'ngx-markdown';

/**
 * MarkdownViewerComponent
 * Renders markdown with syntax-highlighted code blocks using Prism.js
 * Critical for displaying RAG responses with embedded code snippets
 */
@Component({
  selector: 'app-markdown-viewer',
  standalone: true,
  imports: [CommonModule, MarkdownModule],
  template: `
    <div class="markdown-content prose prose-invert max-w-none">
      <!-- Raw markdown content -->
      <markdown 
        [data]="content" 
        (ready)="onMarkdownReady()">
      </markdown>
    </div>
  `,
  styles: [`
    :host {
      display: block;
      width: 100%;
    }

    .markdown-content {
      font-family: system-ui, -apple-system, sans-serif;
      line-height: 1.6;
      color: #e0e0e0;
    }

    .markdown-content :deep(pre) {
      background: #1e1e1e;
      border: 1px solid #333;
      border-radius: 8px;
      padding: 16px;
      overflow-x: auto;
      margin: 12px 0;
    }

    .markdown-content :deep(code) {
      font-family: 'Fira Code', 'Courier New', monospace;
      font-size: 0.875em;
      background: #2d2d2d;
      padding: 2px 6px;
      border-radius: 3px;
    }

    .markdown-content :deep(pre code) {
      background: transparent;
      padding: 0;
    }

    .markdown-content :deep(h1, h2, h3, h4, h5, h6) {
      color: #fff;
      margin-top: 24px;
      margin-bottom: 12px;
      font-weight: 600;
    }

    .markdown-content :deep(h1) { font-size: 1.875em; }
    .markdown-content :deep(h2) { font-size: 1.5em; }
    .markdown-content :deep(h3) { font-size: 1.25em; }

    .markdown-content :deep(a) {
      color: #60a5fa;
      text-decoration: underline;
    }

    .markdown-content :deep(a:hover) {
      color: #93c5fd;
    }

    .markdown-content :deep(blockquote) {
      border-left: 4px solid #3b82f6;
      padding-left: 12px;
      color: #a0aec0;
      margin: 12px 0;
    }

    .markdown-content :deep(ul, ol) {
      margin: 12px 0;
      padding-left: 24px;
    }

    .markdown-content :deep(li) {
      margin: 6px 0;
    }

    .markdown-content :deep(table) {
      border-collapse: collapse;
      width: 100%;
      margin: 12px 0;
    }

    .markdown-content :deep(table td),
    .markdown-content :deep(table th) {
      border: 1px solid #444;
      padding: 8px;
      text-align: left;
    }

    .markdown-content :deep(table th) {
      background: #2d2d2d;
      font-weight: 600;
    }

    /* Prism.js syntax highlighting */
    .markdown-content :deep(.token.keyword) { color: #ff7b72; }
    .markdown-content :deep(.token.string) { color: #a5d6ff; }
    .markdown-content :deep(.token.function) { color: #d2a8ff; }
    .markdown-content :deep(.token.comment) { color: #6e7681; font-style: italic; }
    .markdown-content :deep(.token.number) { color: #79c0ff; }
    .markdown-content :deep(.token.operator) { color: #ff7b72; }
  `]
})
export class MarkdownViewerComponent implements OnInit {
  @Input() content: string = '';

  ngOnInit() {
    // Content loaded
  }

  /**
   * Called when markdown finishes rendering
   * Trigger Prism.js syntax highlighting
   */
  onMarkdownReady() {
    // Trigger Prism.js highlight on all code blocks
    if (typeof window !== 'undefined' && (window as any).Prism) {
      (window as any).Prism.highlightAllUnder(document.querySelector('.markdown-content'));
    }
  }
}
