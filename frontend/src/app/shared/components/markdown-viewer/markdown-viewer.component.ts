import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  ElementRef,
  Input,
  OnChanges,
  SimpleChanges,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { MarkdownModule } from 'ngx-markdown';

/**
 * MarkdownViewerComponent
 *
 * Renders assistant markdown with Prism-highlighted code blocks. Each code
 * block gains a language label and a copy button, injected after render and
 * scoped to this component's host element only.
 */
@Component({
  selector: 'app-markdown-viewer',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [CommonModule, MarkdownModule],
  template: `
    <div class="markdown-content">
      <markdown [data]="content" (ready)="onMarkdownReady()"></markdown>
    </div>
  `,
  styles: [`
    :host {
      display: block;
      width: 100%;
    }

    .markdown-content {
      font-size: inherit;
      line-height: 1.7;
      color: var(--text-primary);
      overflow-wrap: anywhere;
    }

    /* ── Block rhythm ───────────────────────────────────────────── */
    .markdown-content ::ng-deep markdown > *:first-child { margin-top: 0; }
    .markdown-content ::ng-deep p { margin: 0 0 1em; }
    .markdown-content ::ng-deep p:last-child { margin-bottom: 0; }

    .markdown-content ::ng-deep h1,
    .markdown-content ::ng-deep h2,
    .markdown-content ::ng-deep h3,
    .markdown-content ::ng-deep h4,
    .markdown-content ::ng-deep h5,
    .markdown-content ::ng-deep h6 {
      margin: 1.6em 0 0.6em;
      font-weight: 600;
      line-height: 1.3;
      letter-spacing: -0.01em;
      color: var(--text-primary);
    }

    .markdown-content ::ng-deep h1 { font-size: 1.5em; }
    .markdown-content ::ng-deep h2 { font-size: 1.28em; }
    .markdown-content ::ng-deep h3 { font-size: 1.12em; }
    .markdown-content ::ng-deep h4,
    .markdown-content ::ng-deep h5,
    .markdown-content ::ng-deep h6 { font-size: 1em; }

    .markdown-content ::ng-deep ul,
    .markdown-content ::ng-deep ol {
      margin: 0 0 1em;
      padding-left: 1.5em;
    }

    .markdown-content ::ng-deep li { margin: 0.35em 0; }
    .markdown-content ::ng-deep li > ul,
    .markdown-content ::ng-deep li > ol { margin: 0.35em 0; }

    .markdown-content ::ng-deep hr {
      margin: 1.6em 0;
      border: none;
      border-top: 1px solid var(--border-subtle);
    }

    .markdown-content ::ng-deep blockquote {
      margin: 1em 0;
      padding: 0.1em 0 0.1em 1em;
      border-left: 2px solid var(--border-strong);
      color: var(--text-secondary);
    }

    .markdown-content ::ng-deep a {
      color: var(--accent);
      text-decoration: underline;
      text-underline-offset: 2px;
    }

    .markdown-content ::ng-deep a:hover { color: var(--accent-hover); }
    .markdown-content ::ng-deep strong { font-weight: 600; }

    /* ── Inline code ────────────────────────────────────────────── */
    .markdown-content ::ng-deep code {
      font-family: var(--font-mono);
      font-size: 0.875em;
      padding: 0.14em 0.36em;
      border-radius: 4px;
      background: var(--surface-hover);
      color: var(--text-primary);
    }

    /* ── Code blocks ────────────────────────────────────────────── */
    .markdown-content ::ng-deep .code-block {
      margin: 1.1em 0;
      border: 1px solid var(--code-border);
      border-radius: var(--radius-md);
      background: var(--code-surface);
      overflow: hidden;
    }

    .markdown-content ::ng-deep .code-block-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 6px 8px 6px 12px;
      border-bottom: 1px solid var(--code-border);
      background: var(--code-header);
    }

    .markdown-content ::ng-deep .code-block-lang {
      font-family: var(--font-mono);
      font-size: 11.5px;
      letter-spacing: 0.03em;
      color: #9b9890;
    }

    .markdown-content ::ng-deep .code-block-copy {
      padding: 3px 8px;
      border: none;
      border-radius: 5px;
      background: transparent;
      color: #b6b3aa;
      font-family: inherit;
      font-size: 11.5px;
      cursor: pointer;
    }

    .markdown-content ::ng-deep .code-block-copy:hover {
      background: rgba(255, 255, 255, 0.08);
      color: #f0eee7;
    }

    .markdown-content ::ng-deep pre {
      margin: 0;
      padding: 14px 16px;
      background: var(--code-surface) !important;
      overflow-x: auto;
    }

    .markdown-content ::ng-deep pre code {
      display: block;
      padding: 0;
      background: transparent;
      border-radius: 0;
      color: var(--code-text);
      font-size: 13.5px;
      line-height: 1.6;
      text-shadow: none;
    }

    /* ── Tables ─────────────────────────────────────────────────── */
    .markdown-content ::ng-deep .table-scroll {
      margin: 1.1em 0;
      overflow-x: auto;
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
    }

    .markdown-content ::ng-deep table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }

    .markdown-content ::ng-deep th,
    .markdown-content ::ng-deep td {
      padding: 8px 12px;
      border-bottom: 1px solid var(--border-subtle);
      text-align: left;
      vertical-align: top;
    }

    .markdown-content ::ng-deep th {
      background: var(--surface-hover);
      font-weight: 600;
      color: var(--text-primary);
      white-space: nowrap;
    }

    .markdown-content ::ng-deep tr:last-child td { border-bottom: none; }

    .markdown-content ::ng-deep img {
      max-width: 100%;
      border-radius: var(--radius-md);
    }

    /* ── Prism token colours ────────────────────────────────────── */
    .markdown-content ::ng-deep .token.comment,
    .markdown-content ::ng-deep .token.prolog,
    .markdown-content ::ng-deep .token.doctype,
    .markdown-content ::ng-deep .token.cdata { color: #7d7a72; font-style: italic; }
    .markdown-content ::ng-deep .token.punctuation { color: #b6b3aa; }
    .markdown-content ::ng-deep .token.keyword,
    .markdown-content ::ng-deep .token.operator,
    .markdown-content ::ng-deep .token.tag { color: #ff8a7d; }
    .markdown-content ::ng-deep .token.string,
    .markdown-content ::ng-deep .token.char,
    .markdown-content ::ng-deep .token.attr-value { color: #a8d8a0; }
    .markdown-content ::ng-deep .token.function,
    .markdown-content ::ng-deep .token.class-name { color: #d5b0ff; }
    .markdown-content ::ng-deep .token.number,
    .markdown-content ::ng-deep .token.boolean,
    .markdown-content ::ng-deep .token.constant { color: #8fc7ff; }
    .markdown-content ::ng-deep .token.property,
    .markdown-content ::ng-deep .token.attr-name,
    .markdown-content ::ng-deep .token.variable { color: #f4c98a; }
  `]
})
export class MarkdownViewerComponent implements OnChanges {
  @Input() content: string = '';

  constructor(
    private cdr: ChangeDetectorRef,
    private el: ElementRef<HTMLElement>,
  ) {}

  ngOnChanges(_changes: SimpleChanges): void {
    // When content input changes, mark for check so OnPush picks it up.
    this.cdr.markForCheck();
  }

  /**
   * Called when markdown finishes rendering.
   * Scoped to THIS component's host element — never touches sibling components.
   */
  onMarkdownReady(): void {
    const host = this.el.nativeElement;

    if (typeof window !== 'undefined' && (window as any).Prism) {
      (window as any).Prism.highlightAllUnder(host);
    }

    this.decorateCodeBlocks(host);
    this.wrapTables(host);
  }

  private decorateCodeBlocks(host: HTMLElement): void {
    host.querySelectorAll('pre').forEach((pre) => {
      if (pre.parentElement?.classList.contains('code-block')) return;

      const code = pre.querySelector('code');
      const language =
        Array.from(code?.classList ?? [])
          .find((c) => c.startsWith('language-'))
          ?.replace('language-', '') ?? 'text';

      const block = document.createElement('div');
      block.className = 'code-block';

      const header = document.createElement('div');
      header.className = 'code-block-header';

      const label = document.createElement('span');
      label.className = 'code-block-lang';
      label.textContent = language;

      const copyBtn = document.createElement('button');
      copyBtn.type = 'button';
      copyBtn.className = 'code-block-copy';
      copyBtn.textContent = 'Copy';
      copyBtn.addEventListener('click', () => {
        navigator.clipboard?.writeText(code?.textContent ?? '').then(() => {
          copyBtn.textContent = 'Copied';
          setTimeout(() => (copyBtn.textContent = 'Copy'), 1600);
        });
      });

      header.append(label, copyBtn);
      pre.replaceWith(block);
      block.append(header, pre);
    });
  }

  /** Keeps wide tables from stretching the conversation column. */
  private wrapTables(host: HTMLElement): void {
    host.querySelectorAll('table').forEach((table) => {
      if (table.parentElement?.classList.contains('table-scroll')) return;
      const scroller = document.createElement('div');
      scroller.className = 'table-scroll';
      table.replaceWith(scroller);
      scroller.append(table);
    });
  }
}
