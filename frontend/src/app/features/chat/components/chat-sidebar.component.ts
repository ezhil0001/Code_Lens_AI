import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  EventEmitter,
  HostListener,
  Input,
  OnDestroy,
  OnInit,
  Output,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';
import { IconComponent } from '../../../shared/components/icon/icon.component';
import {
  Conversation,
  ConversationGroup,
  ConversationStoreService,
} from '../../../core/services/conversation-store.service';
import { ThemeName, ThemeService } from '../../../core/services/theme.service';

/**
 * Workspace navigation rail: new chat, conversation search, grouped history,
 * and the account menu. Collapses to an icon rail on desktop and becomes an
 * overlay drawer below the tablet breakpoint.
 */
@Component({
  selector: 'app-chat-sidebar',
  standalone: true,
  imports: [CommonModule, FormsModule, IconComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <aside class="sidebar" [class.collapsed]="collapsed" [class.drawer-open]="drawerOpen">
      <!-- Brand -->
      <div class="brand-row">
        <ng-template #mark>
          <span class="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none"
                 stroke="currentColor" stroke-width="2" stroke-linecap="round"
                 stroke-linejoin="round">
              <circle cx="10.5" cy="10.5" r="6.5" />
              <path d="m19.5 19.5-4.2-4.2" />
            </svg>
          </span>
        </ng-template>

        <!-- Collapsed, the mark doubles as the expand control. -->
        <button
          *ngIf="collapsed"
          class="brand"
          type="button"
          (click)="toggleCollapsed.emit()"
          aria-label="Expand sidebar"
          title="Expand sidebar"
        >
          <ng-container *ngTemplateOutlet="mark"></ng-container>
        </button>

        <div class="brand" *ngIf="!collapsed">
          <ng-container *ngTemplateOutlet="mark"></ng-container>
          <span class="brand-name">CodeLens</span>
        </div>

        <button
          class="icon-btn collapse-btn"
          type="button"
          (click)="toggleCollapsed.emit()"
          [attr.aria-label]="collapsed ? 'Expand sidebar' : 'Collapse sidebar'"
          [title]="collapsed ? 'Expand sidebar' : 'Collapse sidebar'"
        >
          <app-icon name="panel-left" [size]="17"></app-icon>
        </button>

        <button
          class="icon-btn drawer-close"
          type="button"
          (click)="closeDrawer.emit()"
          aria-label="Close navigation"
        >
          <app-icon name="x" [size]="18"></app-icon>
        </button>
      </div>

      <!-- Primary actions -->
      <div class="actions">
        <button
          class="new-chat"
          type="button"
          (click)="newChat.emit()"
          [title]="collapsed ? 'New chat' : ''"
        >
          <app-icon name="new-chat" [size]="16"></app-icon>
          <span *ngIf="!collapsed">New chat</span>
        </button>

        <button
          *ngIf="collapsed"
          class="rail-btn"
          type="button"
          (click)="toggleCollapsed.emit()"
          title="Search chats"
        >
          <app-icon name="search" [size]="17"></app-icon>
        </button>

        <div class="search" *ngIf="!collapsed">
          <app-icon name="search" [size]="15"></app-icon>
          <input
            type="text"
            [(ngModel)]="query"
            (ngModelChange)="onQueryChange()"
            placeholder="Search chats"
            aria-label="Search conversations"
          />
          <button
            *ngIf="query"
            class="clear-search"
            type="button"
            (click)="clearQuery()"
            aria-label="Clear search"
          >
            <app-icon name="x" [size]="13"></app-icon>
          </button>
        </div>

        <button
          class="rail-btn knowledge"
          type="button"
          (click)="openKnowledge.emit()"
          [title]="collapsed ? 'Knowledge base' : ''"
        >
          <app-icon name="book" [size]="16"></app-icon>
          <span *ngIf="!collapsed">Knowledge base</span>
        </button>
      </div>

      <!-- History -->
      <nav class="history" [class.hidden]="collapsed" aria-label="Chat history">
        <p class="empty-hint" *ngIf="groups.length === 0 && !query">
          Your conversations will appear here.
        </p>
        <p class="empty-hint" *ngIf="groups.length === 0 && query">
          No chats match “{{ query }}”.
        </p>

        <ng-container *ngFor="let group of groups; trackBy: trackByLabel">
          <p class="group-label">{{ group.label }}</p>
          <div
            *ngFor="let item of group.items; trackBy: trackById"
            class="history-item"
            [class.active]="item.id === activeId"
          >
            <button class="history-link" type="button" (click)="select.emit(item.id)">
              <span class="history-title">{{ item.title }}</span>
            </button>

            <button
              class="row-menu-btn"
              type="button"
              (click)="toggleRowMenu(item.id, $event)"
              [attr.aria-label]="'Options for ' + item.title"
            >
              <app-icon name="more" [size]="15"></app-icon>
            </button>

            <div class="row-menu" *ngIf="openMenuId === item.id" role="menu">
              <button type="button" role="menuitem" (click)="startRename(item)">
                <app-icon name="edit" [size]="14"></app-icon> Rename
              </button>
              <button type="button" role="menuitem" class="danger" (click)="remove(item)">
                <app-icon name="trash" [size]="14"></app-icon> Remove from list
              </button>
            </div>
          </div>
        </ng-container>
      </nav>

      <div class="history-spacer" *ngIf="collapsed"></div>

      <!-- Account -->
      <div class="account">
        <button
          class="account-btn"
          type="button"
          (click)="toggleAccountMenu($event)"
          [title]="collapsed ? userLabel : ''"
        >
          <span class="avatar">{{ initial }}</span>
          <span class="account-text" *ngIf="!collapsed">
            <span class="account-name">{{ userLabel }}</span>
            <span class="account-plan">Workspace</span>
          </span>
          <app-icon *ngIf="!collapsed" name="chevron-down" [size]="14"></app-icon>
        </button>

        <div class="account-menu" *ngIf="accountMenuOpen" role="menu">
          <button type="button" role="menuitem" (click)="onToggleTheme()">
            <app-icon [name]="theme === 'dark' ? 'sun' : 'moon'" [size]="15"></app-icon>
            {{ theme === 'dark' ? 'Light theme' : 'Dark theme' }}
          </button>
          <button type="button" role="menuitem" (click)="openKnowledge.emit(); accountMenuOpen = false">
            <app-icon name="upload" [size]="15"></app-icon> Manage sources
          </button>
          <div class="menu-sep"></div>
          <button type="button" role="menuitem" class="danger" (click)="logout.emit()">
            <app-icon name="log-out" [size]="15"></app-icon> Sign out
          </button>
        </div>
      </div>
    </aside>
  `,
  styles: [
    `
      :host {
        display: contents;
      }

      .sidebar {
        display: flex;
        flex-direction: column;
        width: var(--sidebar-width);
        flex-shrink: 0;
        height: 100%;
        background: var(--surface-sidebar);
        border-right: 1px solid var(--border-subtle);
        transition: width 0.22s var(--ease);
        overflow: hidden;
      }

      .sidebar.collapsed {
        width: var(--sidebar-rail);
      }

      /* ── Brand ───────────────────────────────────────────────── */
      .brand-row {
        display: flex;
        align-items: center;
        gap: 2px;
        height: var(--header-height);
        padding: 0 8px 0 10px;
        flex-shrink: 0;
      }

      .brand {
        display: flex;
        align-items: center;
        gap: 9px;
        flex: 1;
        min-width: 0;
        background: none;
        border: none;
        padding: 0;
        cursor: default;
        color: var(--text-primary);
      }

      .collapsed .brand {
        cursor: pointer;
      }

      .brand-mark {
        display: grid;
        place-items: center;
        width: 26px;
        height: 26px;
        border-radius: 8px;
        background: var(--accent);
        color: var(--accent-contrast);
        flex-shrink: 0;
      }

      .brand-name {
        font-size: 15px;
        font-weight: 600;
        letter-spacing: -0.01em;
        white-space: nowrap;
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
        flex-shrink: 0;
      }

      .icon-btn:hover {
        background: var(--surface-hover);
        color: var(--text-primary);
      }

      .collapsed .collapse-btn {
        display: none;
      }

      .drawer-close {
        display: none;
      }

      /* ── Actions ─────────────────────────────────────────────── */
      .actions {
        display: flex;
        flex-direction: column;
        gap: 2px;
        padding: 4px 8px 8px;
        flex-shrink: 0;
      }

      .new-chat {
        display: flex;
        align-items: center;
        gap: 10px;
        width: 100%;
        padding: 8px 10px;
        border: none;
        border-radius: var(--radius-md);
        background: transparent;
        color: var(--accent);
        font-size: 14px;
        font-weight: 550;
        cursor: pointer;
        text-align: left;
      }

      .new-chat:hover {
        background: var(--surface-hover);
      }

      .rail-btn {
        display: flex;
        align-items: center;
        gap: 10px;
        width: 100%;
        padding: 8px 10px;
        border: none;
        border-radius: var(--radius-md);
        background: transparent;
        color: var(--text-secondary);
        font-size: 14px;
        cursor: pointer;
        text-align: left;
      }

      .rail-btn:hover {
        background: var(--surface-hover);
        color: var(--text-primary);
      }

      .collapsed .new-chat,
      .collapsed .rail-btn {
        justify-content: center;
        padding: 8px 0;
      }

      .search {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 4px 0;
        padding: 0 10px;
        height: 34px;
        border-radius: var(--radius-md);
        background: var(--surface-hover);
        color: var(--text-tertiary);
      }

      .search:focus-within {
        box-shadow: inset 0 0 0 1px var(--accent-border);
      }

      .search input {
        flex: 1;
        min-width: 0;
        border: none;
        background: none;
        outline: none;
        font-size: 13.5px;
        color: var(--text-primary);
      }

      .search input::placeholder {
        color: var(--text-faint);
      }

      .clear-search {
        display: grid;
        place-items: center;
        width: 18px;
        height: 18px;
        border: none;
        border-radius: 50%;
        background: var(--surface-active);
        color: var(--text-secondary);
        cursor: pointer;
      }

      /* ── History ─────────────────────────────────────────────── */
      .history {
        flex: 1;
        min-height: 0;
        overflow-y: auto;
        padding: 4px 8px 8px;
      }

      .history.hidden {
        display: none;
      }

      .history-spacer {
        flex: 1;
      }

      .group-label {
        margin: 14px 0 4px;
        padding: 0 10px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--text-faint);
      }

      .group-label:first-child {
        margin-top: 4px;
      }

      .empty-hint {
        margin: 12px 10px;
        font-size: 13px;
        line-height: 1.5;
        color: var(--text-faint);
      }

      .history-item {
        position: relative;
        display: flex;
        align-items: center;
        border-radius: var(--radius-md);
      }

      .history-item:hover,
      .history-item.active {
        background: var(--surface-hover);
      }

      .history-item.active {
        background: var(--surface-active);
      }

      .history-link {
        flex: 1;
        min-width: 0;
        display: block;
        padding: 7px 10px;
        border: none;
        background: none;
        color: var(--text-secondary);
        font-size: 13.5px;
        text-align: left;
        cursor: pointer;
      }

      .history-item.active .history-link {
        color: var(--text-primary);
      }

      .history-title {
        display: block;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .row-menu-btn {
        display: grid;
        place-items: center;
        width: 26px;
        height: 26px;
        margin-right: 4px;
        border: none;
        border-radius: var(--radius-sm);
        background: transparent;
        color: var(--text-tertiary);
        cursor: pointer;
        opacity: 0;
      }

      .history-item:hover .row-menu-btn,
      .history-item.active .row-menu-btn {
        opacity: 1;
      }

      .row-menu-btn:hover {
        background: var(--surface-active);
        color: var(--text-primary);
      }

      .row-menu,
      .account-menu {
        position: absolute;
        z-index: 40;
        min-width: 186px;
        padding: 4px;
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        background: var(--surface-overlay);
        box-shadow: var(--shadow-lg);
      }

      .row-menu {
        top: calc(100% - 2px);
        right: 4px;
      }

      .row-menu button,
      .account-menu button {
        display: flex;
        align-items: center;
        gap: 9px;
        width: 100%;
        padding: 7px 9px;
        border: none;
        border-radius: var(--radius-sm);
        background: none;
        color: var(--text-secondary);
        font-size: 13.5px;
        text-align: left;
        cursor: pointer;
      }

      .row-menu button:hover,
      .account-menu button:hover {
        background: var(--surface-hover);
        color: var(--text-primary);
      }

      .row-menu button.danger,
      .account-menu button.danger {
        color: var(--danger);
      }

      .menu-sep {
        height: 1px;
        margin: 4px 2px;
        background: var(--border-subtle);
      }

      /* ── Account ─────────────────────────────────────────────── */
      .account {
        position: relative;
        flex-shrink: 0;
        padding: 8px;
        border-top: 1px solid var(--border-subtle);
      }

      .account-btn {
        display: flex;
        align-items: center;
        gap: 9px;
        width: 100%;
        padding: 6px 8px;
        border: none;
        border-radius: var(--radius-md);
        background: none;
        color: var(--text-secondary);
        cursor: pointer;
        text-align: left;
      }

      .account-btn:hover {
        background: var(--surface-hover);
      }

      .collapsed .account-btn {
        justify-content: center;
        padding: 6px 0;
      }

      .avatar {
        display: grid;
        place-items: center;
        width: 26px;
        height: 26px;
        border-radius: 50%;
        background: var(--accent-soft);
        color: var(--accent);
        font-size: 12px;
        font-weight: 600;
        flex-shrink: 0;
        text-transform: uppercase;
      }

      .account-text {
        flex: 1;
        min-width: 0;
        display: flex;
        flex-direction: column;
        line-height: 1.25;
      }

      .account-name {
        font-size: 13px;
        color: var(--text-primary);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .account-plan {
        font-size: 11px;
        color: var(--text-faint);
      }

      .account-menu {
        bottom: calc(100% - 2px);
        left: 8px;
        right: 8px;
      }

      /* ── Responsive drawer ───────────────────────────────────── */
      @media (max-width: 900px) {
        .sidebar {
          position: fixed;
          top: 0;
          left: 0;
          bottom: 0;
          z-index: 60;
          width: min(84vw, 19rem);
          transform: translateX(-100%);
          transition: transform 0.24s var(--ease);
          box-shadow: var(--shadow-lg);
        }

        .sidebar.collapsed {
          width: min(84vw, 19rem);
        }

        .sidebar.drawer-open {
          transform: translateX(0);
        }

        .collapse-btn {
          display: none;
        }

        .drawer-close {
          display: grid;
        }

        .history-spacer {
          display: none;
        }
      }
    `,
  ],
})
export class ChatSidebarComponent implements OnInit, OnDestroy {
  @Input() collapsed = false;
  @Input() drawerOpen = false;
  @Input() userLabel = 'Signed in';

  @Output() newChat = new EventEmitter<void>();
  @Output() select = new EventEmitter<string>();
  @Output() toggleCollapsed = new EventEmitter<void>();
  @Output() closeDrawer = new EventEmitter<void>();
  @Output() openKnowledge = new EventEmitter<void>();
  @Output() logout = new EventEmitter<void>();

  groups: ConversationGroup[] = [];
  activeId: string | null = null;
  query = '';
  openMenuId: string | null = null;
  accountMenuOpen = false;
  theme: ThemeName = 'dark';

  private all: Conversation[] = [];
  private subs = new Subscription();

  constructor(
    private store: ConversationStoreService,
    private themeService: ThemeService,
    private cdr: ChangeDetectorRef,
  ) {}

  get initial(): string {
    return (this.userLabel || '?').trim().charAt(0) || '?';
  }

  ngOnInit(): void {
    this.subs.add(
      this.store.conversations$.subscribe((list) => {
        this.all = list;
        this.applyFilter();
        this.cdr.markForCheck();
      }),
    );
    this.subs.add(
      this.store.activeId$.subscribe((id) => {
        this.activeId = id;
        this.cdr.markForCheck();
      }),
    );
    this.subs.add(
      this.themeService.theme$.subscribe((t) => {
        this.theme = t;
        this.cdr.markForCheck();
      }),
    );
  }

  ngOnDestroy(): void {
    this.subs.unsubscribe();
  }

  @HostListener('document:click')
  onDocumentClick(): void {
    if (this.openMenuId || this.accountMenuOpen) {
      this.openMenuId = null;
      this.accountMenuOpen = false;
      this.cdr.markForCheck();
    }
  }

  onQueryChange(): void {
    this.applyFilter();
  }

  clearQuery(): void {
    this.query = '';
    this.applyFilter();
  }

  toggleRowMenu(id: string, event: MouseEvent): void {
    event.stopPropagation();
    this.openMenuId = this.openMenuId === id ? null : id;
    this.accountMenuOpen = false;
  }

  toggleAccountMenu(event: MouseEvent): void {
    event.stopPropagation();
    this.accountMenuOpen = !this.accountMenuOpen;
    this.openMenuId = null;
  }

  onToggleTheme(): void {
    this.themeService.toggle();
  }

  startRename(item: Conversation): void {
    this.openMenuId = null;
    const next = window.prompt('Rename conversation', item.title);
    if (next !== null) this.store.rename(item.id, next);
  }

  remove(item: Conversation): void {
    this.openMenuId = null;
    this.store.remove(item.id);
  }

  trackById(_i: number, item: Conversation): string {
    return item.id;
  }

  trackByLabel(_i: number, group: ConversationGroup): string {
    return group.label;
  }

  private applyFilter(): void {
    const q = this.query.trim().toLowerCase();
    const filtered = q
      ? this.all.filter((c) => c.title.toLowerCase().includes(q))
      : this.all;
    this.groups = this.store.group(filtered);
  }
}
