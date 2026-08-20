import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';

/**
 * A conversation is a thin local index entry pointing at a real backend
 * session. Message content is never duplicated here — it is always re-fetched
 * from `GET /api/v2/chat/history/{session_id}`.
 */
export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
}

export interface ConversationGroup {
  label: string;
  items: Conversation[];
}

const INDEX_PREFIX = 'codelens.conversations.v1';
const ACTIVE_PREFIX = 'codelens.activeConversation.v1';
/** Written by earlier builds; adopted once so existing history stays reachable. */
const LEGACY_SESSION_KEY = 'chat_session_id';

/** Auto-assigned titles that the first real message is allowed to overwrite. */
const PLACEHOLDER_TITLES = new Set(['New chat', 'Previous conversation']);

@Injectable({ providedIn: 'root' })
export class ConversationStoreService {
  private readonly _conversations = new BehaviorSubject<Conversation[]>([]);
  readonly conversations$: Observable<Conversation[]> =
    this._conversations.asObservable();

  private readonly _activeId = new BehaviorSubject<string | null>(null);
  readonly activeId$: Observable<string | null> = this._activeId.asObservable();

  private scope = 'anonymous';

  constructor() {
    this.reloadForCurrentUser();
  }

  // ── Lifecycle ────────────────────────────────────────────────────────────

  /** Re-reads the index for whoever is currently signed in. */
  reloadForCurrentUser(): void {
    this.scope = this._resolveScope();
    const list = this._read();

    const legacy = this._safeGet(LEGACY_SESSION_KEY);
    if (legacy && !list.some((c) => c.id === legacy)) {
      list.unshift({
        id: legacy,
        title: 'Previous conversation',
        createdAt: Date.now(),
        updatedAt: Date.now(),
      });
      this._write(list);
    }

    this._conversations.next(list);
    const active = this._safeGet(this._activeKey());
    this._activeId.next(active && list.some((c) => c.id === active) ? active : null);
  }

  // ── Reads ────────────────────────────────────────────────────────────────

  get conversations(): Conversation[] {
    return this._conversations.value;
  }

  get activeId(): string | null {
    return this._activeId.value;
  }

  get(id: string): Conversation | undefined {
    return this._conversations.value.find((c) => c.id === id);
  }

  /** Newest-first buckets used by the sidebar. */
  group(items: Conversation[]): ConversationGroup[] {
    const now = new Date();
    const startOfToday = new Date(
      now.getFullYear(),
      now.getMonth(),
      now.getDate(),
    ).getTime();
    const day = 86_400_000;

    const buckets: ConversationGroup[] = [
      { label: 'Today', items: [] },
      { label: 'Yesterday', items: [] },
      { label: 'Previous 7 days', items: [] },
      { label: 'Previous 30 days', items: [] },
      { label: 'Older', items: [] },
    ];

    for (const c of [...items].sort((a, b) => b.updatedAt - a.updatedAt)) {
      if (c.updatedAt >= startOfToday) buckets[0].items.push(c);
      else if (c.updatedAt >= startOfToday - day) buckets[1].items.push(c);
      else if (c.updatedAt >= startOfToday - 7 * day) buckets[2].items.push(c);
      else if (c.updatedAt >= startOfToday - 30 * day) buckets[3].items.push(c);
      else buckets[4].items.push(c);
    }

    return buckets.filter((b) => b.items.length > 0);
  }

  // ── Writes ───────────────────────────────────────────────────────────────

  /** Creates a session id without persisting it — see `commit()`. */
  createDraftId(): string {
    return `session-${crypto.randomUUID()}`;
  }

  /**
   * Persists a draft session the first time it carries a real message and
   * derives its title from that message.
   */
  commit(id: string, firstMessage: string): void {
    const list = [...this._conversations.value];
    const existing = list.find((c) => c.id === id);
    const now = Date.now();

    if (existing) {
      existing.updatedAt = now;
      if (!existing.title || PLACEHOLDER_TITLES.has(existing.title)) {
        existing.title = this._titleFrom(firstMessage);
      }
    } else {
      list.unshift({
        id,
        title: this._titleFrom(firstMessage),
        createdAt: now,
        updatedAt: now,
      });
    }

    this._write(list);
    this._conversations.next(list);
  }

  touch(id: string): void {
    const list = [...this._conversations.value];
    const existing = list.find((c) => c.id === id);
    if (!existing) return;
    existing.updatedAt = Date.now();
    this._write(list);
    this._conversations.next(list);
  }

  rename(id: string, title: string): void {
    const clean = title.trim();
    if (!clean) return;
    const list = [...this._conversations.value];
    const existing = list.find((c) => c.id === id);
    if (!existing) return;
    existing.title = clean.slice(0, 120);
    this._write(list);
    this._conversations.next(list);
  }

  /**
   * Drops the conversation from this browser's index. The backend keeps no
   * delete endpoint, so server-side history is intentionally left untouched.
   */
  remove(id: string): void {
    const list = this._conversations.value.filter((c) => c.id !== id);
    this._write(list);
    this._conversations.next(list);
    if (this._activeId.value === id) this.setActive(null);
  }

  setActive(id: string | null): void {
    this._activeId.next(id);
    try {
      if (id) localStorage.setItem(this._activeKey(), id);
      else localStorage.removeItem(this._activeKey());
      // Kept in sync so a mid-migration reload still lands on the same session.
      if (id) localStorage.setItem(LEGACY_SESSION_KEY, id);
    } catch {
      /* ignore */
    }
  }

  // ── Internals ────────────────────────────────────────────────────────────

  private _titleFrom(message: string): string {
    const flat = message.replace(/\s+/g, ' ').trim();
    if (!flat) return 'New chat';
    return flat.length > 60 ? `${flat.slice(0, 57)}…` : flat;
  }

  private _indexKey(): string {
    return `${INDEX_PREFIX}::${this.scope}`;
  }

  private _activeKey(): string {
    return `${ACTIVE_PREFIX}::${this.scope}`;
  }

  private _read(): Conversation[] {
    const raw = this._safeGet(this._indexKey());
    if (!raw) return [];
    try {
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed.filter(
        (c): c is Conversation =>
          typeof c?.id === 'string' && typeof c?.title === 'string',
      );
    } catch {
      return [];
    }
  }

  private _write(list: Conversation[]): void {
    try {
      localStorage.setItem(this._indexKey(), JSON.stringify(list));
    } catch {
      /* quota or private mode — index simply stays in memory */
    }
  }

  private _safeGet(key: string): string | null {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  }

  /** Namespaces the index per signed-in user so accounts never share a list. */
  private _resolveScope(): string {
    const token = this._safeGet('auth_token');
    if (!token) return 'anonymous';
    try {
      const payload = token.split('.')[1];
      if (!payload) return 'anonymous';
      const json = atob(payload.replace(/-/g, '+').replace(/_/g, '/'));
      const claims = JSON.parse(json);
      return String(claims?.sub ?? claims?.email ?? 'anonymous');
    } catch {
      return 'anonymous';
    }
  }
}
