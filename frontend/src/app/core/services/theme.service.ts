import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';

export type ThemeName = 'dark' | 'light';

const STORAGE_KEY = 'codelens.theme';

/** Applies and persists the colour scheme via the `data-theme` root attribute. */
@Injectable({ providedIn: 'root' })
export class ThemeService {
  private readonly _theme = new BehaviorSubject<ThemeName>(this._read());
  readonly theme$: Observable<ThemeName> = this._theme.asObservable();

  constructor() {
    this.apply(this._theme.value);
  }

  get theme(): ThemeName {
    return this._theme.value;
  }

  toggle(): void {
    this.set(this._theme.value === 'dark' ? 'light' : 'dark');
  }

  set(theme: ThemeName): void {
    this._theme.next(theme);
    this.apply(theme);
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      /* storage unavailable — theme stays session-only */
    }
  }

  private apply(theme: ThemeName): void {
    document.documentElement.setAttribute('data-theme', theme);
  }

  private _read(): ThemeName {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored === 'dark' || stored === 'light') return stored;
    } catch {
      /* ignore */
    }
    return window.matchMedia?.('(prefers-color-scheme: light)').matches
      ? 'light'
      : 'dark';
  }
}
