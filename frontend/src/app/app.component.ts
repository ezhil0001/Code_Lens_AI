import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet } from '@angular/router';
import { SessionService } from './core/services/session.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, RouterOutlet],
  template: `<router-outlet></router-outlet>`,
  styles: [`
    :host {
      display: block;
      width: 100%;
      height: 100vh;
      margin: 0;
      padding: 0;
    }
  `]
})
export class AppComponent implements OnInit {
  title = 'CodeLens AI';

  constructor(private sessionService: SessionService) {}

  ngOnInit(): void {
    // SessionService initializes the session on construction, but injecting it
    // here also ensures it is eagerly created at app startup rather than lazily
    // on the first route that needs it — important for session recovery on page refresh.
  }
}
