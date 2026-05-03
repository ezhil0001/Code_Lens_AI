import { Component, OnInit } from '@angular/core';
import { ChatComponent } from './components/chat.component';
import { SessionService } from './core/services/session.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [ChatComponent],
  template: `<app-chat></app-chat>`,
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

  constructor(private sessionService: SessionService) {
    console.log('🚀 AppComponent initialized');
  }

  ngOnInit(): void {
    console.log('📌 AppComponent ngOnInit - initializing session');
    // SessionService is injected and automatically initializes session
    // This happens on app startup to recover chat history on refresh
  }
}
