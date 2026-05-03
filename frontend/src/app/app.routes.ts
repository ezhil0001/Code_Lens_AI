import { Routes } from '@angular/router';
import { ChatComponent } from './components/chat.component';
import { authGuard } from './core/guards/auth.guard';

export const routes: Routes = [
  {
    path: '',
    redirectTo: '/chat',
    pathMatch: 'full'
  },
  {
    path: 'chat',
    component: ChatComponent,
    canActivate: [authGuard]  // ✅ Protected route - requires authentication
  },
  {
    path: '**',
    redirectTo: '/chat'
  }
];
