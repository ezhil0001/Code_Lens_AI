import { Routes } from '@angular/router';
import { LoginComponent } from './components/login.component';
import { authGuard } from './core/guards/auth.guard';
import { ChatComponent } from './features/chat/components/chat.component';

export const routes: Routes = [
  {
    path: '',
    redirectTo: '/login',  // ← Changed: Default route goes to login, not chat
    pathMatch: 'full'
  },
  {
    path: 'login',
    component: LoginComponent  // ✅ Login route - no guard
  },
  {
    path: 'chat',
    component: ChatComponent,
    canActivate: [authGuard]  // ✅ Protected route - requires authentication
  },
  {
    path: '**',
    redirectTo: '/login'  // ✅ Redirect unknown routes to login
  }
];
