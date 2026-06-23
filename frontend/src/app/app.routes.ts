import { Routes } from '@angular/router';
import { LoginComponent } from './components/login.component';
import { authGuard } from './core/guards/auth.guard';
import { ChatComponent } from './features/chat/components/chat.component';

export const routes: Routes = [
  {
    path: '',
    redirectTo: '/login',
    pathMatch: 'full'
  },
  {
    path: 'login',
    component: LoginComponent
  },
  {
    path: 'chat',
    component: ChatComponent,
    canActivate: [authGuard]
  },
  {
    path: '**',
    redirectTo: '/login'
  }
];
