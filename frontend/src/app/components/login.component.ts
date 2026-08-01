/**
 * Login Component - User Authentication
 * 
 * Features:
 * - Simple email/password login form
 * - Session creation on successful login
 * - Redirect to chat on success
 * - Error handling
 */

import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { HttpClient } from '@angular/common/http';
import { SessionService } from '../core/services/session.service';
import { environment } from '../../environments/environment';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.scss'],
})
export class LoginComponent implements OnInit {
  loginForm!: FormGroup;
  isLoading = false;
  errorMessage = '';

  constructor(
    private formBuilder: FormBuilder,
    private router: Router,
    private http: HttpClient,
    private sessionService: SessionService
  ) {}

  ngOnInit(): void {
    console.log('🔐 [LoginComponent] Initializing login form');
    this.setupForm();
  }

  /**
   * Setup login form with email and password fields
   */
  private setupForm(): void {
    this.loginForm = this.formBuilder.group({
      email: ['', [Validators.required, Validators.email]],
      password: ['', [Validators.required, Validators.minLength(6)]],
    });
  }

  /**
   * Handle login form submission
   */
  onLogin(): void {
    if (this.loginForm.invalid) {
      this.errorMessage = 'Please enter valid email and password';
      return;
    }

    this.isLoading = true;
    this.errorMessage = '';

    const { email, password } = this.loginForm.value;

    console.log(`🔐 [LoginComponent] Attempting login for: ${email}`);

    // Call backend login endpoint
    this.http.post<{ token: string; user: any }>('http://localhost:8000/api/v1/auth/login', {
      email,
      password,
    }).subscribe({
      next: (response) => {
        console.log('✅ Login successful');
        
        // Store token in localStorage
        localStorage.setItem('auth_token', response.token);
        
        // Create session
        this.sessionService.createNewSession();
        
        // Redirect to chat
        this.router.navigate(['/chat']);
      },
      error: (error) => {
        console.error('❌ Login failed:', error);
        this.isLoading = false;
        
        if (error.status === 401) {
          this.errorMessage = 'Invalid email or password';
        } else if (error.status === 0 && !environment.production) {
          // C-5: development-only offline login. NEVER reachable in
          // production builds — a backend outage must not grant a session.
          console.warn('⚠️ Backend unreachable, creating development session');
          this.createDevelopmentSession();
        } else if (error.status === 0) {
          this.errorMessage = 'Cannot reach the server. Please try again later.';
        } else {
          this.errorMessage = 'Login failed. Please try again.';
        }
      },
    });
  }

  /**
   * Create development session (for testing without backend).
   * C-5: guarded — throws in production builds.
   */
  private createDevelopmentSession(): void {
    if (environment.production) {
      throw new Error('Development session is not available in production builds');
    }
    const email = this.loginForm.get('email')?.value || 'dev@codelens.local';
    
    // Create development token
    const now = Math.floor(Date.now() / 1000);
    const exp = now + (24 * 60 * 60);
    
    const header = btoa(JSON.stringify({ alg: 'HS256', typ: 'JWT' }));
    const payload = btoa(JSON.stringify({
      sub: email,
      exp: exp,
      iat: now,
      email: email,
    }));
    const signature = btoa('dev-signature');
    const devToken = `${header}.${payload}.${signature}`;
    
    localStorage.setItem('auth_token', devToken);
    this.sessionService.createNewSession();
    
    console.log('✅ Development session created');
    this.router.navigate(['/chat']);
  }

  /**
   * Getter for form controls (for template)
   */
  get f() {
    return this.loginForm.controls;
  }
}
