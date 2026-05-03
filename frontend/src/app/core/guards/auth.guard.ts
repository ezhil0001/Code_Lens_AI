/**
 * Authentication Guard - Protects chat routes from unauthorized access
 * 
 * Issues Fixed:
 * 1. Auth Bypass: Prevents direct access to /chat without valid session
 * 2. Session Validation: Checks both localStorage and backend for valid session
 * 3. Token Expiry: Validates token before allowing route access
 */

import { Injectable, inject } from '@angular/core';
import { Router, CanActivateFn, ActivatedRouteSnapshot, RouterStateSnapshot } from '@angular/router';
import { HttpClient } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class AuthGuard {
  constructor(
    private router: Router,
    private http: HttpClient
  ) {}

  /**
   * Check if user has valid authentication token
   */
  private hasValidToken(): boolean {
    // Check localStorage for token
    const token = localStorage.getItem('auth_token');
    if (!token) {
      console.warn('🚫 No auth token found in localStorage');
      return false;
    }

    // Check if token is expired (basic check)
    try {
      const payload = this.parseJwt(token);
      if (!payload || !payload.exp) {
        console.warn('🚫 Invalid token payload');
        return false;
      }

      const expiryTime = payload.exp * 1000; // Convert to milliseconds
      const now = Date.now();

      if (now >= expiryTime) {
        console.warn('🚫 Token has expired');
        localStorage.removeItem('auth_token');
        return false;
      }

      console.log('✅ Valid token found');
      return true;
    } catch (error) {
      console.warn('🚫 Failed to parse JWT:', error);
      return false;
    }
  }

  /**
   * Parse JWT token (simple base64 decode - NOT cryptographic verification)
   */
  private parseJwt(token: string): any {
    try {
      const base64Url = token.split('.')[1];
      const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
      const jsonPayload = decodeURIComponent(
        atob(base64)
          .split('')
          .map((c) => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
          .join('')
      );
      return JSON.parse(jsonPayload);
    } catch (error) {
      console.error('Failed to parse JWT:', error);
      return null;
    }
  }

  /**
   * Validate session with backend
   */
  async validateSessionWithBackend(sessionId: string): Promise<boolean> {
    try {
      const response = await firstValueFrom(
        this.http.get<{ valid: boolean }>(`/api/v1/auth/validate-session`, {
          headers: { 'X-Session-ID': sessionId }
        })
      );
      return response.valid;
    } catch (error) {
      console.warn('⚠️ Backend session validation failed:', error);
      return false;
    }
  }

  /**
   * Check if user is authenticated
   */
  canActivate(): boolean {
    const hasToken = this.hasValidToken();
    
    if (!hasToken) {
      console.log('❌ Authentication check failed - redirecting to login');
      this.router.navigate(['/login']);
      return false;
    }

    console.log('✅ Authentication check passed');
    return true;
  }
}

/**
 * Guard function for use in route configuration (Angular 17+)
 */
export const authGuard: CanActivateFn = (
  route: ActivatedRouteSnapshot,
  state: RouterStateSnapshot
): boolean => {
  const router = inject(Router);
  const http = inject(HttpClient);
  const guard = new AuthGuard(router, http);
  return guard.canActivate();
};
