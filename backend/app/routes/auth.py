"""
Authentication Routes - Login, register, token refresh endpoints

Security Features:
- JWT Revocation: Checks token blacklist before processing (Feature #1)
- Rate Limiting: Limits login attempts to prevent brute-force (Feature #2)
- Audit Logging: Tracks all authentication events
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from typing import Optional
import logging

from app.core.config import get_db
from app.auth.service import AuthenticationService
from app.services.user_service import UserService
from app.schemas.user import (
    LoginRequest,
    TokenResponse,
    TokenRefreshRequest,
    UserCreate,
    UserResponse,
    UserWithToken,
    ChangePasswordRequest,
)
from app.auth.jwt import decode_token_payload, extract_jti_from_payload
from app.auth.token_blacklist import get_token_blacklist_manager
from app.core.config import get_settings
from app.middleware.rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])


# ==================== Helper Functions ====================

def get_client_ip(request: Request) -> str:
    """Extract client IP from request"""
    if request.client:
        return request.client.host
    return "unknown"


# ==================== Login Endpoint ====================

@router.post("/login", response_model=dict)
async def login(
    credentials: LoginRequest,
    force_login: bool = False,
    request: Request = None,
    db: Session = Depends(get_db)
):
    """
    Login with email and password (Alhena Pattern)
    
    Security Features:
    - Rate Limiting: Max 5 login attempts per minute per IP (brute-force protection)
    - JWT with JTI: Each token gets unique ID for revocation tracking
    - Multi-device support: forceLogin flag controls session replacement
    
    Supports multi-device login prevention via forceLogin flag.
    If user is already logged in from another device:
    - forceLogin=False: Returns error
    - forceLogin=True: Invalidates old session, creates new one
    
    Query Parameters:
        force_login: Force login even if already logged in (default: False)
    
    Returns:
        {
            "access_token": "...",
            "refresh_token": "...",
            "token_type": "bearer",
            "user": {...},
            "success": True
        }
        
    Raises:
        HTTPException: If credentials invalid, account inactive, or rate limited
    """
    ip_address = get_client_ip(request) if request else "unknown"
    
    # Security: Apply rate limiting (Feature #2: Brute-force Protection)
    try:
        settings = get_settings()
        if settings.rate_limit_enabled:
            rate_limiter = get_rate_limiter(settings.redis_url)
            allowed, stats = rate_limiter.is_allowed(
                key=f"login:{ip_address}",
                limit=settings.login_rate_limit,  # Default: 5
                window_seconds=settings.login_rate_limit_window  # Default: 60
            )
            
            if not allowed:
                logger.warning(
                    f"Login rate limit exceeded for IP {ip_address}. "
                    f"Limit: {stats['limit']}/{stats['window_seconds']}s"
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Too many login attempts. Try again in {stats['reset']} seconds.",
                    headers={
                        "Retry-After": str(stats['reset']),
                        "X-RateLimit-Limit": str(stats['limit']),
                        "X-RateLimit-Remaining": str(stats['remaining']),
                    }
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Rate limiter error (proceeding with caution): {str(e)}")
        # Fail-open: if rate limiter fails, allow login to proceed
    
    try:
        # Login with Alhena pattern (includes loginId generation and JTI)
        result = AuthenticationService.login(
            db,
            email=credentials.email,
            password=credentials.password,
            force_login=force_login,
            ip_address=ip_address
        )
        
        return result
        
    except ValueError as e:
        # Log failed login
        AuthenticationService.log_audit(
            db,
            user_id=None,
            action="LOGIN_FAILED",
            resource="USER",
            ip_address=ip_address
        )
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


# ==================== Register Endpoint ====================

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Register new user
    
    Args:
        user_data: User email, username, and password
        request: HTTP request (for IP logging)
        db: Database session
        
    Returns:
        Created user info
        
    Raises:
        HTTPException: If user already exists
    """
    ip_address = get_client_ip(request)
    
    # Check if user exists
    existing_user = UserService.get_user_by_email(db, user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    existing_username = UserService.get_user_by_username(db, user_data.username)
    if existing_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Create user
    user = UserService.create_user(
        db,
        email=user_data.email,
        username=user_data.username,
        password=user_data.password,
        full_name=user_data.full_name
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create user"
        )
    
    # Assign default 'user' role
    user_role = db.query(Role).filter(Role.name == 'user').first()
    if user_role:
        user.roles.append(user_role)
        db.commit()
    
    # Log registration
    AuthenticationService.log_audit(
        db,
        user_id=user.id,
        action="REGISTER",
        resource="USER",
        ip_address=ip_address
    )
    
    return user


# ==================== Token Refresh Endpoint ====================

@router.post("/refresh", response_model=dict)
async def refresh_token(
    refresh_request: TokenRefreshRequest,
    request: Request = None,
    db: Session = Depends(get_db)
):
    """
    Refresh access token using refresh token (Alhena Pattern)
    
    Validates refresh token and generates new access token
    while keeping the same refresh token valid.
    
    Args:
        refresh_request: Refresh token
        request: HTTP request
        db: Database session
        
    Returns:
        {
            "access_token": "...",
            "refresh_token": "...",
            "token_type": "bearer"
        }
        
    Raises:
        HTTPException: If refresh token invalid or expired
    """
    ip_address = get_client_ip(request) if request else "unknown"
    
    try:
        result = AuthenticationService.refresh_access_token(
            db,
            refresh_request.refresh_token,
            ip_address=ip_address
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


# ==================== Logout Endpoint ====================

@router.post("/logout")
async def logout(
    request: Request = None,
    db: Session = Depends(get_db)
):
    """
    Logout user by clearing session data and revoking token (Alhena Pattern)
    
    Security Feature #1: JWT Revocation
    - Extracts access token from Authorization header
    - Adds token JTI to blacklist (revocation)
    - Prevents token replay after logout
    
    Args:
        request: HTTP request (contains Authorization header and access token)
        db: Database session
        
    Returns:
        {"message": "Logged out successfully"}
    """
    ip_address = get_client_ip(request) if request else "unknown"
    
    try:
        # Extract access token from Authorization header
        auth_header = request.headers.get("Authorization", "") if request else ""
        access_token = None
        
        if auth_header.startswith("Bearer "):
            access_token = auth_header[7:]  # Remove "Bearer " prefix
        
        # Extract user_id from token
        from app.auth.jwt import verify_access_token
        user_id = verify_access_token(access_token) if access_token else None
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid access token"
            )
        
        # Logout (Alhena pattern + JWT revocation)
        result = AuthenticationService.logout(
            db,
            user_id=user_id,
            access_token=access_token,  # Pass token for JTI extraction and revocation
            ip_address=ip_address
        )
        
        return result
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


# ==================== Authentication Dependency ====================

async def get_current_user(
    request: Request,
    db: Session = Depends(get_db)
) -> dict:
    """
    Dependency to get current authenticated user from JWT token (Alhena Pattern)
    
    Security Feature #1: JWT Revocation Check
    - Verifies token is not blacklisted (revoked)
    - Prevents use of logged-out tokens
    
    Can be used in protected endpoints:
    
    Example:
        ```python
        @router.get("/me")
        async def get_profile(current_user = Depends(get_current_user)):
            return current_user
        ```
    
    Args:
        request: HTTP request (contains Authorization header)
        db: Database session
        
    Returns:
        Current user object with org context
        
    Raises:
        HTTPException: If token missing, invalid, revoked, or expired
    """
    # Extract token from header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = auth_header[7:]  # Remove "Bearer " prefix
    
    # Security: Check if token is revoked (blacklisted)
    try:
        settings = get_settings()
        payload = decode_token_payload(token)
        
        if payload and settings.enable_jwt_blacklist:
            jti = extract_jti_from_payload(payload)
            if jti:
                token_blacklist = get_token_blacklist_manager(settings.redis_url)
                if token_blacklist.is_token_revoked(jti):
                    logger.warning(f"Attempt to use revoked token (JTI: {jti})")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Token has been revoked. Please log in again.",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Error checking token blacklist (proceeding): {str(e)}")
        # Fail-open: if blacklist check fails, continue with token validation
    
    # Validate token
    user = AuthenticationService.verify_token(token, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


# ==================== Get Current User Profile ====================

@router.get("/me", response_model=dict)
async def get_profile(
    current_user: dict = Depends(get_current_user)
):
    """
    Get current authenticated user profile with organization (Alhena Pattern)
    
    Requires valid JWT access token
    
    Args:
        current_user: Current authenticated user (injected by dependency)
        
    Returns:
        Current user profile with org info
    """
    return {
        "id": current_user.id,
        "email": current_user.email,
        "username": current_user.username,
        "full_name": current_user.full_name,
        "is_active": current_user.is_active,
        "is_verified": current_user.is_verified,
        "role": current_user.role.name if current_user.role else None,
        "org_id": current_user.org_id,
        "org_name": current_user.org.name if current_user.org else None,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
    }


# ==================== Get User Info ====================

@router.get("/user-info", response_model=dict)
async def get_user_info(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user info with organization and role details (Alhena Pattern)
    
    Args:
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        User info with org and role context
    """
    is_admin = current_user.role and current_user.role.name == "admin" if current_user.role else False
    
    result = AuthenticationService.get_user_info(
        db,
        current_user,
        is_admin=is_admin
    )
    
    return result


# ==================== Switch Organization ====================

@router.post("/switch-org/{org_id}", response_model=dict)
async def switch_organization(
    org_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Switch user to different organization (Superadmin Only - Alhena Pattern)
    
    Allows admin to switch between organizations they have access to
    
    Args:
        org_id: Target organization ID
        current_user: Current admin user
        db: Database session
        
    Returns:
        New access token for organization
        
    Raises:
        HTTPException: If user not admin or org not found
    """
    try:
        result = AuthenticationService.switch_org(
            db,
            target_org_id=org_id,
            user=current_user,
            login_id=current_user.login_id
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )


# ==================== Switch User (Admin Impersonation) ====================

@router.post("/switch-user/{user_id}", response_model=dict)
async def switch_user(
    user_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Switch to another user (Admin Only - Alhena Pattern)
    
    Allows admin to impersonate another user for support/debugging
    
    Args:
        user_id: Target user ID
        current_user: Current admin user
        db: Database session
        
    Returns:
        New access token as target user
        
    Raises:
        HTTPException: If user not admin or target user not found
    """
    try:
        result = AuthenticationService.switch_user(
            db,
            admin_user=current_user,
            target_user_id=user_id,
            login_id=current_user.login_id
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )


# Import models at end to avoid circular imports
from app.models.database import Role, Organization
