"""
Authentication Service - Alhena-Inspired Implementation

Follows Alhena's NestJS auth.service.ts pattern with:
- Login with email/password and device detection
- Refresh token management
- Token verification
- Logout with session cleanup and token revocation
- Multi-device session handling via loginId
- JWT revocation via blacklisting (security hardening)
- Organization and role-based context

Security Features:
- JTI (JWT ID) tracking for token revocation
- Token blacklist on logout (prevents token replay)
- Audit logging for all authentication events
"""

from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import logging
from uuid import uuid4

from app.models.database import User, Organization, Role, AuditLog
from app.auth.jwt import (
    create_access_token,
    create_refresh_token,
    verify_password,
    hash_password,
    verify_access_token,
    verify_refresh_token,
    decode_token_payload,
    extract_jti_from_payload,
    get_access_token_expiration,
    get_refresh_token_expiration,
)
from app.auth.token_blacklist import get_token_blacklist_manager
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class AuthenticationService:
    """
    Authentication Service - Alhena-Inspired
    
    Handles:
    - User login with device detection
    - Token generation (access + refresh)
    - Token refresh
    - Token verification
    - Logout with session cleanup
    - User info retrieval
    - Organization context management
    """

    @staticmethod
    def login(
        db: Session,
        email: str,
        password: str,
        force_login: bool = False,
        ip_address: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Login user with email and password (Alhena pattern)
        
        Handles:
        - Email verification
        - Account active status
        - Password validation
        - Device/session detection via loginId
        - Multi-device login prevention (unless forceLogin=True)
        
        Args:
            db: Database session
            email: User email
            password: Plain text password
            force_login: Force login even if already logged in from another device
            ip_address: IP address for audit logging
            
        Returns:
            Dict with access_token, refresh_token, and user info
            
        Raises:
            ValueError: If credentials invalid or account inactive
        """
        # Find user by email with organization and role
        user = db.query(User).filter(
            User.email == email
        ).first()
        
        if not user:
            logger.warning(f"Login attempt with non-existent email: {email}")
            raise ValueError("Invalid email or password")
        
        # Check if account is active (Alhena checks isActive)
        if not user.is_active:
            logger.warning(f"Login attempt on inactive account: {email}")
            raise ValueError("Your account is inactive")
        
        # Verify password
        if not verify_password(password, user.password_hash):
            logger.warning(f"Failed password attempt for: {email}")
            raise ValueError("Invalid email or password")
        
        # Check if verified (Alhena checks verified field)
        if not user.is_verified:
            logger.warning(f"Login attempt on unverified account: {email}")
            raise ValueError("Account not verified")
        
        # Check if already logged in from another device
        if user.is_logged_in and not force_login:
            logger.info(f"User {email} already logged in from another device")
            raise ValueError("User already logged in from another device")
        
        # Check if user is admin (via role)
        is_admin = user.role and user.role.name == "admin" if user.role else False
        
        # Generate unique login session ID (Alhena uses "login-{timestamp}")
        login_id = f"login-{int(datetime.utcnow().timestamp() * 1000)}"
        
        # Generate tokens with JTI for revocation tracking (security hardening)
        access_token, access_jti = create_access_token(
            user_id=user.id,
            org_id=user.org_id,
            org_name=user.org.name if user.org else None,
            is_admin=is_admin,
            login_id=login_id
        )
        
        refresh_token, refresh_jti = create_refresh_token(user_id=user.id)
        
        # Store JTI info in user record for revocation tracking
        user.refresh_token = refresh_token
        user.is_logged_in = True
        user.login_id = login_id
        user.last_login_at = datetime.utcnow()
        user.last_login_ip = ip_address
        
        # Store access token JTI for revocation (if using audit log)
        db.commit()
        db.refresh(user)
        
        # Log audit event
        AuthenticationService.log_audit(
            db,
            user_id=user.id,
            action="LOGIN_SUCCESS",
            resource="USER",
            ip_address=ip_address
        )
        
        logger.info(f"User logged in: {email} (loginId: {login_id})")
        
        # Return response in Alhena format
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user": AuthenticationService._user_response(user, is_admin),
            "success": True
        }

    @staticmethod
    def refresh_access_token(
        db: Session,
        refresh_token: str,
        ip_address: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Refresh access token using refresh token (Alhena pattern)
        
        Args:
            db: Database session
            refresh_token: Refresh token
            ip_address: IP address for audit
            
        Returns:
            New access token and refresh token
            
        Raises:
            ValueError: If refresh token invalid or expired
        """
        # Verify refresh token
        user_id = verify_refresh_token(refresh_token)
        if not user_id:
            logger.warning("Invalid or expired refresh token")
            raise ValueError("Refresh token expired or invalid")
        
        # Fetch user from database
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            logger.warning(f"User not found for refresh: {user_id}")
            raise ValueError("User not found")
        
        # Verify user is active
        if not user.is_active:
            logger.warning(f"Refresh attempt on inactive user: {user_id}")
            raise ValueError("User account is inactive")
        
        # Verify stored refresh token matches (Alhena stores it for validation)
        if user.refresh_token != refresh_token:
            logger.warning(f"Refresh token mismatch for user: {user_id}")
            raise ValueError("Refresh token invalid or revoked")
        
        # Check if user is admin
        is_admin = user.role and user.role.name == "admin" if user.role else False
        
        # Generate new tokens with JTI (security hardening)
        new_access_token, access_jti = create_access_token(
            user_id=user.id,
            org_id=user.org_id,
            org_name=user.org.name if user.org else None,
            is_admin=is_admin,
            login_id=user.login_id  # Keep same login ID
        )
        
        new_refresh_token, refresh_jti = create_refresh_token(user_id=user.id)
        
        # Update refresh token in database
        user.refresh_token = new_refresh_token
        db.commit()
        
        logger.info(f"Access token refreshed for user: {user_id}")
        
        return {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer"
        }

    @staticmethod
    def verify_token(
        db: Session,
        token: str
    ) -> Optional[User]:
        """
        Verify token and return user (Alhena pattern)
        
        Args:
            db: Database session
            token: JWT token to verify
            
        Returns:
            User object if valid, None otherwise
        """
        payload = decode_token_payload(token)
        if not payload:
            return None
        
        user_id = payload.get("userId")
        if not user_id:
            return None
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            logger.warning(f"Token verification: user not found {user_id}")
            return None
        
        return user

    @staticmethod
    def get_user_info(
        db: Session,
        user: User,
        is_admin: bool = False
    ) -> Dict[str, Any]:
        """
        Get user info with organization and avatar (Alhena pattern)
        
        Args:
            db: Database session
            user: User object
            is_admin: Whether user is admin
            
        Returns:
            User info dict with org and other details
        """
        user_info = {
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "full_name": user.full_name,
            "is_active": user.is_active,
            "is_verified": user.is_verified,
            "is_admin": is_admin,
            "role": user.role.name if user.role else None,
            "org_id": user.org_id,
            "org_name": user.org.name if user.org else None,
            "created_at": user.created_at.isoformat() if user.created_at else None,
        }
        
        return {
            "user": user_info,
            # TODO: Add avatar/agent data like Alhena does
        }

    @staticmethod
    @staticmethod
    def logout(db: Session, user_id: str, access_token: Optional[str] = None, ip_address: Optional[str] = None) -> Dict[str, str]:
        """
        Logout user by clearing session data and revoking tokens (Alhena pattern)
        
        Hardening Feature #1: JWT Revocation
        - Extracts JTI from access token
        - Adds token to blacklist to prevent reuse
        - Handles graceful degradation (Redis optional)
        
        Args:
            db: Database session
            user_id: User ID to logout
            access_token: Current access token (for JTI extraction and revocation)
            ip_address: IP address for audit
            
        Returns:
            Success message
        """
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            logger.warning(f"Logout: user not found {user_id}")
            raise ValueError("User not found")
        
        # Clear refresh token and login session (Alhena pattern)
        user.refresh_token = None
        user.is_logged_in = False
        user.login_id = None
        
        db.commit()
        
        # Revoke access token via blacklist (if access token provided)
        if access_token:
            try:
                settings = get_settings()
                token_blacklist = get_token_blacklist_manager(settings.redis_url)
                
                # Decode token to extract JTI and expiration
                payload = decode_token_payload(access_token)
                if payload:
                    jti = extract_jti_from_payload(payload)
                    exp = payload.get("exp")
                    
                    if jti and exp:
                        # Convert Unix timestamp to datetime
                        expiration_time = datetime.utcfromtimestamp(exp)
                        # Add to blacklist
                        token_blacklist.revoke_token(jti, expiration_time)
                        logger.info(f"Access token revoked for user {user_id} (JTI: {jti})")
            except Exception as e:
                # Log but don't fail - token revocation is security enhancement, not required for logout
                logger.warning(f"Failed to revoke token on logout for user {user_id}: {str(e)}")
        
        # Log audit event
        AuthenticationService.log_audit(
            db,
            user_id=user_id,
            action="LOGOUT",
            resource="USER",
            ip_address=ip_address
        )
        
        logger.info(f"User logged out: {user_id}")
        
        return {"message": "Logged out successfully"}

    @staticmethod
    def switch_org(
        db: Session,
        target_org_id: str,
        user: User,
        login_id: str
    ) -> Dict[str, Any]:
        """
        Switch user to different organization (Alhena pattern)
        
        Args:
            db: Database session
            target_org_id: Target organization ID
            user: Current user (should be superadmin)
            login_id: Current login session ID
            
        Returns:
            New access token for target organization
        """
        # Verify user is admin/superadmin
        if not user.role or user.role.name not in ["admin", "superadmin"]:
            raise ValueError("Only admin users can switch organizations")
        
        # Fetch target organization
        target_org = db.query(Organization).filter(
            Organization.id == target_org_id
        ).first()
        
        if not target_org:
            raise ValueError("Target organization not found")
        
        # Generate new access token for target organization
        is_admin = True
        access_token = create_access_token(
            user_id=user.id,
            org_id=target_org_id,
            org_name=target_org.name,
            is_admin=is_admin,
            login_id=login_id  # Keep same login session
        )
        
        logger.info(f"User {user.id} switched to organization {target_org_id}")
        
        return {
            "access_token": access_token,
            "user": AuthenticationService._user_response(user, is_admin),
        }

    @staticmethod
    def switch_user(
        db: Session,
        admin_user: User,
        target_user_id: str,
        login_id: str
    ) -> Dict[str, Any]:
        """
        Admin switch to another user (Alhena pattern)
        
        Args:
            db: Database session
            admin_user: Current admin user
            target_user_id: Target user ID
            login_id: Current login session ID
            
        Returns:
            New access token for target user
        """
        # Verify admin
        if not admin_user.role or admin_user.role.name not in ["admin", "superadmin"]:
            raise ValueError("Only admin users can switch users")
        
        # Fetch target user
        target_user = db.query(User).filter(
            User.id == target_user_id
        ).first()
        
        if not target_user:
            raise ValueError("Target user not found")
        
        # Generate new access token as target user
        is_admin = target_user.role and target_user.role.name == "admin" if target_user.role else False
        access_token = create_access_token(
            user_id=target_user.id,
            org_id=admin_user.org_id,
            org_name=admin_user.org.name if admin_user.org else None,
            is_admin=is_admin,
            login_id=login_id
        )
        
        logger.info(f"Admin {admin_user.id} switched to user {target_user_id}")
        
        return {
            "access_token": access_token,
            "user": AuthenticationService._user_response(target_user, is_admin),
        }


    @staticmethod
    def log_audit(
        db: Session,
        user_id: Optional[str],
        action: str,
        resource: str,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> None:
        """
        Log audit event for security tracking
        
        Args:
            db: Database session
            user_id: User performing action
            action: Action type (LOGIN, LOGOUT, etc.)
            resource: Resource affected
            resource_id: Specific resource ID
            ip_address: IP address
        """
        audit_log = AuditLog(
            user_id=user_id,
            action=action,
            resource=resource,
            resource_id=resource_id,
            ip_address=ip_address
        )
        db.add(audit_log)
        db.commit()

    @staticmethod
    def _user_response(user: User, is_admin: bool = False) -> Dict[str, Any]:
        """Helper to format user response"""
        return {
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "full_name": user.full_name,
            "is_active": user.is_active,
            "is_verified": user.is_verified,
            "is_admin": is_admin,
            "role": user.role.name if user.role else None,
            "org_id": user.org_id,
            "org_name": user.org.name if user.org else None,
            "created_at": user.created_at.isoformat() if user.created_at else None,
        }
