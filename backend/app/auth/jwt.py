"""
JWT Token generation and validation utilities

Inspired by Alhena NestJS implementation.
Handles token generation, verification, password hashing, and token revocation.

Features:
- JTI (JWT ID) for token revocation & blacklisting
- Standard Alhena payload structure (userId, orgId, loginId, etc.)
- Support for multi-device sessions via loginId
- Graceful token expiration calculations
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
import logging
import uuid

logger = logging.getLogger(__name__)

# ==================== Configuration ====================

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = int(os.getenv("ACCESS_TOKEN_EXPIRE_DAYS", "1"))  # 1 day like Alhena
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Password hashing configuration
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12
)


# ==================== Password Hashing ====================

def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.

    Bcrypt has a hard 72-byte input limit. We truncate the *byte* (not char)
    representation defensively so that:
      • Multi-byte unicode passwords don't silently overflow.
      • A misconfigured env var (e.g. a JWT-style token pasted as a password)
        produces a usable hash instead of crashing DB initialisation.
    Truncation matches the OpenBSD bcrypt convention used by every major
    bcrypt impl, so existing hashes remain verifiable.

    Args:
        password: Plain text password

    Returns:
        Hashed password
    """
    if password is None:
        raise ValueError("password must not be None")
    pw_bytes = password.encode("utf-8")
    if len(pw_bytes) > 72:
        # Decode-tolerant truncation: cut on a UTF-8 boundary so we never
        # produce an invalid string, then re-encode for bcrypt.
        password = pw_bytes[:72].decode("utf-8", errors="ignore")
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash. Applies the same 72-byte truncation
    as ``hash_password`` so passwords hashed via this module always verify.

    Args:
        plain_password: Plain text password to verify
        hashed_password: Hashed password from database

    Returns:
        True if password matches, False otherwise
    """
    if plain_password is None or hashed_password is None:
        return False
    pw_bytes = plain_password.encode("utf-8")
    if len(pw_bytes) > 72:
        plain_password = pw_bytes[:72].decode("utf-8", errors="ignore")
    return pwd_context.verify(plain_password, hashed_password)


# ==================== JTI (JWT ID) Generation ====================

def generate_jti() -> str:
    """
    Generate a unique JTI (JWT ID) for token revocation tracking
    
    JTI allows tokens to be individually revoked (blacklisted) without
    affecting other valid tokens from the same user.
    
    Returns:
        Unique JWT ID (UUID4 format)
    """
    return str(uuid.uuid4())


def extract_jti_from_payload(payload: Dict[str, Any]) -> Optional[str]:
    """
    Extract JTI from a decoded token payload
    
    Args:
        payload: Decoded JWT payload
        
    Returns:
        JTI if present, None otherwise
    """
    return payload.get("jti")


# ==================== JWT Token Generation ====================

def create_access_token(
    user_id: str,
    org_id: Optional[str] = None,
    org_name: Optional[str] = None,
    is_admin: bool = False,
    login_id: Optional[str] = None,
    expires_delta: Optional[timedelta] = None,
    jti: Optional[str] = None
) -> Tuple[str, str]:
    """
    Create a JWT access token (Alhena pattern) with JTI for revocation
    
    Args:
        user_id: User's unique identifier
        org_id: Organization ID
        org_name: Organization name
        is_admin: Whether user is admin
        login_id: Unique login session ID (e.g., "login-1699000000")
        expires_delta: Custom expiration time
        jti: JWT ID for revocation (auto-generated if not provided)
        
    Returns:
        Tuple of (token, jti) where:
        - token: Encoded JWT access token
        - jti: JWT ID for revocation tracking
        
    Example:
        >>> token, jti = create_access_token(
        ...     user_id='user123',
        ...     org_id='org456',
        ...     org_name='Acme Corp',
        ...     is_admin=True,
        ...     login_id='login-1699000000'
        ... )
        >>> # Later, revoke with: blacklist.revoke_token(jti, token_expiration)
    """
    # Generate JTI if not provided
    if jti is None:
        jti = generate_jti()
    
    to_encode = {
        "userId": user_id,
        "type": "access-token",  # Alhena uses "access-token"
        "jti": jti,  # JWT ID for revocation
    }
    
    # Add optional org information
    if org_id:
        to_encode["orgId"] = org_id
    if org_name:
        to_encode["orgName"] = org_name
    if login_id:
        to_encode["loginId"] = login_id
    
    to_encode["isAdmin"] = is_admin
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({"exp": expire})
    
    try:
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        logger.debug(f"Access token created for user {user_id} with JTI {jti}")
        return encoded_jwt, jti
    except Exception as e:
        logger.error(f"Failed to create access token: {str(e)}")
        raise


def create_refresh_token(
    user_id: str,
    expires_delta: Optional[timedelta] = None,
    jti: Optional[str] = None
) -> Tuple[str, str]:
    """
    Create a JWT refresh token (Alhena pattern) with JTI for revocation
    
    Args:
        user_id: User's unique identifier
        expires_delta: Custom expiration time
        jti: JWT ID for revocation (auto-generated if not provided)
        
    Returns:
        Tuple of (token, jti) where:
        - token: Encoded JWT refresh token
        - jti: JWT ID for revocation tracking
    """
    # Generate JTI if not provided
    if jti is None:
        jti = generate_jti()
    
    to_encode = {
        "userId": user_id,
        "type": "refresh-token",  # Alhena uses "refresh-token"
        "jti": jti,  # JWT ID for revocation
    }
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({"exp": expire})
    
    try:
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        logger.debug(f"Refresh token created for user {user_id} with JTI {jti}")
        return encoded_jwt, jti
    except Exception as e:
        logger.error(f"Failed to create refresh token: {str(e)}")
        raise


# ==================== JWT Token Validation ====================

def verify_token(
    token: str,
    token_type: str = "access-token"
) -> Optional[Dict[str, Any]]:
    """
    Verify and decode a JWT token (Alhena pattern)
    
    Args:
        token: JWT token to verify
        token_type: Expected token type ("access-token" or "refresh-token")
        
    Returns:
        Decoded payload if valid, None if invalid/expired
        
    Raises:
        JWTError: If token is malformed or tampered
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Verify token type (Alhena uses "access-token"/"refresh-token")
        if payload.get("type") != token_type:
            logger.warning(f"Token type mismatch: expected {token_type}, got {payload.get('type')}")
            return None
        
        user_id: str = payload.get("userId")
        if user_id is None:
            logger.warning("Token missing userId")
            return None
        
        logger.debug(f"Token verified for user {user_id}")
        return payload
    except JWTError as e:
        logger.warning(f"JWT verification failed: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error verifying token: {str(e)}")
        return None


def verify_access_token(token: str) -> Optional[str]:
    """
    Verify an access token and return user_id (Alhena pattern)
    
    Args:
        token: JWT access token
        
    Returns:
        User ID if valid, None otherwise
    """
    payload = verify_token(token, token_type="access-token")
    if payload is None:
        return None
    return payload.get("userId")


def verify_refresh_token(token: str) -> Optional[str]:
    """
    Verify a refresh token and return user_id (Alhena pattern)
    
    Args:
        token: JWT refresh token
        
    Returns:
        User ID if valid, None otherwise
    """
    payload = verify_token(token, token_type="refresh-token")
    if payload is None:
        return None
    return payload.get("userId")


def decode_token_payload(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode token payload without type checking
    Used for extracting claims from valid tokens
    
    Args:
        token: JWT token to decode
        
    Returns:
        Decoded payload if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        logger.warning("Failed to decode token payload")
        return None


# ==================== Token Expiration Calculations ====================

def get_access_token_expiration() -> timedelta:
    """Get access token expiration delta (1 day like Alhena)"""
    return timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)


def get_refresh_token_expiration() -> timedelta:
    """Get refresh token expiration delta (7 days)"""
    return timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
