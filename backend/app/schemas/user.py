"""
Pydantic Schemas for API request/response validation
"""

from pydantic import BaseModel, EmailStr, Field, field_validator, validator
from typing import List, Optional
from datetime import datetime
import re

# Minimal email pattern — accepts any user@domain format including .local,
# .internal, and other special-use TLDs that email-validator 2.x rejects.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ==================== User Schemas ====================

class UserBase(BaseModel):
    """Base user schema"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=255)
    full_name: Optional[str] = None


class UserCreate(UserBase):
    """User creation request"""
    password: str = Field(..., min_length=8)

    @validator('password')
    def password_strength(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain uppercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain digit')
        return v


class UserUpdate(BaseModel):
    """User update request"""
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None


class UserResponse(UserBase):
    """User response (never expose password)"""
    id: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    roles: List['RoleResponse'] = []

    class Config:
        from_attributes = True


class UserWithToken(UserResponse):
    """User response with authentication token"""
    access_token: str
    token_type: str = "bearer"


# ==================== Role Schemas ====================

class RoleBase(BaseModel):
    """Base role schema"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None


class RoleCreate(RoleBase):
    """Role creation request"""
    pass


class RoleResponse(RoleBase):
    """Role response"""
    id: str
    created_at: datetime
    permissions: List['PermissionResponse'] = []

    class Config:
        from_attributes = True


# ==================== Permission Schemas ====================

class PermissionBase(BaseModel):
    """Base permission schema"""
    name: str
    resource: str
    action: str
    description: Optional[str] = None


class PermissionCreate(PermissionBase):
    """Permission creation request"""
    pass


class PermissionResponse(PermissionBase):
    """Permission response"""
    id: str
    created_at: datetime

    class Config:
        from_attributes = True


# ==================== API Key Schemas ====================

class APIKeyCreate(BaseModel):
    """API key creation request"""
    name: Optional[str] = None


class APIKeyResponse(BaseModel):
    """API key response"""
    id: str
    name: Optional[str]
    key: str  # Only shown once on creation
    is_active: bool
    created_at: datetime
    expires_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class APIKeyListResponse(BaseModel):
    """API key list response (without exposing key)"""
    id: str
    name: Optional[str]
    is_active: bool
    last_used_at: Optional[datetime]
    created_at: datetime
    expires_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ==================== Authentication Schemas ====================

class LoginRequest(BaseModel):
    """Login request.

    Uses a plain str for ``email`` instead of Pydantic's ``EmailStr`` so that
    non-public TLDs (.local, .internal, .test, corporate intranets) are
    accepted.  email-validator ≥ 2.1 rejects those domains at parse time,
    causing a 422 before the route function is ever reached.
    """
    email: str = Field(..., description="User email address")
    password: str

    @field_validator("email")
    @classmethod
    def _email_format(cls, v: str) -> str:
        v = v.strip().lower()
        if not _EMAIL_RE.match(v):
            raise ValueError("Invalid email format")
        return v


class TokenResponse(BaseModel):
    """JWT token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenRefreshRequest(BaseModel):
    """Refresh token request"""
    refresh_token: str


class ChangePasswordRequest(BaseModel):
    """Change password request"""
    current_password: str
    new_password: str = Field(..., min_length=8)


# ==================== Audit Log Schemas ====================

class AuditLogResponse(BaseModel):
    """Audit log response"""
    id: str
    user_id: Optional[str]
    action: str
    resource: str
    resource_id: Optional[str]
    details: Optional[str]
    ip_address: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


# ==================== Error Schemas ====================

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: Optional[str] = None
    status_code: int


# Update forward references
UserResponse.model_rebuild()
RoleResponse.model_rebuild()
