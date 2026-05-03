"""
Database Models using SQLAlchemy
Equivalent to Prisma schema.prisma
"""

from sqlalchemy import Column, String, Integer, DateTime, Boolean, ForeignKey, Table, Text, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()

# Many-to-Many Association Table for User-Role relationships
user_role_association = Table(
    'user_role_association',
    Base.metadata,
    Column('user_id', String(36), ForeignKey('users.id'), primary_key=True),
    Column('role_id', String(36), ForeignKey('roles.id'), primary_key=True)
)


class RoleEnum(str, enum.Enum):
    """Enum for standard roles"""
    SUPERADMIN = "superadmin"
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


class Organization(Base):
    """
    Organization Model - Alhena Pattern
    Represents a workspace/organization/tenant
    """
    __tablename__ = "organizations"

    id = Column(String(36), primary_key=True, index=True)
    name = Column(String(255), unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    slug = Column(String(255), unique=True, index=True, nullable=True)
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    users = relationship("User", back_populates="org")

    def __repr__(self):
        return f"<Organization(id={self.id}, name={self.name})>"


class User(Base):
    """
    User Model - Alhena-Inspired
    Equivalent to Prisma User model with multi-device prevention
    
    Key Fields:
    - loginId: Unique session identifier for multi-device prevention
    - isLoggedIn: Tracks current login state
    - refreshToken: Stored refresh token for validation
    - role_id: Single primary role (with additionalRoles via association)
    - org_id: Organization/workspace the user belongs to
    """
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    
    # Session management (Alhena pattern)
    is_logged_in = Column(Boolean, default=False, index=True)
    login_id = Column(String(100), nullable=True, index=True)  # Format: "login-{timestamp}"
    refresh_token = Column(String(500), nullable=True)  # Stored for validation
    
    # Account status
    is_active = Column(Boolean, default=True, index=True)
    is_verified = Column(Boolean, default=False)
    
    # Login tracking
    last_login_at = Column(DateTime, nullable=True)
    last_login_ip = Column(String(45), nullable=True)
    
    # Organization (Alhena pattern)
    org_id = Column(String(36), ForeignKey('organizations.id'), nullable=True)
    
    # Primary role (can have additional roles via association)
    role_id = Column(String(36), ForeignKey('roles.id'), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    role = relationship("Role", foreign_keys=[role_id], lazy="selectin")
    org = relationship("Organization", foreign_keys=[org_id])
    roles = relationship(
        "Role",
        secondary=user_role_association,
        back_populates="users",
        lazy="selectin"
    )
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, username={self.username}, loginId={self.login_id})>"


class Role(Base):
    """
    Role Model
    Equivalent to Prisma Role model
    """
    __tablename__ = "roles"

    id = Column(String(36), primary_key=True, index=True)
    name = Column(String(255), unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    users = relationship(
        "User",
        secondary=user_role_association,
        back_populates="roles"
    )
    permissions = relationship(
        "Permission",
        secondary="role_permission_association",
        back_populates="roles"
    )

    def __repr__(self):
        return f"<Role(id={self.id}, name={self.name})>"


class Permission(Base):
    """
    Permission Model
    Equivalent to Prisma Permission model
    """
    __tablename__ = "permissions"

    id = Column(String(36), primary_key=True, index=True)
    name = Column(String(255), unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    resource = Column(String(255), nullable=False)  # e.g., "users", "roles"
    action = Column(String(255), nullable=False)    # e.g., "create", "read", "update", "delete"
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    roles = relationship(
        "Role",
        secondary="role_permission_association",
        back_populates="permissions"
    )

    def __repr__(self):
        return f"<Permission(id={self.id}, name={self.name})>"


# Many-to-Many Association Table for Role-Permission relationships
role_permission_association = Table(
    'role_permission_association',
    Base.metadata,
    Column('role_id', String(36), ForeignKey('roles.id'), primary_key=True),
    Column('permission_id', String(36), ForeignKey('permissions.id'), primary_key=True)
)


class APIKey(Base):
    """
    API Key Model
    For x-api-key authentication
    """
    __tablename__ = "api_keys"

    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False)
    key = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, index=True)
    last_used_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="api_keys")

    def __repr__(self):
        return f"<APIKey(id={self.id}, user_id={self.user_id}, name={self.name})>"


class RefreshToken(Base):
    """
    Refresh Token Model
    Stores JWT refresh tokens for session management
    """
    __tablename__ = "refresh_tokens"

    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False)
    token = Column(String(500), unique=True, index=True, nullable=False)
    is_revoked = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)

    # Relationships
    user = relationship("User", back_populates="tokens")

    def __repr__(self):
        return f"<RefreshToken(id={self.id}, user_id={self.user_id})>"


class AuditLog(Base):
    """
    Audit Log Model
    Tracks all important actions for security/compliance
    """
    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=True)
    action = Column(String(255), nullable=False)
    resource = Column(String(255), nullable=False)
    resource_id = Column(String(36), nullable=True)
    details = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    def __repr__(self):
        return f"<AuditLog(id={self.id}, action={self.action}, user_id={self.user_id})>"
