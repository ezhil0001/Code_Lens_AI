"""
User Service - User CRUD operations and management
"""

from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import uuid

from app.models.database import User, Role, RefreshToken
from app.auth.jwt import hash_password
from app.auth.service import AuthenticationService


class UserService:
    """
    User Service
    
    Handles:
    - User CRUD operations
    - User role management
    - User activation/deactivation
    - Superadmin initialization
    """

    @staticmethod
    def create_user(
        db: Session,
        email: str,
        username: str,
        password: str,
        full_name: Optional[str] = None
    ) -> Optional[User]:
        """
        Create a new user
        
        Args:
            db: Database session
            email: User email (must be unique)
            username: Username (must be unique)
            password: Plain text password (will be hashed)
            full_name: Optional full name
            
        Returns:
            Created User object or None if user already exists
        """
        try:
            # Check if user already exists
            existing_user = db.query(User).filter(
                (User.email == email) | (User.username == username)
            ).first()
            
            if existing_user:
                return None
            
            # Hash password
            password_hash = hash_password(password)
            
            # Create user
            user = User(
                id=str(uuid.uuid4()),
                email=email,
                username=username,
                password_hash=password_hash,
                full_name=full_name,
                is_active=True,
                is_verified=False
            )
            
            db.add(user)
            db.commit()
            db.refresh(user)
            
            return user
        
        except IntegrityError:
            db.rollback()
            return None

    @staticmethod
    def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
        """
        Get user by ID
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            User object or None if not found
        """
        return db.query(User).filter(User.id == user_id).first()

    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        """
        Get user by email
        
        Args:
            db: Database session
            email: User email
            
        Returns:
            User object or None if not found
        """
        return db.query(User).filter(User.email == email).first()

    @staticmethod
    def get_user_by_username(db: Session, username: str) -> Optional[User]:
        """
        Get user by username
        
        Args:
            db: Database session
            username: Username
            
        Returns:
            User object or None if not found
        """
        return db.query(User).filter(User.username == username).first()

    @staticmethod
    def list_users(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        is_active: Optional[bool] = None
    ) -> List[User]:
        """
        List users with pagination
        
        Args:
            db: Database session
            skip: Number of records to skip
            limit: Number of records to return
            is_active: Filter by active status (optional)
            
        Returns:
            List of User objects
        """
        query = db.query(User)
        
        if is_active is not None:
            query = query.filter(User.is_active == is_active)
        
        return query.offset(skip).limit(limit).all()

    @staticmethod
    def update_user(
        db: Session,
        user_id: str,
        **kwargs
    ) -> Optional[User]:
        """
        Update user fields
        
        Args:
            db: Database session
            user_id: User ID
            **kwargs: Fields to update (email, username, full_name, is_active, is_verified)
            
        Returns:
            Updated User object or None if not found
        """
        user = UserService.get_user_by_id(db, user_id)
        if not user:
            return None
        
        # Allowed fields to update
        allowed_fields = ['email', 'username', 'full_name', 'is_active', 'is_verified']
        
        for field, value in kwargs.items():
            if field in allowed_fields and value is not None:
                setattr(user, field, value)
        
        try:
            db.commit()
            db.refresh(user)
            return user
        except IntegrityError:
            db.rollback()
            return None

    @staticmethod
    def change_password(
        db: Session,
        user_id: str,
        old_password: str,
        new_password: str
    ) -> bool:
        """
        Change user password
        
        Args:
            db: Database session
            user_id: User ID
            old_password: Current password (plain text)
            new_password: New password (plain text)
            
        Returns:
            True if changed, False if old password invalid or user not found
        """
        user = UserService.get_user_by_id(db, user_id)
        if not user:
            return False
        
        # Verify old password
        if not AuthenticationService.authenticate_user(db, user.email, old_password):
            return False
        
        # Hash and update new password
        user.password_hash = hash_password(new_password)
        db.commit()
        
        return True

    @staticmethod
    def delete_user(db: Session, user_id: str) -> bool:
        """
        Delete user (soft delete - mark inactive)
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            True if deleted, False if not found
        """
        user = UserService.get_user_by_id(db, user_id)
        if not user:
            return False
        
        # Soft delete - mark as inactive
        user.is_active = False
        
        # Revoke all tokens
        AuthenticationService.revoke_all_user_tokens(db, user_id)
        
        db.commit()
        return True

    @staticmethod
    def add_role_to_user(
        db: Session,
        user_id: str,
        role_id: str
    ) -> Optional[User]:
        """
        Add role to user
        
        Args:
            db: Database session
            user_id: User ID
            role_id: Role ID
            
        Returns:
            Updated User object or None if user/role not found
        """
        user = UserService.get_user_by_id(db, user_id)
        if not user:
            return None
        
        role = db.query(Role).filter(Role.id == role_id).first()
        if not role:
            return None
        
        if role not in user.roles:
            user.roles.append(role)
            db.commit()
            db.refresh(user)
        
        return user

    @staticmethod
    def remove_role_from_user(
        db: Session,
        user_id: str,
        role_id: str
    ) -> Optional[User]:
        """
        Remove role from user
        
        Args:
            db: Database session
            user_id: User ID
            role_id: Role ID
            
        Returns:
            Updated User object or None if user/role not found
        """
        user = UserService.get_user_by_id(db, user_id)
        if not user:
            return None
        
        role = db.query(Role).filter(Role.id == role_id).first()
        if not role:
            return None
        
        if role in user.roles:
            user.roles.remove(role)
            db.commit()
            db.refresh(user)
        
        return user

    @staticmethod
    def verify_user(db: Session, user_id: str) -> Optional[User]:
        """
        Mark user as verified (for email verification)
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Updated User object or None if not found
        """
        return UserService.update_user(db, user_id, is_verified=True)

    @staticmethod
    def create_superadmin_user(
        db: Session,
        email: str,
        username: str,
        password: str
    ) -> Optional[User]:
        """
        Create superadmin user (called during app startup)
        
        Args:
            db: Database session
            email: Superadmin email
            username: Superadmin username
            password: Superadmin password
            
        Returns:
            Created User object or None if already exists
        """
        # Check if superadmin already exists
        existing_admin = db.query(User).join(User.roles).filter(
            Role.name == 'superadmin'
        ).first()
        
        if existing_admin:
            return None
        
        # Create superadmin user
        user = UserService.create_user(
            db,
            email=email,
            username=username,
            password=password,
            full_name="System Administrator"
        )
        
        if not user:
            return None
        
        # Get or create superadmin role
        superadmin_role = db.query(Role).filter(Role.name == 'superadmin').first()
        
        if superadmin_role:
            # Assign superadmin role
            user.roles.append(superadmin_role)
            user.is_verified = True
            db.commit()
            db.refresh(user)
        
        return user

    @staticmethod
    def get_user_count(db: Session) -> int:
        """
        Get total user count
        
        Args:
            db: Database session
            
        Returns:
            Total number of users
        """
        return db.query(User).count()

    @staticmethod
    def get_active_user_count(db: Session) -> int:
        """
        Get count of active users
        
        Args:
            db: Database session
            
        Returns:
            Number of active users
        """
        return db.query(User).filter(User.is_active == True).count()
