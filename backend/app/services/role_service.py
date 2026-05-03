"""
Role Service - Role and Permission management
"""

from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import uuid

from app.models.database import Role, Permission, User


class RoleService:
    """
    Role Service
    
    Handles:
    - Role CRUD operations
    - Permission management
    - Default role creation
    """

    # ==================== Role CRUD ====================

    @staticmethod
    def create_role(
        db: Session,
        name: str,
        description: Optional[str] = None
    ) -> Optional[Role]:
        """
        Create a new role
        
        Args:
            db: Database session
            name: Role name (must be unique)
            description: Optional role description
            
        Returns:
            Created Role object or None if already exists
        """
        try:
            # Check if role exists
            existing_role = db.query(Role).filter(Role.name == name).first()
            if existing_role:
                return None
            
            role = Role(
                id=str(uuid.uuid4()),
                name=name,
                description=description
            )
            
            db.add(role)
            db.commit()
            db.refresh(role)
            
            return role
        except IntegrityError:
            db.rollback()
            return None

    @staticmethod
    def get_role_by_id(db: Session, role_id: str) -> Optional[Role]:
        """
        Get role by ID
        
        Args:
            db: Database session
            role_id: Role ID
            
        Returns:
            Role object or None if not found
        """
        return db.query(Role).filter(Role.id == role_id).first()

    @staticmethod
    def get_role_by_name(db: Session, name: str) -> Optional[Role]:
        """
        Get role by name
        
        Args:
            db: Database session
            name: Role name
            
        Returns:
            Role object or None if not found
        """
        return db.query(Role).filter(Role.name == name).first()

    @staticmethod
    def list_roles(db: Session, skip: int = 0, limit: int = 100) -> List[Role]:
        """
        List all roles with pagination
        
        Args:
            db: Database session
            skip: Number of records to skip
            limit: Number of records to return
            
        Returns:
            List of Role objects
        """
        return db.query(Role).offset(skip).limit(limit).all()

    @staticmethod
    def update_role(
        db: Session,
        role_id: str,
        **kwargs
    ) -> Optional[Role]:
        """
        Update role fields
        
        Args:
            db: Database session
            role_id: Role ID
            **kwargs: Fields to update (name, description)
            
        Returns:
            Updated Role object or None if not found
        """
        role = RoleService.get_role_by_id(db, role_id)
        if not role:
            return None
        
        allowed_fields = ['name', 'description']
        
        for field, value in kwargs.items():
            if field in allowed_fields and value is not None:
                setattr(role, field, value)
        
        try:
            db.commit()
            db.refresh(role)
            return role
        except IntegrityError:
            db.rollback()
            return None

    @staticmethod
    def delete_role(db: Session, role_id: str) -> bool:
        """
        Delete role (cannot delete if users have this role)
        
        Args:
            db: Database session
            role_id: Role ID
            
        Returns:
            True if deleted, False if not found or has users
        """
        role = RoleService.get_role_by_id(db, role_id)
        if not role:
            return False
        
        # Check if users have this role
        users_with_role = db.query(User).filter(User.roles.contains(role)).count()
        if users_with_role > 0:
            return False
        
        db.delete(role)
        db.commit()
        
        return True

    # ==================== Permission Management ====================

    @staticmethod
    def add_permission_to_role(
        db: Session,
        role_id: str,
        permission_id: str
    ) -> Optional[Role]:
        """
        Add permission to role
        
        Args:
            db: Database session
            role_id: Role ID
            permission_id: Permission ID
            
        Returns:
            Updated Role object or None if role/permission not found
        """
        role = RoleService.get_role_by_id(db, role_id)
        if not role:
            return None
        
        permission = db.query(Permission).filter(Permission.id == permission_id).first()
        if not permission:
            return None
        
        if permission not in role.permissions:
            role.permissions.append(permission)
            db.commit()
            db.refresh(role)
        
        return role

    @staticmethod
    def remove_permission_from_role(
        db: Session,
        role_id: str,
        permission_id: str
    ) -> Optional[Role]:
        """
        Remove permission from role
        
        Args:
            db: Database session
            role_id: Role ID
            permission_id: Permission ID
            
        Returns:
            Updated Role object or None if role/permission not found
        """
        role = RoleService.get_role_by_id(db, role_id)
        if not role:
            return None
        
        permission = db.query(Permission).filter(Permission.id == permission_id).first()
        if not permission:
            return None
        
        if permission in role.permissions:
            role.permissions.remove(permission)
            db.commit()
            db.refresh(role)
        
        return role

    # ==================== Permission CRUD ====================

    @staticmethod
    def create_permission(
        db: Session,
        name: str,
        resource: str,
        action: str,
        description: Optional[str] = None
    ) -> Optional[Permission]:
        """
        Create a new permission
        
        Args:
            db: Database session
            name: Permission name
            resource: Resource type (USER, ROLE, DOCUMENT, etc.)
            action: Action type (READ, CREATE, UPDATE, DELETE)
            description: Optional description
            
        Returns:
            Created Permission object or None if already exists
        """
        try:
            # Check if permission exists
            existing_perm = db.query(Permission).filter(
                (Permission.name == name)
            ).first()
            if existing_perm:
                return None
            
            permission = Permission(
                id=str(uuid.uuid4()),
                name=name,
                resource=resource,
                action=action,
                description=description
            )
            
            db.add(permission)
            db.commit()
            db.refresh(permission)
            
            return permission
        except IntegrityError:
            db.rollback()
            return None

    @staticmethod
    def get_permission_by_id(db: Session, permission_id: str) -> Optional[Permission]:
        """
        Get permission by ID
        
        Args:
            db: Database session
            permission_id: Permission ID
            
        Returns:
            Permission object or None if not found
        """
        return db.query(Permission).filter(Permission.id == permission_id).first()

    @staticmethod
    def get_permission_by_resource_action(
        db: Session,
        resource: str,
        action: str
    ) -> Optional[Permission]:
        """
        Get permission by resource and action
        
        Args:
            db: Database session
            resource: Resource type
            action: Action type
            
        Returns:
            Permission object or None if not found
        """
        return db.query(Permission).filter(
            Permission.resource == resource,
            Permission.action == action
        ).first()

    @staticmethod
    def list_permissions(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        resource: Optional[str] = None
    ) -> List[Permission]:
        """
        List permissions with optional filtering
        
        Args:
            db: Database session
            skip: Number of records to skip
            limit: Number of records to return
            resource: Filter by resource (optional)
            
        Returns:
            List of Permission objects
        """
        query = db.query(Permission)
        
        if resource:
            query = query.filter(Permission.resource == resource)
        
        return query.offset(skip).limit(limit).all()

    # ==================== Batch Operations ====================

    @staticmethod
    def create_default_roles(db: Session) -> tuple:
        """
        Create default roles for system
        
        Args:
            db: Database session
            
        Returns:
            Tuple of (superadmin_role, admin_role, user_role, viewer_role)
        """
        default_roles = {
            'superadmin': 'System Administrator - Full access',
            'admin': 'Administrator - Manage users and content',
            'user': 'User - Can create and edit content',
            'viewer': 'Viewer - Read-only access'
        }
        
        created_roles = {}
        
        for role_name, description in default_roles.items():
            existing = RoleService.get_role_by_name(db, role_name)
            if existing:
                created_roles[role_name] = existing
            else:
                role = RoleService.create_role(db, role_name, description)
                if role:
                    created_roles[role_name] = role
        
        return (
            created_roles.get('superadmin'),
            created_roles.get('admin'),
            created_roles.get('user'),
            created_roles.get('viewer')
        )

    @staticmethod
    def create_default_permissions(db: Session) -> None:
        """
        Create default permissions for system
        
        Args:
            db: Database session
        """
        resources = ['USER', 'ROLE', 'PERMISSION', 'DOCUMENT', 'API_KEY']
        actions = ['CREATE', 'READ', 'UPDATE', 'DELETE', 'LIST']
        
        for resource in resources:
            for action in actions:
                existing = RoleService.get_permission_by_resource_action(
                    db, resource, action
                )
                if not existing:
                    RoleService.create_permission(
                        db,
                        name=f"{resource}_{action}",
                        resource=resource,
                        action=action,
                        description=f"Permission to {action.lower()} {resource.lower()}"
                    )

    @staticmethod
    def assign_permissions_to_superadmin(db: Session) -> Optional[Role]:
        """
        Assign all permissions to superadmin role
        
        Args:
            db: Database session
            
        Returns:
            Updated superadmin Role or None if not found
        """
        superadmin = RoleService.get_role_by_name(db, 'superadmin')
        if not superadmin:
            return None
        
        all_permissions = db.query(Permission).all()
        
        for permission in all_permissions:
            if permission not in superadmin.permissions:
                superadmin.permissions.append(permission)
        
        db.commit()
        db.refresh(superadmin)
        
        return superadmin

    @staticmethod
    def get_user_permissions(db: Session, user_id: str) -> List[Permission]:
        """
        Get all permissions for a user (via their roles)
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            List of Permission objects user has access to
        """
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return []
        
        permissions = set()
        for role in user.roles:
            for permission in role.permissions:
                permissions.add(permission)
        
        return list(permissions)

    @staticmethod
    def user_has_permission(
        db: Session,
        user_id: str,
        resource: str,
        action: str
    ) -> bool:
        """
        Check if user has specific permission
        
        Args:
            db: Database session
            user_id: User ID
            resource: Resource type
            action: Action type
            
        Returns:
            True if user has permission, False otherwise
        """
        permissions = RoleService.get_user_permissions(db, user_id)
        
        for perm in permissions:
            if perm.resource == resource and perm.action == action:
                return True
        
        return False

    @staticmethod
    def get_role_count(db: Session) -> int:
        """
        Get total role count
        
        Args:
            db: Database session
            
        Returns:
            Total number of roles
        """
        return db.query(Role).count()
