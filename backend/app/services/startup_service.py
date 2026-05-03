"""
Startup Service - Application initialization on server boot
Equivalent to NestJS OnModuleInit hook
"""

from typing import Optional
from sqlalchemy.orm import Session
import os
from datetime import datetime

from app.models.database import User, Role
from app.services.user_service import UserService
from app.services.role_service import RoleService
from app.auth.service import AuthenticationService


class StartupService:
    """
    Startup Service
    
    Initializes the application on startup:
    - Create default roles
    - Create default permissions
    - Assign permissions to roles
    - Create superadmin user (if doesn't exist)
    
    This is called once when the FastAPI app starts
    Similar to NestJS OnModuleInit hook
    """

    @staticmethod
    def initialize_database(db: Session) -> dict:
        """
        Initialize database with default data
        
        This function is idempotent - it's safe to call multiple times
        
        Args:
            db: Database session
            
        Returns:
            Dictionary with initialization results
            
        Example:
            ```python
            @app.on_event("startup")
            async def startup():
                from app.database.config import SessionLocal
                from app.services.startup_service import StartupService
                
                db = SessionLocal()
                try:
                    result = StartupService.initialize_database(db)
                    print(f"Database initialized: {result}")
                finally:
                    db.close()
            ```
        """
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "initializing",
            "roles_created": 0,
            "permissions_created": 0,
            "superadmin_created": False,
            "messages": []
        }

        try:
            # Step 1: Create default roles
            results["roles_created"] = StartupService._create_default_roles(db, results)

            # Step 2: Create default permissions
            results["permissions_created"] = StartupService._create_default_permissions(db, results)

            # Step 3: Assign permissions to superadmin role
            StartupService._assign_permissions_to_superadmin(db, results)

            # Step 4: Create superadmin user
            results["superadmin_created"] = StartupService._create_superadmin_user(db, results)

            results["status"] = "completed"
            results["messages"].append("Database initialization completed successfully")

        except Exception as e:
            results["status"] = "failed"
            results["messages"].append(f"Error during initialization: {str(e)}")

        return results

    @staticmethod
    def _create_default_roles(db: Session, results: dict) -> int:
        """
        Create default roles
        
        Args:
            db: Database session
            results: Results dictionary to update
            
        Returns:
            Number of new roles created
        """
        default_roles = {
            'superadmin': 'System Administrator - Full access to all resources',
            'admin': 'Administrator - Manage users and content',
            'user': 'User - Can create and edit own content',
            'viewer': 'Viewer - Read-only access to public content'
        }

        created_count = 0

        for role_name, description in default_roles.items():
            existing = RoleService.get_role_by_name(db, role_name)
            if existing:
                results["messages"].append(f"✓ Role '{role_name}' already exists")
            else:
                role = RoleService.create_role(db, role_name, description)
                if role:
                    created_count += 1
                    results["messages"].append(f"✓ Created role '{role_name}'")
                else:
                    results["messages"].append(f"✗ Failed to create role '{role_name}'")

        return created_count

    @staticmethod
    def _create_default_permissions(db: Session, results: dict) -> int:
        """
        Create default permissions for all resources
        
        Args:
            db: Database session
            results: Results dictionary to update
            
        Returns:
            Number of new permissions created
        """
        resources = ['USER', 'ROLE', 'PERMISSION', 'DOCUMENT', 'API_KEY', 'AUDIT_LOG']
        actions = ['CREATE', 'READ', 'UPDATE', 'DELETE', 'LIST']

        created_count = 0

        for resource in resources:
            for action in actions:
                existing = RoleService.get_permission_by_resource_action(db, resource, action)
                if existing:
                    continue  # Skip if already exists
                else:
                    permission = RoleService.create_permission(
                        db,
                        name=f"{resource}_{action}",
                        resource=resource,
                        action=action,
                        description=f"Permission to {action.lower()} {resource.lower()}"
                    )
                    if permission:
                        created_count += 1

        if created_count > 0:
            results["messages"].append(f"✓ Created {created_count} new permissions")
        else:
            results["messages"].append("✓ All permissions already exist")

        return created_count

    @staticmethod
    def _assign_permissions_to_superadmin(db: Session, results: dict) -> None:
        """
        Assign all permissions to superadmin role
        
        Args:
            db: Database session
            results: Results dictionary to update
        """
        superadmin = RoleService.get_role_by_name(db, 'superadmin')
        if not superadmin:
            results["messages"].append("✗ Superadmin role not found")
            return

        all_permissions = db.query(Permission).all()
        initial_perm_count = len(superadmin.permissions)

        for permission in all_permissions:
            if permission not in superadmin.permissions:
                superadmin.permissions.append(permission)

        db.commit()

        final_perm_count = len(superadmin.permissions)
        new_perms = final_perm_count - initial_perm_count

        if new_perms > 0:
            results["messages"].append(f"✓ Assigned {new_perms} new permissions to superadmin")
        else:
            results["messages"].append("✓ Superadmin already has all permissions")

    @staticmethod
    def _create_superadmin_user(db: Session, results: dict) -> bool:
        """
        Create superadmin user if it doesn't exist
        
        Args:
            db: Database session
            results: Results dictionary to update
            
        Returns:
            True if created, False otherwise
        """
        # Check if superadmin already exists
        existing_superadmin = db.query(User).join(User.roles).filter(
            Role.name == 'superadmin'
        ).first()

        if existing_superadmin:
            results["messages"].append(f"✓ Superadmin user '{existing_superadmin.email}' already exists")
            return False

        # Get credentials from environment variables
        superadmin_email = os.getenv(
            'SUPER_USER_EMAIL',
            'admin@codelens.local'
        )
        superadmin_username = os.getenv(
            'SUPER_USER_USERNAME',
            'admin'
        )
        superadmin_password = os.getenv(
            'SUPER_USER_PASSWORD',
            'DefaultPassword123!'  # Change in production!
        )

        # Create superadmin user
        superadmin = UserService.create_superadmin_user(
            db,
            email=superadmin_email,
            username=superadmin_username,
            password=superadmin_password
        )

        if superadmin:
            results["messages"].append(
                f"✓ Created superadmin user '{superadmin_email}' - "
                f"CHANGE PASSWORD ON FIRST LOGIN"
            )
            return True
        else:
            results["messages"].append("✗ Failed to create superadmin user")
            return False

    @staticmethod
    def log_initialization_result(results: dict) -> None:
        """
        Log initialization results (for debugging)
        
        Args:
            results: Results dictionary from initialize_database()
        """
        print("\n" + "="*60)
        print("🚀 DATABASE INITIALIZATION RESULT")
        print("="*60)
        print(f"Status: {results['status'].upper()}")
        print(f"Timestamp: {results['timestamp']}")
        print(f"Roles Created: {results['roles_created']}")
        print(f"Permissions Created: {results['permissions_created']}")
        print(f"Superadmin Created: {results['superadmin_created']}")
        print("\nMessages:")
        for msg in results['messages']:
            print(f"  {msg}")
        print("="*60 + "\n")


# Import at end to avoid circular imports
from app.models.database import Permission
