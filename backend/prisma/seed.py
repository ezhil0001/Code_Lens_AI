"""
Database seeding script for CodeLens_AI

This script populates the database with initial data:
- ✅ Default roles and permissions
- ✅ Admin user (if configured)
"""

import os
import hashlib
from datetime import datetime
from prisma import Prisma


async def seed():
    """Seed the database with initial data."""
    db = Prisma()
    await db.connect()
    
    try:
        print("🌱 Starting database seed...")
        
        # 1. Create default roles
        print("📋 Creating default roles...")
        
        roles_data = [
            {
                "name": "admin",
                "description": "Administrator with full access",
            },
            {
                "name": "user",
                "description": "Regular user with standard access",
            },
            {
                "name": "viewer",
                "description": "Read-only access",
            },
        ]
        
        roles = {}
        for role_data in roles_data:
            role = await db.role.upsert(
                where={"name": role_data["name"]},
                data={
                    "create": {
                        "name": role_data["name"],
                    },
                    "update": {},
                }
            )
            roles[role_data["name"]] = role
            print(f"   ✅ Role created: {role.name}")
        
        # 3. Create default permissions
        print("🔐 Creating default permissions...")
        
        permissions_data = [
            {"name": "users:create", "description": "Create users"},
            {"name": "users:read", "description": "Read users"},
            {"name": "users:update", "description": "Update users"},
            {"name": "users:delete", "description": "Delete users"},
            {"name": "roles:create", "description": "Create roles"},
            {"name": "roles:read", "description": "Read roles"},
            {"name": "roles:update", "description": "Update roles"},
            {"name": "roles:delete", "description": "Delete roles"},
            {"name": "api_keys:create", "description": "Create API keys"},
            {"name": "api_keys:read", "description": "Read API keys"},
            {"name": "api_keys:delete", "description": "Delete API keys"},
            {"name": "audit_logs:read", "description": "Read audit logs"},
            {"name": "code:read", "description": "Read code repositories"},
            {"name": "code:create", "description": "Create code repositories"},
            {"name": "code:delete", "description": "Delete code repositories"},
            {"name": "chat:create", "description": "Create chat messages"},
            {"name": "chat:read", "description": "Read chat messages"},
        ]
        
        permissions = {}
        for perm_data in permissions_data:
            perm = await db.permission.upsert(
                where={"name": perm_data["name"]},
                data={
                    "create": perm_data,
                    "update": {},
                }
            )
            permissions[perm_data["name"]] = perm
            print(f"   ✅ Permission created: {perm.name}")
        
        # 4. Assign permissions to admin role
        print("🔗 Assigning permissions to admin role...")
        admin_role = roles["admin"]
        admin_role_updated = await db.role.update(
            where={"id": admin_role.id},
            data={
                "permissions": {
                    "set": [{"id": perm.id} for perm in permissions.values()]
                }
            }
        )
        print(f"   ✅ Assigned {len(permissions)} permissions to admin role")
        
        # 5. Assign limited permissions to user role
        print("🔗 Assigning permissions to user role...")
        user_role = roles["user"]
        user_permissions = [
            permissions["users:read"],
            permissions["code:read"],
            permissions["chat:create"],
            permissions["chat:read"],
        ]
        user_role_updated = await db.role.update(
            where={"id": user_role.id},
            data={
                "permissions": {
                    "set": [{"id": perm.id} for perm in user_permissions]
                }
            }
        )
        print(f"   ✅ Assigned {len(user_permissions)} permissions to user role")
        
        # 6. Assign read-only permissions to viewer role
        print("🔗 Assigning permissions to viewer role...")
        viewer_role = roles["viewer"]
        viewer_permissions = [
            permissions["users:read"],
            permissions["roles:read"],
            permissions["code:read"],
            permissions["chat:read"],
            permissions["audit_logs:read"],
        ]
        viewer_role_updated = await db.role.update(
            where={"id": viewer_role.id},
            data={
                "permissions": {
                    "set": [{"id": perm.id} for perm in viewer_permissions]
                }
            }
        )
        print(f"   ✅ Assigned {len(viewer_permissions)} permissions to viewer role")
        
        # 7. Create admin user (if configured in environment)
        super_user_email = os.getenv("SUPER_USER_EMAIL")
        super_user_password = os.getenv("SUPER_USER_PASSWORD")
        
        if super_user_email and super_user_password:
            print(f"👤 Creating admin user...")
            
            # Hash password (in production, use proper password hashing)
            hashed_password = hashlib.sha256(super_user_password.encode()).hexdigest()
            
            user = await db.user.upsert(
                where={"email": super_user_email},
                data={
                    "create": {
                        "email": super_user_email,
                        "password": hashed_password,
                        "firstName": os.getenv("SUPER_USER_FIRST_NAME", "Admin"),
                        "lastName": os.getenv("SUPER_USER_LAST_NAME", "User"),
                        "roleId": roles["admin"].id,
                        "isAdmin": True,
                        "isLoggedIn": False,
                    },
                    "update": {},
                }
            )
            print(f"   ✅ Admin user created: {user.email}")
        else:
            print("   ℹ️  No admin user configured (set SUPER_USER_EMAIL and SUPER_USER_PASSWORD)")
        
        print("\n✅ Database seeding completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during seeding: {e}")
        raise
    
    finally:
        await db.disconnect()


async def main():
    """Main entry point."""
    await seed()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
