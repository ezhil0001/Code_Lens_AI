#!/usr/bin/env python
"""
Prisma ORM Migration Manager for CodeLens_AI

This script handles database schema migrations using Prisma ORM.
For Python projects, Prisma uses an internal migration system.

Usage:
    python manage_migrations.py generate <name>  - Generate a migration
    python manage_migrations.py migrate          - Apply pending migrations
    python manage_migrations.py status           - Check migration status
    python manage_migrations.py reset            - Reset database (development only)
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.absolute()
PRISMA_DIR = PROJECT_ROOT / "prisma"


def run_command(command: list[str]) -> int:
    """Run a command and return exit code."""
    try:
        result = subprocess.run(command, cwd=PROJECT_ROOT)
        return result.returncode
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return 1


def generate_migration(name: str) -> int:
    """Generate a new migration based on schema changes."""
    print(f"📝 Generating migration: {name}")
    return run_command([
        "prisma",
        "migrate",
        "dev",
        "--name",
        name,
        "--create-only"
    ])


def apply_migrations() -> int:
    """Apply pending migrations to the database."""
    print("🔄 Applying pending migrations...")
    return run_command([
        "prisma",
        "migrate",
        "deploy"
    ])


def check_status() -> int:
    """Check migration status."""
    print("📊 Checking migration status...")
    return run_command([
        "prisma",
        "migrate",
        "status"
    ])


def reset_database() -> int:
    """
    Reset database (development only).
    WARNING: This will delete all data!
    """
    env = os.getenv("ENVIRONMENT", "development")
    
    if env != "development":
        print("❌ Database reset is only allowed in development environment!")
        print(f"   Current environment: {env}")
        return 1
    
    confirm = input("⚠️  This will DELETE ALL DATA. Type 'yes' to confirm: ")
    if confirm.lower() != "yes":
        print("❌ Reset cancelled")
        return 1
    
    print("🔄 Resetting database...")
    return run_command([
        "prisma",
        "migrate",
        "reset",
        "--force"
    ])


def seed_database() -> int:
    """Seed database with initial data."""
    print("🌱 Seeding database...")
    seed_script = PROJECT_ROOT / "prisma" / "seed.py"
    
    if not seed_script.exists():
        print(f"⚠️  Seed script not found: {seed_script}")
        return 1
    
    return run_command(["python", str(seed_script)])


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prisma ORM Migration Manager for CodeLens_AI"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Generate migration
    gen_parser = subparsers.add_parser("generate", help="Generate a new migration")
    gen_parser.add_argument("name", help="Migration name")
    
    # Apply migrations
    subparsers.add_parser("migrate", help="Apply pending migrations")
    subparsers.add_parser("deploy", help="Deploy migrations (production)")
    
    # Check status
    subparsers.add_parser("status", help="Check migration status")
    
    # Reset database
    reset_parser = subparsers.add_parser("reset", help="Reset database (dev only)")
    reset_parser.add_argument(
        "--force",
        action="store_true",
        help="Force reset without confirmation"
    )
    
    # Seed database
    subparsers.add_parser("seed", help="Seed database with initial data")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate function
    if args.command == "generate":
        return generate_migration(args.name)
    elif args.command == "migrate":
        return apply_migrations()
    elif args.command == "deploy":
        return apply_migrations()
    elif args.command == "status":
        return check_status()
    elif args.command == "reset":
        return reset_database()
    elif args.command == "seed":
        return seed_database()
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
