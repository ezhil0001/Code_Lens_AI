"""
Database Initialization Script
Initializes PostgreSQL with pgVector extension and creates all tables
"""

import os
from sqlalchemy import text, create_engine
from sqlalchemy.orm import sessionmaker
from app.db.base import Base
from app.db import models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://codelens:codelens_password@localhost:5432/codelens_ai"
)

def init_db():
    """Initialize database with pgVector extension and create all tables"""
    
    engine = create_engine(DATABASE_URL)
    
    try:
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.scalar()
            logger.info(f"✅ Connected to PostgreSQL: {version}")
            
            # Enable pgVector extension
            logger.info("📦 Enabling pgVector extension...")
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            logger.info("✅ pgVector extension enabled")
        
        # Create all tables
        logger.info("📊 Creating tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ All tables created successfully")
        
        # Verify tables created
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """))
            tables = [row[0] for row in result.fetchall()]
            logger.info(f"📋 Tables in database: {', '.join(tables)}")
        
        logger.info("✅ Database initialization complete!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False

if __name__ == "__main__":
    init_db()
