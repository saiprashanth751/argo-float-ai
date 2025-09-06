# create_schema.py
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Float, String, DateTime, ForeignKey
import os

def create_database_schema(db_type="postgresql"):
    """
    Create database schema for PostgreSQL with Docker
    """
    
    if db_type == "postgresql":
        try:
            # Docker PostgreSQL connection
            engine = create_engine('postgresql://argo_user:argo_password@localhost:5432/argo_data')
            print("Using PostgreSQL database with Docker configuration")
        except Exception as e:
            print(f"Error creating engine: {e}")
            return None
    else:
        raise ValueError("db_type must be 'postgresql'")
    
    metadata = MetaData()
    
    # Floats Metadata Table
    floats_metadata = Table(
        'floats_metadata',
        metadata,
        Column('id', Integer, primary_key=True),
        Column('platform_number', String),  # Float ID
        Column('cycle_number', Integer),    # Profile number
        Column('date', DateTime),           # Measurement date
        Column('latitude', Float),
        Column('longitude', Float),
        Column('project_name', String),
        Column('pi_name', String),
        Column('institution', String),
        Column('wmo_inst_type', String),
        Column('data_mode', String),
        Column('source_file', String)
    )
    
    # Measurements Table
    measurements = Table(
        'measurements',
        metadata,
        Column('id', Integer, primary_key=True),
        Column('metadata_id', Integer, ForeignKey('floats_metadata.id')),
        Column('pressure', Float),      # Depth level
        Column('temperature', Float),   # TEMP
        Column('salinity', Float),      # PSAL
        Column('oxygen', Float),        # DOXY (if available)
        Column('chlorophyll', Float),   # CHLA (if available)
        Column('backscatter', Float)    # BBP (if available)
    )
    
    try:
        # Create all tables
        metadata.create_all(engine)
        print("✅ Database schema created successfully!")
        return engine
        
    except Exception as e:
        print(f"❌ Error creating database schema: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Make sure Docker is running: docker-compose up -d")
        print("2. Check if PostgreSQL container is healthy: docker ps")
        return None

def test_connection(engine):
    """Test the database connection"""
    try:
        with engine.connect() as connection:
            print("✅ Database connection successful!")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Creating PostgreSQL database schema...")
    engine = create_database_schema("postgresql")
    
    if engine:
        if test_connection(engine):
            print("🎉 PostgreSQL setup complete!")
        else:
            print("❌ PostgreSQL setup failed!")
    else:
        print("❌ Could not create database schema!")