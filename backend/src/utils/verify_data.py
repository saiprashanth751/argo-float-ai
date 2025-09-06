# verify_data.py
from sqlalchemy import create_engine, text
import os

def get_database_engine(db_type="postgresql"):
    """Get database engine with Docker PostgreSQL"""
    
    if db_type == "postgresql":
        try:
            # Docker PostgreSQL connection
            engine = create_engine('postgresql://argo_user:argo_password@localhost:5432/argo_data')
            
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            print("✅ Connected to PostgreSQL in Docker")
            return engine
            
        except Exception as e:
            print(f"❌ Could not connect to PostgreSQL: {e}")
            print("💡 Make sure Docker is running: docker-compose up -d")
            return None
    
    else:
        raise ValueError("db_type must be 'postgresql'")

def verify_data():
    """Verify the data in the database"""
    
    engine = get_database_engine("postgresql")
    if engine is None:
        return
    
    try:
        with engine.connect() as conn:
            # Check if tables exist
            try:
                # Check metadata table
                result = conn.execute(text("SELECT COUNT(*) FROM floats_metadata"))
                metadata_count = result.scalar()
                print(f"📊 Floats in metadata table: {metadata_count}")
                
                # Check measurements table
                result = conn.execute(text("SELECT COUNT(*) FROM measurements"))
                measurements_count = result.scalar()
                print(f"📊 Measurements in database: {measurements_count}")
                
                if metadata_count == 0:
                    print("\n❌ No data found in the database.")
                    print("💡 Please run the conversion script first:")
                    print("  python3 convert_netcdf_to_sql.py")
                    return
                
                # Show sample metadata
                print("\n" + "="*50)
                print("📋 SAMPLE METADATA:")
                print("="*50)
                result = conn.execute(text("SELECT * FROM floats_metadata LIMIT 1"))
                for row in result:
                    row_dict = dict(row._mapping)
                    for key, value in row_dict.items():
                        print(f"{key:20}: {value}")
                
                if measurements_count > 0:
                    # Show sample measurements
                    print("\n" + "="*50)
                    print("📋 SAMPLE MEASUREMENTS:")
                    print("="*50)
                    result = conn.execute(text("SELECT * FROM measurements LIMIT 5"))
                    for i, row in enumerate(result, 1):
                        print(f"\nMeasurement {i}:")
                        row_dict = dict(row._mapping)
                        for key, value in row_dict.items():
                            if value is not None:
                                if key in ['pressure', 'temperature', 'salinity'] and isinstance(value, float):
                                    print(f"  {key:15}: {value:.4f}")
                                else:
                                    print(f"  {key:15}: {value}")
                
                # Show statistics
                print("\n" + "="*50)
                print("📈 DATA STATISTICS:")
                print("="*50)
                
                # Pressure range
                result = conn.execute(text("SELECT MIN(pressure), MAX(pressure) FROM measurements WHERE pressure IS NOT NULL"))
                min_pres, max_pres = result.fetchone()
                if min_pres is not None:
                    print(f"Pressure range    : {min_pres:.2f} - {max_pres:.2f} dbar")
                
                # Temperature range
                result = conn.execute(text("SELECT MIN(temperature), MAX(temperature) FROM measurements WHERE temperature IS NOT NULL"))
                min_temp, max_temp = result.fetchone()
                if min_temp is not None:
                    print(f"Temperature range : {min_temp:.4f} - {max_temp:.4f} °C")
                
                # Salinity range
                result = conn.execute(text("SELECT MIN(salinity), MAX(salinity) FROM measurements WHERE salinity IS NOT NULL"))
                min_sal, max_sal = result.fetchone()
                if min_sal is not None:
                    print(f"Salinity range    : {min_sal:.4f} - {max_sal:.4f} PSU")
                
                # Count valid measurements by parameter
                result = conn.execute(text("""
                    SELECT 
                        COUNT(CASE WHEN pressure IS NOT NULL THEN 1 END) as pressure_count,
                        COUNT(CASE WHEN temperature IS NOT NULL THEN 1 END) as temperature_count,
                        COUNT(CASE WHEN salinity IS NOT NULL THEN 1 END) as salinity_count,
                        COUNT(CASE WHEN oxygen IS NOT NULL THEN 1 END) as oxygen_count
                    FROM measurements
                """))
                counts = result.fetchone()
                print(f"\n✅ Valid measurements:")
                print(f"  Pressure      : {counts[0]}")
                print(f"  Temperature   : {counts[1]}")
                print(f"  Salinity      : {counts[2]}")
                print(f"  Oxygen        : {counts[3]}")
                
                # Show unique platforms
                result = conn.execute(text("SELECT DISTINCT platform_number FROM floats_metadata ORDER BY platform_number"))
                platforms = [row[0] for row in result]
                print(f"\n🌊 Float platforms   : {', '.join(platforms)}")
                
                # Show date range
                result = conn.execute(text("SELECT MIN(date), MAX(date) FROM floats_metadata"))
                min_date, max_date = result.fetchone()
                if min_date:
                    print(f"📅 Date range        : {min_date} to {max_date}")
                
            except Exception as table_error:
                print(f"❌ Error accessing tables: {table_error}")
                print("💡 The database tables may not exist.")
                print("💡 Please run create_schema.py first, then convert_netcdf_to_sql.py")
                
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        print("💡 Please check your database configuration.")

if __name__ == "__main__":
    print("🔍 Verifying data in PostgreSQL database...")
    print("="*50)
    verify_data()