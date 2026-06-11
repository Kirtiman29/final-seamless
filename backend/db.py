import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

load_dotenv(Path(__file__).resolve().parent / ".env")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:password@localhost/auth_db",
)

# Auto-create MySQL database if it doesn't exist
if DATABASE_URL.startswith("mysql"):
    try:
        from sqlalchemy import text
        base_url, db_name = DATABASE_URL.rsplit("/", 1)
        temp_engine = create_engine(base_url)
        with temp_engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {db_name}"))
            conn.commit()
        temp_engine.dispose()
    except Exception as e:
        print(f"Warning: could not auto-create database: {e}")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
