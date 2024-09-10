import subprocess
import sys

from datetime import datetime
import json

from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

# Define the SQLite database
DATABASE_URL = "sqlite:///./mydatabase.db"

# Create an engine and a session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a base class for our models
Base = declarative_base()

# Define your database models
class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(String)
    filepath = Column(String)

    # Relationship to DatasetVersion
    rules = relationship("DQRule", back_populates="dataset")
    versions = relationship("DatasetVersion", back_populates="dataset")

class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    version_number = Column(String, nullable=False)
    description = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="versions")
    operations = relationship("DatasetOperation", back_populates="version")

class DatasetOperation(Base):
    __tablename__ = "dataset_operations"

    id = Column(Integer, primary_key=True, index=True)
    version_id = Column(Integer, ForeignKey("dataset_versions.id"), nullable=False)
    operation_type = Column(String, nullable=False)
    parameters = Column(Text, nullable=False)  # JSON encoded string to store action details

    version = relationship("DatasetVersion", back_populates="operations")

class DQRule(Base):
    __tablename__ = "dq_rules"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    rule_name = Column(String, nullable=False)
    rule_type = Column(String, nullable=False)
    target_column = Column(String, nullable=False)
    condition = Column(Text, nullable=False)
    severity = Column(String, nullable=False)
    message = Column(Text, nullable=True)

    dataset = relationship("Dataset", back_populates="rules")

# Create the database tables
Base.metadata.create_all(bind=engine)

# Dependency to get the session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
