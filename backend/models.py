from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.sql import func

from db import Base


class SeamlessJob(Base):
    __tablename__ = "seamless_jobs"

    id = Column(Integer, primary_key=True, index=True)

    uuid = Column(String(36), unique=True, index=True, nullable=False)

    original_path = Column(String(500), nullable=False)
    output_path = Column(String(500), nullable=False)
    preview_path = Column(String(500), nullable=False)

    status = Column(String(50), nullable=False, default="completed")

    created_at = Column(DateTime(timezone=True), server_default=func.now())