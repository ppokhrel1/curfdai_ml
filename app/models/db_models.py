from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import ForeignKey, Text, String, JSON, Integer
from typing import List
import uuid
from datetime import datetime

class Base(DeclarativeBase):
    pass

class AssetFile(Base):
    __tablename__ = "asset_files"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    bucket: Mapped[str] = mapped_column(String(100))
    path: Mapped[str] = mapped_column(Text)
    file_type: Mapped[str] = mapped_column(String(20))
    filesize: Mapped[int] = mapped_column(Integer)
    file_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    generated_assets: Mapped[List["GeneratedAsset"]] = relationship(back_populates="file", cascade="all, delete-orphan")

class GeneratedAsset(Base):
    __tablename__ = "generated_assets"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    resource_type: Mapped[str] = mapped_column(String(50))
    resource_id: Mapped[str] = mapped_column(String(36))
    target_format: Mapped[str] = mapped_column(String(50))
    file_id: Mapped[str] = mapped_column(ForeignKey("asset_files.id"))
    file_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    file: Mapped[AssetFile] = relationship(back_populates="generated_assets")