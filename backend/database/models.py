import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class UserRole(str, enum.Enum):
    VIEWER = "VIEWER"
    CLINICIAN = "CLINICIAN"
    ADMIN = "ADMIN"


class BatchJobStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.VIEWER, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    predictions: Mapped[list["PredictionLog"]] = relationship(back_populates="user")


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    power: Mapped[float] = mapped_column(Float, nullable=False)
    time: Mapped[float] = mapped_column(Float, nullable=False)
    antenna_type: Mapped[str] = mapped_column(String(100), nullable=False)
    predicted_diameter: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_length: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_volume: Mapped[float] = mapped_column(Float, nullable=False)
    model_used: Mapped[str] = mapped_column(String(100), nullable=False)
    response_time_ms: Mapped[float] = mapped_column(Float, nullable=False)

    user: Mapped[User | None] = relationship(back_populates="predictions")


class BatchJob(Base):
    __tablename__ = "batch_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    status: Mapped[BatchJobStatus] = mapped_column(Enum(BatchJobStatus), default=BatchJobStatus.PENDING)
    input_file_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    total_rows: Mapped[int] = mapped_column(Integer, default=0)
    successful: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
