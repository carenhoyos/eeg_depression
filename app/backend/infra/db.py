"""Persistencia relacional (SQLAlchemy 2.0): usuarios (rol psiquiatra), pacientes y resumen de análisis.
SQLite por defecto; la URL viene de la configuración, así que Postgres es un cambio de cadena de conexión."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(120), default="")
    role: Mapped[str] = mapped_column(String(32))
    password_hash: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Patient(Base):
    __tablename__ = "patients"
    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    pseudonym: Mapped[str] = mapped_column(String(64), index=True)
    age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    sex: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    medication: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    analyses: Mapped[list["AnalysisRecord"]] = relationship(back_populates="patient", cascade="all, delete-orphan")


class AnalysisRecord(Base):
    __tablename__ = "analyses"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    patient_id: Mapped[Optional[int]] = mapped_column(ForeignKey("patients.id"), nullable=True, index=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    status: Mapped[str] = mapped_column(String(16), default="queued")
    input_kind: Mapped[str] = mapped_column(String(32))
    filename: Mapped[str] = mapped_column(String(200))
    pretest_prevalence: Mapped[float] = mapped_column(Float)
    condition: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    p_calibrated: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    p_posterior: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tier_code: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    patient: Mapped[Optional[Patient]] = relationship(back_populates="analyses")

    def to_dict(self) -> dict[str, Any]:
        return {"analysis_id": self.id, "patient_id": self.patient_id, "created_at": self.created_at.isoformat(), "status": self.status,
                "input_kind": self.input_kind, "filename": self.filename, "pretest_prevalence": self.pretest_prevalence,
                "condition": self.condition, "p_calibrated": self.p_calibrated, "p_posterior": self.p_posterior,
                "tier_code": self.tier_code, "model_version": self.model_version, "error": self.error}


class Database:
    def __init__(self, url: str):
        self.engine = create_engine(url, future=True, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)

    def session(self) -> Session:
        return self.Session()

    # ---- usuarios
    def get_user(self, username: str) -> Optional[User]:
        with self.session() as s:
            return s.scalar(select(User).where(User.username == username))

    def ensure_user(self, username: str, password_hash: str, role: str, full_name: str = "") -> User:
        with self.session() as s:
            u = s.scalar(select(User).where(User.username == username))
            if u is None:
                u = User(username=username, password_hash=password_hash, role=role, full_name=full_name)
                s.add(u); s.commit(); s.refresh(u)
            return u

    # ---- pacientes
    def list_patients(self, owner_id: int) -> list[Patient]:
        with self.session() as s:
            return list(s.scalars(select(Patient).where(Patient.owner_id == owner_id).order_by(Patient.created_at.desc())))

    def get_patient(self, owner_id: int, patient_id: int) -> Optional[Patient]:
        with self.session() as s:
            return s.scalar(select(Patient).where(Patient.id == patient_id, Patient.owner_id == owner_id))

    def create_patient(self, owner_id: int, **fields: Any) -> Patient:
        with self.session() as s:
            p = Patient(owner_id=owner_id, **fields); s.add(p); s.commit(); s.refresh(p); return p

    def update_patient(self, owner_id: int, patient_id: int, **fields: Any) -> Optional[Patient]:
        with self.session() as s:
            p = s.scalar(select(Patient).where(Patient.id == patient_id, Patient.owner_id == owner_id))
            if p is None:
                return None
            for k, v in fields.items():
                if v is not None:
                    setattr(p, k, v)
            s.commit(); s.refresh(p); return p

    def delete_patient(self, owner_id: int, patient_id: int) -> bool:
        with self.session() as s:
            p = s.scalar(select(Patient).where(Patient.id == patient_id, Patient.owner_id == owner_id))
            if p is None:
                return False
            s.delete(p); s.commit(); return True

    # ---- análisis
    def create_analysis(self, **fields: Any) -> AnalysisRecord:
        with self.session() as s:
            a = AnalysisRecord(**fields); s.add(a); s.commit(); s.refresh(a); return a

    def update_analysis(self, analysis_id: str, **fields: Any) -> None:
        with self.session() as s:
            a = s.get(AnalysisRecord, analysis_id)
            if a is not None:
                for k, v in fields.items():
                    setattr(a, k, v)
                s.commit()

    def get_analysis(self, owner_id: int, analysis_id: str) -> Optional[AnalysisRecord]:
        with self.session() as s:
            return s.scalar(select(AnalysisRecord).where(AnalysisRecord.id == analysis_id, AnalysisRecord.owner_id == owner_id))

    def list_analyses(self, owner_id: int, patient_id: Optional[int] = None, limit: int = 100) -> list[AnalysisRecord]:
        with self.session() as s:
            q = select(AnalysisRecord).where(AnalysisRecord.owner_id == owner_id)
            if patient_id is not None:
                q = q.where(AnalysisRecord.patient_id == patient_id)
            return list(s.scalars(q.order_by(AnalysisRecord.created_at.desc()).limit(limit)))
