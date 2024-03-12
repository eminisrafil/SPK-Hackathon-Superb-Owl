from __future__ import annotations

import threading
from datetime import timezone, datetime

from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import scoped_session, declarative_base, relationship, Mapped, sessionmaker
from sqlmodel import Session

Base = declarative_base()


class Context(Base):
    __tablename__ = "contexts"
    created_at: datetime = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    updated_at: datetime = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                                  onupdate=lambda: datetime.now(timezone.utc), index=False)
    id: int = Column(Integer, primary_key=True, index=True)
    frames: Mapped["Frame"] = relationship("Frame", back_populates="context")
    utterances: Mapped["Utterance"] = relationship("Utterance", back_populates="context")
    humans: Mapped["Human"] = relationship("Human", back_populates="context")


class Frame(Base):
    __tablename__ = "frames"
    id: int = Column(Integer, primary_key=True, index=True)
    frame: int = Column(String, index=False)
    context_id: int = Column(Integer, ForeignKey("contexts.id"))
    created_at: datetime = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    context: Mapped["Context"] = relationship("Context", back_populates="frames")


class Utterance(Base):
    __tablename__ = "utterances"
    id: int = Column(Integer, primary_key=True, index=True)
    start: float = Column(Float, index=False)
    end: float = Column(Float, index=False)
    spoken_at: str = Column(String, index=False)
    text: str = Column(String, index=False)
    speaker: str = Column(String, index=True)
    created_at: datetime = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    context_id: int = Column(Integer, ForeignKey("contexts.id"))
    context: Mapped["Context"] = relationship("Context", back_populates="utterances")


class Human(Base):
    __tablename__ = "humans"
    id: int = Column(Integer, primary_key=True, index=True)
    name: str = Column(String, index=True)
    text: str = Column(String, index=False)
    picture: bytes = Column(String, index=False)
    created_at: datetime = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    context_id: int = Column(Integer, ForeignKey("contexts.id"))
    context: Mapped["Context"] = relationship("Context", back_populates="humans")


def create_utterance(db: Session, utterance: Utterance) -> Utterance:
    with threading.Lock():
        db.add(utterance)
        db.commit()
        db.refresh(utterance)
        current_context = db.get(Context, utterance.context_id)
        current_context.updated_at = datetime.now(timezone.utc)
        db.commit()
    return utterance


def create_human(db: Session, human: Human) -> Human:
    db.add(human)
    db.commit()
    db.refresh(human)
    current_context = db.get(Context, human.context_id)
    current_context.updated_at = datetime.now(timezone.utc)
    db.commit()
    return human


def create_frame(db: Session, frame: Frame) -> Frame:
    db.add(frame)
    db.commit()
    db.refresh(frame)
    current_context = db.get(Context, frame.context_id)
    current_context.updated_at = datetime.now(timezone.utc)
    db.commit()
    return frame


def create_context(db: Session, context: Context) -> Context:
    db.add(context)
    db.commit()
    db.refresh(context)
    print(context.created_at)
    return context


def get_active_context(db: Session) -> Context:
    # latest context
    return db.query(Context).order_by(Context.id.desc()).first()


class Database:
    def __init__(self, url: str = "sqlite:///test.db"):
        engine = create_engine(
            url,
            pool_size=50,
            max_overflow=100,
            echo=False,
            pool_timeout=30,
            pool_recycle=1800
        )
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
        self._session_local = scoped_session(session_factory)
        self._db = None

    def get_session(self):
        self._db = self._session_local()
        return self._db

    def __del__(self):
        if self._db:
            self._db.close()
        self._db = None
        self._session_local.remove()

    def __enter__(self):
        return self.get_session()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()


def fetch_and_process(model):
    with Database() as db:
        ctx: Context = get_active_context(db)
        query = db.query(model).filter(model.context_id == ctx.id)
        return query.all()

# def main():
#     with Database() as db:
#         context = Context()
#         frame = Frame(frame=1, context=context)
#         utterance = Utterance(start=0, end=1, spoken_at="2021-01-01", text="Hello", speaker="Alice", context=context)
#         create_context(db, context)
#         create_frame(db, frame)
#         create_utterance(db, utterance)
#
#
# if __name__ == "__main__":
#     main()
