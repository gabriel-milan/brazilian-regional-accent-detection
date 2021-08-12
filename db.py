import json
from os import getenv
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Float, Integer, String, Boolean, create_engine
from sqlalchemy_utils import database_exists, create_database


__all__ = ["DBManager"]

Base = declarative_base()


class TrainTable(Base):

    __tablename__ = "train"

    # Columns
    id = Column(Integer, primary_key=True)
    config = Column(String)
    val_auc = Column(Float)
    weights_path = Column(String)


class DBManager:

    def __init__(self):
        connection_url: str = getenv("DB_CONNECTION_URL", "")
        if connection_url == "":
            raise Exception(
                "Environment variable DB_CONNECTION_URL not set.")
        self._engine = create_engine(connection_url)
        if not database_exists(self._engine.url):
            create_database(self._engine.url)
            Base.metadata.create_all(self._engine)

        self._session: Session = sessionmaker(bind=self._engine)()

    @property
    def session(self) -> Session:
        return self._session

    def add_result(self, config: dict, val_auc: float, weights_path: str):
        result = TrainTable(
            val_auc=val_auc,
            weights_path=weights_path,
            config=json.dumps(config)
        )
        self._session.add(result)
        self._session.commit()
