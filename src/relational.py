from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from src.models import *
import psycopg2
from psycopg2 import sql


class RelationalDB(ABC):
    """
    Schema: id:int (PK) | wiki_id:int | category: str | wiki_url:str | super_categories: List[str] | sub_categories:List[str] | sub_articles: List[str]
    """

    @abstractmethod
    def _connect(self):
        pass

    @abstractmethod
    def add_record(self):
        pass

    @abstractmethod
    def get_record(self, title: str):
        pass

    @abstractmethod
    def update_record(self, title: str):
        pass


class PgVector(RelationalDB):
    """
    Extension of abstract base class specifically for pgvetor on aws rds

    """
    def __init__(self, dbname, user, password, host, port):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.conn = None
        self._connect()

    def _connect(self):
        try:
            self.conn = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
        except psycopg2.Error as e:
            print(f"Unable to connect to the database: {e}")

    def add_record(self):
        # Logic to add a category to the database
        pass

    def get_record(self, title: str):
        # Logic to retrieve a category from the database
        pass

    def update_record(self, title: str):
        # Logic to update a category in the database
        pass

    def add_super_category(self, category_id: int, super_category: str):
        pass


class EmbeddingModelTable:
    def __init__(self, dbname, user, password, host, port):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.conn = None
        self._connect()

    def _connect(self):
        try:
            self.conn = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            print(f'Connected to the database {self.dbname} successfully!')
        except psycopg2.Error as e:
            print(f"Unable to connect to the database: {e}")

    def add_record(self, id, text, title):
        """
        :param id: Id of the record
        :param text: Resultant Text of the record
        :param title: Title of the wikipedia article
        :return: None i gues
        """
        with self.conn.cursor() as cur:
            cur.execute("INSERT INTO embedding_model_dev (id, text, title) VALUES (%s, %s, %s)", (id, text, title))
            self.conn.commit()

    def get_record(self, title):
        """
        :param title: Title to search
        :return: Record with the title
        """
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM embedding_model_dev WHERE title = %s", (title,))
            return cur.fetchone()

    def print_sample_data(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM embedding_model_dev LIMIT 3")
            records = cur.fetchall()
            for record in records:
                print(record)

    def update_record(self, title: str):
        # Logic to update a category in the database
        pass

    def close_connection(self):
        if self.conn is not None:
            self.conn.close()

