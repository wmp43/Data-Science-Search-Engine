from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from src.base_classes import *


class RelationalDB(ABC):
    """
    Schema: id:int (PK) | wiki_id:int | category: str | wiki_url:str | super_categories: List[str] | sub_categories:List[str] | sub_articles: List[str]
    """

    @abstractmethod
    def connect(self, credentials: Dict[str, Any]):
        pass

    @abstractmethod
    def add_record(self, category: Category):
        pass

    @abstractmethod
    def get_record(self, category_id: int) -> Optional[Category]:
        pass

    @abstractmethod
    def update_record(self, category: Category):
        pass

    @abstractmethod
    def add_super_category(self, category_id: int, super_category: str):
        """
        Implementation for appending a super category to the super_category list.
        This handles the case for when a category can be reached from multiple super-categories
        Cat:Machine Leanring -> Cat:Neural Networks
        Cat: Artifical Intelligence -> Cat: Neural networks

        :param category_id: ID for category
        :param super_category: new super category
        :return:
        """
        pass


class LocalRelationalDB(RelationalDB):
    def connect(self, credentials: Dict[str, Any]):
        # Implement actual connection logic
        pass

    def add_record(self, category: Category):
        # Logic to add a category to the database
        pass

    def get_record(self, category_id: int) -> Optional[Category]:
        # Logic to retrieve a category from the database
        pass

    def update_record(self, category: Category):
        # Logic to update a category in the database
        pass

    def add_super_category(self, category_id: int, super_category: str):
        pass
