"""
This fil should define classes and connection, search, etc. methods for the vector db pinecone.
"""
from dataclasses import dataclass
from src.
import pinecone

@dataclass
class Pinecone:
    def __init__(self, api_key, env, dim, metric, idx):
        self.api_key = api_key
        self.idx = idx
        self.env = env
        self.dim = 512
        self.metric = "dotproduct"
        self.backup_idx = f"backup_{idx}"
        pinecone.init(api_key=self.api_key, environment=self.env)

    def create_idx(self, idx):
        try:
            pinecone.create_index(idx, dimension=self.dim, metric=self.metric)
            print(f"Success creating {self.idx} -- dimensions: {self.dim}")
        except Exception as e:
            print(f'Exception Error: {str(e)}')

    def create_backup_idx(self):
        try:
            pinecone.create_collection(f"backup_idx_{self.idx}", self.idx)

        except Exception as e:
            print(f"Error Occured: {str(e)}")

    def delete_backup_idx(self):
        try:
            pinecone.delete_collection(self.backup_idx)

        except Exception as e:
            print(f"Error Occured: {str(e)}")

    # todo: figure out update for the adding of categories
    # def update_metadata(self):
    #     index = pinecone.Index(self.idx)
    #     try:
    #         update_response = index.Update

    def upsert_record(self):


    def idx_query(self) -> bool:
        """
        Return a record in the db that may be edited
        :return: bool
        """
        pass

    def sematic_query(self, query: str):
        """
        :param query: string query
        :return: returned records (vectors)
        """

    def delete_index(self):
        key1 = 'I want to delete this idx'
        user_input = input(f"Enter: f'I want to delete {self.idx}': ")
        if user_input == key1:
            pinecone.delete_index(self.idx)
