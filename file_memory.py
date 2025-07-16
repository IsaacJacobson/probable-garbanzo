"""
This module provides the FileMemory class to handle interaction with the SQLite database created from a CSV file, including conversion, vector database creation, and querying capabilities.
"""
 

import os
from typing import Optional, List, Dict
import pandas as pd
import sqlite3
from openai import OpenAI
import numpy as np
import faiss

from config import TOPK_COLS, EMBEDDINGS_MODEL, API_KEY


class FileMemory():
    """
    FileMemory class to handle CSV to SQLite conversion and vector database creation.
    """

    def __init__(self, csv: str) -> None:      
        self.csv = csv    
        self.convert_csv_to_sqlite()
        self.create_vecDB()


    def convert_csv_to_sqlite(self) -> None:
        """
        Convert a CSV file to an SQLite database stored in the same directory.
        """

        try:
            if not os.path.exists(self.csv):
                raise FileNotFoundError(f"CSV file '{self.csv}' does not exist.")

            df = pd.read_csv(self.csv, sep=';')
            df.columns = df.columns.str.replace(' ', '_', regex=False)
            csv_dir = os.path.dirname(self.csv)
            filename, _ = os.path.splitext(os.path.basename(self.csv))
            db_path = os.path.join(csv_dir, f"{filename}.sqilte")
            conn = sqlite3.connect(db_path)
            df.to_sql('debug_table', conn, if_exists='replace', index=False)
            conn.close()

            self.client = OpenAI(api_key=API_KEY)


            if not os.path.exists(db_path):
                raise FileNotFoundError(f"Database file '{db_path}' was not created.")

            self.cols = df.columns.tolist()
            self.dbPath = db_path

        except Exception as e:
            raise RuntimeError(f"Failed to create SQLite database: {e}")


    def get_embedding(self, text: str, model: str = EMBEDDINGS_MODEL) -> list[float]:
        """
        Get the embedding for a given text using the specified model.
        """
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(
            input=[text],
            model=model
        )
        return response.data[0].embedding

 
    def create_vecDB(self) -> None:
        """
        Create a vector database from the columns of the SQLite database.
        """

        try:
            embeddings = [self.get_embedding(col) for col in self.cols]
            embedding_matrix = np.array(embeddings).astype("float32")
            faiss.normalize_L2(embedding_matrix)
            index = faiss.IndexFlatL2(embedding_matrix.shape[1])
            index.add(embedding_matrix)
            self.colDB = index
        except Exception as e:
            raise RuntimeError(f"Failed to create vector database: {e}")


    def query_vecDB(self, query: str, top_k: Optional[int] = TOPK_COLS) -> List[str]:
        """
        Query the FAISS vector database with a natural language query returning the top_k results (by similarity search).
        """
       
        try:
            if not self.colDB:
                raise ValueError("Vector database is not initialized.")

            query_embedding = np.array([self.get_embedding(query)]).astype("float32")
            faiss.normalize_L2(query_embedding)
            _, indices = self.colDB.search(query_embedding, top_k)
            results = [self.cols[i] for i in indices[0] if i < len(self.cols)]

            return results
        except Exception as e:
            raise RuntimeError(f"Failed to query FAISS vector database: {e}")


    def describe_data(self, data_dict: Dict[str, List], cols: List[str] = None) -> str:
        """
        Generate a description of the data in the specified columns.
        """

        description = []
        for column, values in data_dict.items():
            if column in cols:
                description.append(f"Column '{column}': Examples -> {', '.join(map(str, values))}")
        return "\n".join(description)


    def format_data(self, columns: List[str], rows: List[tuple]) -> Dict[str, List]:
        """
        Format the fetched data into a dictionary with column names as keys and lists of values as values.
        """

        out = {}
        for i, col in enumerate(columns):
            out[col] = [row[i] for row in rows]

        return out

   
    def execute_sql_statement(self, sql_statement: str) -> Dict[str, List]:
        """
        Execute a SQL statement on the SQLite database and return the results as a dictionary.
        """

        try:
            conn = sqlite3.connect(self.dbPath)
            cursor = conn.cursor()
            cursor.execute(sql_statement)
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            results = self.format_data(columns, rows)
            conn.close()
            return results
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to execute SQL statement: {e}")


    def get_examples(self, cols: List[str], k:Optional[int] = 5) -> Dict[str, List]:
        """
        Get k examples for each specified column from the SQLite database.
        """

        out = {}
        conn = sqlite3.connect(self.dbPath)
        cursor = conn.cursor()
        for col in cols:
            cursor.execute(f"SELECT {col} FROM debug_table LIMIT {k}")
            rows = cursor.fetchall()
            out[col] = [row[0] for row in rows]
        return out

       
    def topK_cols(self, query: str) -> List[str]:
        """
        Get the top k columns from the vector database based on the query.
        """
        if not hasattr(self, 'colDB'):
            raise ValueError("Vector database is not initialized.")

        results = self.query_vecDB(query)
        return results

       
    def sample_cols(self, query: str) -> List[str]:
        """
        Sample k columns from the vector database based on the query.
        """

        top_cols = self.topK_cols(query)
        examples = self.get_examples(top_cols)
        description = self.describe_data(examples, top_cols)
        return description