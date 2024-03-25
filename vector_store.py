import sqlite3
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

class SQLiteVectorStore:
    def __init__(self, db_name, word_embedding_model_path):
        self.conn = sqlite3.connect(db_name)
        self.word_model = KeyedVectors.load(word_embedding_model_path, mmap='r')
        self.create_table()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS vectors
                          (id INTEGER PRIMARY KEY, vector BLOB, binary_data BLOB)''')
        self.conn.commit()

    def insert_vector(self, vector, binary_data):
        cursor = self.conn.cursor()
        vector_blob = vector.tobytes()
        cursor.execute('INSERT INTO vectors (vector, binary_data) VALUES (?, ?)', (vector_blob, binary_data))
        self.conn.commit()

    def get_all_vectors(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, vector FROM vectors')
        results = cursor.fetchall()
        vectors = []
        vector_ids = []
        for result in results:
            vector_id = result[0]
            vector_blob = result[1]
            vector = np.frombuffer(vector_blob, dtype=np.float64)
            vectors.append(vector)
            vector_ids.append(vector_id)
        return vectors, vector_ids

    def search_vectors(self, input_text, top_n=10):
        input_vector = self.create_vector_from_text(input_text)
        all_vectors, vector_ids = self.get_all_vectors()
        similarities = cosine_similarity([input_vector], all_vectors)
        sorted_indices = np.argsort(similarities[0])[::-1][:top_n]
        top_results = [(vector_ids[i], similarities[0][i]) for i in sorted_indices]
        return top_results

    def create_vector_from_text(self, input_text):
        words = input_text.split()
        word_vectors = []
        for word in words:
            try:
                word_vector = self.word_model[word]
                word_vectors.append(word_vector)
            except KeyError:
                # If word not in vocabulary, ignore it
                pass
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:
            # If no word vectors found, return None
            return None

    def close(self):
        self.conn.close()

# Example usage:
if __name__ == "__main__":
    db_name = 'vector_store.db'
    # Downloaded from https://www.kaggle.com/datasets/adarshsng/googlenewsvectors?resource=download
    word_embedding_model_path = 'path_to_word2vec_model.bin'  # Provide path to your Word2Vec model
    vector_store = SQLiteVectorStore(db_name, word_embedding_model_path)

    # Example input text
    input_text = "Example input text"

    # Search for top 10 most relevant results
    top_results = vector_store.search_vectors(input_text)
    print("Top 10 most relevant results:")
    for result in top_results:
        vector_id, similarity = result
        print(f"Vector ID: {vector_id}, Similarity: {similarity}")

    vector_store.close()
