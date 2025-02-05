import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
import os

class VectorStoreManager:
    def __init__(self, db_path: str, embeddings_model: str = "sentence-transformers/all-mpnet-base-v2"):
        self.db_path = db_path
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model, model_kwargs={'device': 'cpu'})
        self.vectorstore = self.load_vectorstore()

    def load_vectorstore(self) -> FAISS:
        """Load or initialize a FAISS vector store."""
        if os.path.exists(self.db_path):
            print(f"Loading existing vector store from {self.db_path}")
            return FAISS.load_local(self.db_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            print("Creating new vector store.")
            dimensions = len(self.embeddings.embed_query("dummy"))
            return FAISS(embedding_function=self.embeddings,
                         index=faiss.IndexFlatL2(dimensions),
                         docstore=InMemoryDocstore(),
                         index_to_docstore_id={})

    def add_documents(self, documents):
        """Add documents to the vector store and save it."""
        self.vectorstore.add_documents(documents)
        self.vectorstore.save_local(self.db_path)
