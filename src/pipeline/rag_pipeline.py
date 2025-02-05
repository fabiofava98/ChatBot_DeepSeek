from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from utils.vector_store import VectorStoreManager
from utils.file_processing import process_files_in_folder
from config.settings import MODEL_NAME, DB_PATH

class RAGPipeline:
    def __init__(self):
        self.llm = OllamaLLM(model=MODEL_NAME, temperature=0.5, top_p=0.9, max_tokens=250)
        self.vectorstore_manager = VectorStoreManager(DB_PATH)
        
        self.prompt = ChatPromptTemplate.from_template(""" 
        Answer the question based on the following context. Be concise.
        Context: {context}
        Question: {question}
        Answer: """)

    def add_documents_from_folder(self, folder_path):
        documents = process_files_in_folder(folder_path)
        self.vectorstore_manager.add_documents(documents)

    def get_qa_chain(self):
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore_manager.vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': self.prompt}
        )
