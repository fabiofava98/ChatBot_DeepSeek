from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.chains import RetrievalQA
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.docstore.in_memory import InMemoryDocstore

from unstructured.partition.pdf import partition_pdf

from typing import cast
import logging
import psutil
import os
import chainlit as cl
import faiss
import numpy as np

class RAGPipeline:
    def __init__(self, model_name: str = "llama2:7b-chat-q4", max_memory_gb: float = 3.0, db_path: str = ""):
        self.setup_logging()
        self.check_system_memory(max_memory_gb)
        
        self.db_path = db_path
        
        # Load the language model (LLM)
        self.llm = OllamaLLM(model=model_name,
                            temperature=0.5,  # Lower creativity for concise responses
                            top_p=0.9,       # Adjust diversity slightly
                            max_tokens=250    # Limit response length
                            )  
        
        # Initialize embeddings using a lightweight model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}  # Use CPU for efficiency
        )
        
        # Load the vector store (if it exists)
        self.vectorstore = self.load_vectorstore()
        
        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_template(""" 
        Answer the question based on the following context. Be concise.
        Reason about the context to provide a well-thought-out answer.
        If you cannot find the answer in the context, use your general knowledge to provide an answer.
                
        Context: {context}
        Question: {question}
        Answer: """)

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def check_system_memory(self, max_memory_gb: float):
        available_memory = psutil.virtual_memory().available / (1024 ** 3)
        self.logger.info(f"Available system memory: {available_memory:.1f} GB")
        if available_memory < max_memory_gb:
            self.logger.warning("Memory is below recommended threshold.")
    
    def load_vectorstore(self) -> FAISS:
        """Load the existing vector store or create a new one if it doesn't exist."""
        if os.path.exists(self.db_path):
            self.logger.info(f"Loading existing vector store from {self.db_path}")
            return FAISS.load_local(self.db_path, self.embeddings)
        else:
            self.logger.info("No existing vector store found. Creating a new one.")
            dimensions: int = len(self.embeddings.embed_query("dummy"))
            db = FAISS(
                embedding_function=self.embeddings,
                index=faiss.IndexFlatL2(dimensions),
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
                normalize_L2=False,
                )
            return db

    def load_and_split_documents(self, file_path: str) -> List[Document]:
        """Load and split documents based on the file type."""
        if file_path.endswith('.csv'):
            loader = CSVLoader(file_path=file_path, csv_args={'delimiter': ';'})
            documents = loader.load()        
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path=file_path)
            documents = loader.load()
        elif file_path.endswith('.pdf'):
            #loader = PyPDFLoader(file_path)
            elements = partition_pdf(file_path)
            raw_text = "\n".join([el.text for el in elements if el.text])
            documents = [Document(page_content=raw_text)]
        else:
            raise ValueError(f"Unsupported file type: {file_path}. Please provide .csv, .txt, or .pdf files.")
        
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            add_start_index=True,
        )
        splits = text_splitter.split_documents(documents)
        self.logger.info(f"Created {len(splits)} document chunks from {file_path}")
        return splits

    def add_documents_to_vectorstore(self, documents: List[Document]):
        """Add new documents to the existing vector store."""
        batch_size = 32
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self.vectorstore.add_documents(batch)
            self.logger.info(f"Processed batch {i//batch_size + 1}")
        
        # Save the updated vector store
        if self.db_path != None:
            self.vectorstore.save_local(self.db_path)
        self.logger.info("Vector store updated and saved.")
        
    def process_files_in_folder(self, folder_path: str):
        """Process all files in a folder and add them to the vector store."""
        all_documents = []
        
        # List all files in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            
            # Skip directories, only process files
            if os.path.isdir(file_path):
                continue
            
            # Check for supported file extensions and process accordingly
            try:
                documents = self.load_and_split_documents(file_path)
                all_documents.extend(documents)
            except ValueError as e:
                self.logger.warning(f"Skipping unsupported file {file_name}: {e}")
        
        # After processing all files, add them to the vectorstore
        if all_documents:
            self.add_documents_to_vectorstore(all_documents)
        
    def retrieval_qa_chain(self):
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 3}),  # Retrieve the most relevant document
            return_source_documents=True,
            chain_type_kwargs={'prompt': self.prompt}
        )
        return qa_chain 

    def add_new_folder_to_knowledge(self, folder_path: str):
        """Method to add a new folder of documents to the knowledge base."""
        self.process_files_in_folder(folder_path)


def qa_bot():
    rag = RAGPipeline(model_name="deepseek-r1:8b", max_memory_gb=3.0, db_path="")
    #rag = RAGPipeline(model_name="deepseek-r1:1.5b", max_memory_gb=3.0, db_path="vectorstore/db_faiss")
    rag.add_new_folder_to_knowledge("rag_files")
    qa = rag.retrieval_qa_chain()
    
    return qa
    
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to the DeepSeek ChatBot. How can I assist you today?"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("chain"))  # Get the 'chain' from the user session

    if not runnable:
        await cl.Message(content="Error: The runnable chain is not initialized.").send()
        return

    msg = cl.Message(content="")

    # Pass 'query' as the expected key
    async for chunk in runnable.astream(
        {"query": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        # Assuming 'chunk' is a dictionary, extract the text content
        chunk_text = chunk.get("result", "")  # Update this if the key is different
        await msg.stream_token(chunk_text)

    await msg.send()


