from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from unstructured.partition.pdf import partition_pdf
import os

def load_and_split_documents(file_path: str) -> List[Document]:
    """Load and split documents based on the file type."""
    
    if file_path.endswith('.csv'):
        loader = CSVLoader(file_path=file_path, csv_args={'delimiter': ';'})
        documents = loader.load()        
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path=file_path)
        documents = loader.load()
    elif file_path.endswith('.pdf'):
        elements = partition_pdf(file_path)
        raw_text = "\n".join([el.text for el in elements if el.text])
        documents = [Document(page_content=raw_text)]
    else:
        raise ValueError(f"Unsupported file type: {file_path}. Please provide .csv, .txt, or .pdf files.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)
    
def process_files_in_folder(folder_path: str) -> List[Document]:
    """Process all files in a folder and return a list of document chunks."""
    all_documents = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if os.path.isdir(file_path):
            continue  # Skip directories

        try:
            documents = load_and_split_documents(file_path)
            all_documents.extend(documents)
        except ValueError as e:
            print(f"Skipping unsupported file {file_name}: {e}")

    return all_documents