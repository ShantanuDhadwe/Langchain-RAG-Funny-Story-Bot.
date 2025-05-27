from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


curr_dir = os.getcwd()
books_dir = os.path.join(curr_dir,"documents")
db_dir = os.path.join(curr_dir,"db")
persistent_dir = os.path.join(db_dir,"chroma_db")

if not os.path.exists(persistent_dir):
    print("persistent_dir does not exist. Initializng vector store...")
    
    book_files = [i for i in os.listdir(books_dir)]
    print(book_files)
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir,book_file)
        loader = PyPDFLoader(file_path)
        book_docs = loader.load()
        print(book_docs[0])
        for doc in book_docs:
            documents.append(doc)
    text_spilter = RecursiveCharacterTextSplitter(chunk_size = 1000,
                                        chunk_overlap =150,
                                         separators=["\\n\\n", "\\n", " ", ""])
                                         
    docs = text_spilter.split_documents(documents)
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print("\n--- Creating embeddings ---")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    # Create the vector store and persist it
    db = Chroma.from_documents(
        docs, embeddings, persist_directory= persistent_dir)
    print("\n--- Finished creating and persisting vector store ---")
    
else:
    print("Vector store already exists. No need to initialize.")