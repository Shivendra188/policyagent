
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Load env variables
load_dotenv()

def index_document():
    PDF_PATH = "./policy.pdf"   # <-- another PDF (policy)

    # 1️⃣ Load PDF
    loader = PyPDFLoader(PDF_PATH)
    raw_docs = loader.load()
    print("PDF loaded")

    # 2️⃣ Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunked_docs = text_splitter.split_documents(raw_docs)
    print("Chunking completed")

    # 3️⃣ Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    print("Embedding model configured")

    # 4️⃣ Pinecone setup
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")

    vectorstore = PineconeVectorStore.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        index_name=index_name
    )

    print("Data stored successfully")

if __name__ == "__main__":
    index_document()
