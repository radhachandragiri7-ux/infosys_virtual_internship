from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. LOAD: Read the PDF
print("Step 1: Loading PDF...")
loader = PyPDFLoader("data.pdf")
pages = loader.load()

# 2. SPLIT: Break the text into smaller pieces (Chunks)
# This is crucial so the AI doesn't get overwhelmed by too much text at once.
print("Step 2: Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=100
)
chunks = text_splitter.split_documents(pages)

# 3. EMBED & STORE: Convert text to math and save to a Database
print("Step 3: Generating embeddings and saving to Vector DB (Chroma)...")
# We use a free model from HuggingFace to create the numerical 'vectors'
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# This creates a local folder called 'vector_storage' to keep your data
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./vector_storage"
)

print("\n✅ Milestone 1 Successful!")
print(f"Total chunks created and indexed: {len(chunks)}")
print("Your database is now saved in the 'vector_storage' folder.")
