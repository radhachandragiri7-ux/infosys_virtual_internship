from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Connect to the existing database we created
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./vector_storage", embedding_function=embeddings)

# 2. Ask a question about the PDF
query = "What is Cloud Computing?" # Or any topic inside your PDF

# 3. Search for the most relevant chunks
print(f"\nSearching for: {query}")
results = vector_db.similarity_search(query, k=2)

# 4. Print the results
print("\n--- Found relevant text in the PDF ---")
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(doc.page_content[:300] + "...") # Show first 300 characters