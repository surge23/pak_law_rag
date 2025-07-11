import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Load the scraped constitution
with open("../data/constitution_clean.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

# Prepare LangChain Document objects
documents = []
for article in articles:
    content = article["text"]
    metadata = {"title": article["title"]}
    documents.append(Document(page_content=content, metadata=metadata))

# Initialize embedding model
print("🔄 Generating embeddings...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Generate FAISS vector store
vectorstore = FAISS.from_documents(documents, embedding_model)
vectorstore.save_local("../faiss_index/constitution")

print("✅ Vector store created and saved to: faiss_index/constitution")
