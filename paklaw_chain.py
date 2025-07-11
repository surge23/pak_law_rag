from langchain_community.chat_models import ChatTogether
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# === Load FAISS vectorstore ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index/pak_law_combined", embedding_model, allow_dangerous_deserialization=True)

# === Load Together LLaMA-3 LLM ===
llm = ChatTogether(
    model="meta-llama/Llama-3-8b-chat-hf",
    temperature=0.2
)

# === Build RetrievalQA Chain ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_type="mmr", k=3),
    return_source_documents=True
)
