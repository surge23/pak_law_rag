# from langchain_community.chat_models import ChatTogether
from langchain_together import ChatTogether
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from dotenv import load_dotenv
# import os

load_dotenv()  # Load TOGETHER_API_KEY from .env

# Load FAISS vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "../faiss_index/pak_law_combined",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)


# Use Together.ai's LLaMA 3 model
llm = ChatTogether(
    model="meta-llama/Llama-3-8b-chat-hf",  # or "meta-llama/Llama-3-70b-chat-hf"
    temperature=0.2
)

# Prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a Pakistani law assistant. Only answer based on the Constitution of Pakistan.

Context:
{context}

Question: {question}

Answer in formal legal English with references. Do not guess or make up facts.
"""
)

# RAG QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    # retriever=vectorstore.as_retriever(search_type="similarity", k=4),
    retriever=vectorstore.as_retriever(search_type="mmr", k=4),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)

# Console interface
def chat():
    print("🇵🇰 Ask any question about the Constitution of Pakistan (type 'exit' to quit)\n")
    while True:
        query = input("🧑‍⚖️ Question: ")
        if query.lower() in ['exit', 'quit']:
            break

        result = qa_chain.invoke(query)
        print("\n📘 Answer:\n", result["result"], "\n")
        print("📚 Sources:")
        for doc in result["source_documents"]:
            print(f"– {doc.metadata['source']}: {doc.metadata['title']}")

        print("-" * 60)

if __name__ == "__main__":
    chat()
