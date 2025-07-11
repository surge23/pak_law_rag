from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_together import ChatTogether
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

import os

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'templates')

app = Flask(__name__, template_folder=template_dir)
CORS(app)

# Load environment variables
load_dotenv()

# Initialize your existing chatbot components
def initialize_chatbot():
    # Load FAISS vector store
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        "../faiss_index/pak_law_combined",  # This should work since faiss_index is one level up
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    # Use Together.ai's LLaMA 3 model
    llm = ChatTogether(
        model="meta-llama/Llama-3-8b-chat-hf",
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
        retriever=vectorstore.as_retriever(search_type="mmr", k=4),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )
    
    return qa_chain

# Initialize the chatbot
print("🔄 Initializing Pakistani Law Chatbot...")
qa_chain = initialize_chatbot()
print("✅ Chatbot initialized successfully!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Get answer from your existing chatbot
        result = qa_chain.invoke(question)
        
        # Format sources
        sources = []
        for doc in result["source_documents"]:
            sources.append({
                'source': doc.metadata.get('source', 'Unknown'),
                'title': doc.metadata.get('title', 'Unknown')
            })
        
        return jsonify({
            'answer': result["result"],
            'sources': sources
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)