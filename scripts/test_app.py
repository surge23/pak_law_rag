from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_together import ChatTogether
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()

# Initialize your existing chatbot components
def initialize_chatbot():
    # Load FAISS vector store
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        "../faiss_index/pak_law_combined",  # Adjust this path if needed
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
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pakistani Law Assistant 🇵🇰</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #28a745; text-align: center; }
            .chat-box { height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 15px; margin: 20px 0; background: #fafafa; }
            .input-group { display: flex; gap: 10px; }
            input[type="text"] { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            button { padding: 10px 20px; background: #28a745; color: white; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #218838; }
            .user-msg { text-align: right; margin: 10px 0; }
            .bot-msg { text-align: left; margin: 10px 0; padding: 10px; background: #e9ecef; border-radius: 5px; }
            .sources { margin-top: 10px; font-size: 12px; color: #666; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🇵🇰 Pakistani Law Assistant</h1>
            <p>Ask questions about the Constitution of Pakistan, PPC, and CrPC</p>
            <div class="chat-box" id="chatBox">
                <div class="bot-msg">
                    <strong>Welcome!</strong> Ask any question about Pakistani law. 
                    <br><strong>Example:</strong> "What are the fundamental rights in the Constitution?"
                </div>
            </div>
            <div class="input-group">
                <input type="text" id="questionInput" placeholder="Ask your legal question here..." />
                <button onclick="askQuestion()">Send</button>
            </div>
        </div>

        <script>
            function askQuestion() {
                const input = document.getElementById('questionInput');
                const chatBox = document.getElementById('chatBox');
                const question = input.value.trim();
                
                if (!question) return;
                
                // Add user message
                chatBox.innerHTML += `<div class="user-msg"><strong>You:</strong> ${question}</div>`;
                
                // Clear input
                input.value = '';
                
                // Show loading
                chatBox.innerHTML += `<div class="bot-msg"><strong>Assistant:</strong> <em>Getting legal information...</em></div>`;
                chatBox.scrollTop = chatBox.scrollHeight;
                
                // Send request
                fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question })
                })
                .then(response => response.json())
                .then(data => {
                    // Remove loading message
                    chatBox.removeChild(chatBox.lastElementChild);
                    
                    if (data.answer) {
                        let sourcesHtml = '';
                        if (data.sources && data.sources.length > 0) {
                            sourcesHtml = '<div class="sources"><strong>Sources:</strong><br>' + 
                                data.sources.map(s => `• ${s.source}: ${s.title}`).join('<br>') + '</div>';
                        }
                        
                        chatBox.innerHTML += `<div class="bot-msg"><strong>Assistant:</strong> ${data.answer.replace(/\\n/g, '<br>')}${sourcesHtml}</div>`;
                    } else {
                        chatBox.innerHTML += `<div class="bot-msg"><strong>Error:</strong> ${data.error || 'Something went wrong'}</div>`;
                    }
                    
                    chatBox.scrollTop = chatBox.scrollHeight;
                })
                .catch(error => {
                    chatBox.removeChild(chatBox.lastElementChild);
                    chatBox.innerHTML += `<div class="bot-msg"><strong>Error:</strong> Unable to connect to server</div>`;
                    chatBox.scrollTop = chatBox.scrollHeight;
                });
            }
            
            // Allow Enter key to send
            document.getElementById('questionInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    askQuestion();
                }
            });
        </script>
    </body>
    </html>
    '''

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)