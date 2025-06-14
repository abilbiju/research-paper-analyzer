import os
import json
import pickle
import tempfile
from flask import Flask, request, render_template, jsonify, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import pycountry

# Get the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables from .env file
load_dotenv()

# Check if API key exists
if os.getenv("GEMINI_API_KEY") is not None:
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
else:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Import LangChain components
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# Update the import for HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

# Configure uploads and storage with absolute paths relative to the script
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"Created uploads directory at: {UPLOAD_FOLDER}")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'docx'}

# Create storage folder
STORAGE_FOLDER = os.path.join(BASE_DIR, 'storage')
if not os.path.exists(STORAGE_FOLDER):
    os.makedirs(STORAGE_FOLDER)
    print(f"Created storage directory at: {STORAGE_FOLDER}")

# Initialize models
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
chat_model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

# Function to save document metadata to disk
def save_document_metadata(file_id, metadata):
    """Save document metadata to disk"""
    metadata_file = os.path.join(STORAGE_FOLDER, f"{file_id}_metadata.json")
    
    # Create a serializable copy of the metadata
    serializable_metadata = {}
    
    # Save collection_name - this is critical for retrieval
    if "collection_name" in metadata:
        serializable_metadata["collection_name"] = metadata["collection_name"]
    else:
        print(f"WARNING: No collection_name found in metadata for {file_id}")
    
    # Handle chunks (list of Document objects)
    if "chunks" in metadata:
        serializable_metadata["chunks"] = [document_to_dict(doc) for doc in metadata["chunks"]]
    
    # Handle raw_text
    if "raw_text" in metadata:
        serializable_metadata["raw_text"] = metadata["raw_text"]
    
    # Print what we're saving for debugging
    print(f"Saving metadata for {file_id} with keys: {list(serializable_metadata.keys())}")
    
    # Save serializable metadata to JSON
    with open(metadata_file, 'w') as f:
        json.dump(serializable_metadata, f)
    
    # Return the original metadata with retriever intact
    return metadata

# Add this function to convert Document objects to serializable dictionaries
def document_to_dict(doc):
    """Convert a LangChain Document object to a serializable dictionary."""
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata
    }

# Function to load document metadata from disk
def load_document_metadata(file_id):
    """Load document metadata from disk"""
    from langchain_core.documents import Document
    
    metadata_file = os.path.join(STORAGE_FOLDER, f"{file_id}_metadata.json")
    retriever_file = os.path.join(STORAGE_FOLDER, f"{file_id}_retriever.pkl")
    
    if not os.path.exists(metadata_file):
        return None
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Convert serialized chunks back to Document objects if they exist
    if "chunks" in metadata:
        metadata["chunks"] = [
            Document(
                page_content=chunk["page_content"],
                metadata=chunk["metadata"]
            ) for chunk in metadata["chunks"]
        ]
    
    # Load retriever if it exists
    if os.path.exists(retriever_file):
        with open(retriever_file, 'rb') as f:
            metadata["retriever"] = pickle.load(f)
    
    return metadata

# In-memory storage for document metadata index
document_index = {}

# Initialize embeddings with updated class
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Translation-related
LANGUAGES = sorted([language.name for language in pycountry.languages if hasattr(language, 'name')])

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Update process_document to remove the unnecessary persist() call

def process_document(file_path):
    """Process the uploaded document"""
    if file_path.lower().endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = UnstructuredFileLoader(file_path)
    
    documents = loader.load()
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Increase chunk size
        chunk_overlap=200,  # Increase overlap
        separators=["\n\n", "\n", " ", ""]  # Prioritize paragraph breaks
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create a unique ID for this document
    import uuid
    collection_name = str(uuid.uuid4())
    print(f"Generated collection name: {collection_name}")
    
    # Create persistent vector store in the storage folder
    persist_directory = os.path.join(STORAGE_FOLDER, collection_name)
    os.makedirs(persist_directory, exist_ok=True)
    
    # Create vector store - documents are automatically persisted
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # Extract metadata using the new function
    extracted_metadata = extract_paper_metadata(documents)
    
    result = {
        "chunks": chunks,
        "collection_name": collection_name,
        "raw_text": "\n\n".join([doc.page_content for doc in documents]),
        "paper_metadata": extracted_metadata  # Add extracted metadata to result
    }
    
    print(f"Document processed with {len(chunks)} chunks, collection name: {collection_name}")
    return result

def generate_summary(chunks):
    """Generate a summary from document chunks"""
    summary_prompt = ChatPromptTemplate.from_template("""
    You are a helpful research assistant that creates concise yet comprehensive summaries of academic papers.
    
    Please summarize the following research paper chunks:
    
    {text}
    
    Provide the summary in the following format:
    
    ## Title (if identified)
    ## Abstract
    ## Key Findings
    ## Methodology
    ## Conclusions
    ## Limitations (if any)
    
    Focus on the most important information and ensure the summary is coherent.
    """)
    
    # Create chain
    chain = summary_prompt | chat_model | StrOutputParser()
    
    # Join the first few chunks to avoid token limits
    combined_text = "\n\n".join([chunk.page_content for chunk in chunks[:10]])
    
    return chain.invoke({"text": combined_text})

def generate_insights(summary):
    """Generate insights based on the summary"""
    insight_prompt = ChatPromptTemplate.from_template("""
    Based on the following research paper summary, please identify:
    1. The key innovations or contributions
    2. Potential applications or implications
    3. Connections to other research areas
    4. Future research directions
    
    Research summary:
    {summary}
    
    Provide your analysis in a structured format.
    """)
    
    chain = insight_prompt | chat_model | StrOutputParser()
    return chain.invoke({"summary": summary})

def answer_question(question, retriever):
    """Answer a question using retrieval QA"""
    qa_prompt = ChatPromptTemplate.from_template("""
    You are a helpful research assistant. Answer the question based only on the following context:
    
    {context}
    
    Question: {input}
    
    If you don't have enough information to answer this question based on the context, say "I don't have enough information to answer this question."
    """)
    
    # Change the prompt to use {input} instead of {question} to match what's passed by the retrieval chain
    qa_chain = create_stuff_documents_chain(chat_model, qa_prompt)
    retrieval_chain = create_retrieval_chain(retriever, qa_chain)
    
    # This remains the same - passing the question as "input"
    return retrieval_chain.invoke({"input": question})

def translate_text(text, target_language, source_language="English"):
    """Translate text to target language"""
    translation_prompt = ChatPromptTemplate.from_messages([
        ("system", "Translate the following text from {source_language} to {target_language}: "),
        ("user", "{text}")
    ])
    
    translation_chain = translation_prompt | model | StrOutputParser()
    
    return translation_chain.invoke({
        "text": text,
        "source_language": source_language,
        "target_language": target_language
    })

def extract_paper_metadata(documents):
    """Extract basic metadata from document"""
    # Combine first 2000 characters to capture header information
    header_text = "".join([doc.page_content for doc in documents])[:2000]
    
    metadata_prompt = ChatPromptTemplate.from_template("""
    Extract the following metadata from this academic paper header, if present:
    - Title
    - Authors (full names)
    - Publication year
    - Journal/Conference
    - DOI/identifier
    
    Paper header text:
    {text}
    
    Respond in JSON format with these fields. Use null for missing information.
    """)
    
    metadata_chain = metadata_prompt | chat_model | StrOutputParser()
    result = metadata_chain.invoke({"text": header_text})
    
    # Try to parse as JSON but handle errors gracefully
    try:
        import json
        return json.loads(result)
    except:
        return {"error": "Failed to parse metadata", "raw_result": result}

# Routes
@app.route('/')
def home():
    return render_template('home.html', languages=LANGUAGES)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not supported"}), 400
    
    # Get file extension
    original_filename = secure_filename(file.filename)
    extension = original_filename.rsplit('.', 1)[1].lower()
    
    # Always save as sample.{extension}
    filename = f"sample.{extension}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Remove existing file if it exists
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # Save file
    file.save(file_path)
    
    # Generate unique ID
    file_id = f"doc_{len(document_index) + 1}"
    
    try:
        # Process document
        print(f"Processing document (saved as {filename}), file_id: {file_id}")
        document_data = process_document(file_path)
        
        # Store the document data in memory with the original filename for display purposes
        document_index[file_id] = {
            "filename": original_filename,  # Keep the original filename for display
            "path": file_path,
            "saved_as": filename           # Track the actual saved filename
        }
        
        # Save to disk
        save_document_metadata(file_id, document_data)
        print(f"Document stored with ID {file_id}. Available IDs: {list(document_index.keys())}")
        
        # Generate summary
        summary = generate_summary(document_data["chunks"])
        
        # Generate insights
        insights = generate_insights(summary)
        
        return jsonify({
            "file_id": file_id,
            "filename": original_filename,  # Return original filename to display to user
            "summary": summary,
            "insights": insights
        })
    except Exception as e:
        import traceback
        print(f"Error processing document: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Update ask endpoint for better error handling

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    file_id = data.get('file_id')
    question = data.get('question')
    
    print(f"Received question request: file_id={file_id}, question={question}")
    print(f"Available documents in index: {list(document_index.keys())}")
    
    # Check if file_id exists in document_index
    if not file_id or file_id not in document_index:
        print(f"File ID not in index: {file_id}")
        return jsonify({"error": f"Invalid file ID: {file_id}"}), 400
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    try:
        # Load document metadata from disk
        document_data = load_document_metadata(file_id)
        
        # Add detailed debug information
        if document_data is None:
            print(f"ERROR: Document data is None for file ID: {file_id}")
            return jsonify({"error": "Failed to load document metadata"}), 500
        
        print(f"Document data keys: {list(document_data.keys())}")
        
        if "collection_name" not in document_data:
            print(f"ERROR: No collection_name in document data for file ID: {file_id}")
            return jsonify({"error": "Document data missing collection ID"}), 500
        
        # Get collection name
        collection_name = document_data["collection_name"]
        persist_directory = os.path.join(STORAGE_FOLDER, collection_name)
        
        # Check if directory exists
        if not os.path.exists(persist_directory):
            print(f"ERROR: Vector store directory not found: {persist_directory}")
            return jsonify({"error": "Vector database not found"}), 500
        
        # Load the persisted vector store
        try:
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            print(f"Successfully loaded vector store from: {persist_directory}")
        except Exception as ve:
            print(f"ERROR loading vector store: {ve}")
            return jsonify({"error": f"Failed to load vector database: {str(ve)}"}), 500
        
        print(f"Processing question for file {file_id}: {question}")
        response = answer_question(question, retriever)
        print("Answer generated successfully")
        return jsonify({"answer": response["answer"]})
    except Exception as e:
        import traceback
        print(f"Error answering question: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/translate', methods=['POST'])
def translate():
    """Translate text to target language"""
    data = request.get_json()
    text = data.get('text')
    target_language = data.get('target_language')
    source_language = data.get('source_language', 'English')
    
    if not text or not target_language:
        return jsonify({"error": "Text and target language are required"}), 400
    
    try:
        translation = translate_text(text, target_language, source_language)
        print(f"Translation completed to {target_language}")
        return jsonify({"translation": translation})
    except Exception as e:
        print(f"Translation error: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
