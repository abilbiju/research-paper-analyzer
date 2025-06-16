import os
import json
import uuid
import streamlit as st
from dotenv import load_dotenv
import pycountry
import traceback
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Get the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables from .env file
load_dotenv()

# Check if API key exists
if os.getenv("GEMINI_API_KEY") is not None:
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
else:
    st.error("GEMINI_API_KEY not found in .env file")
    st.stop()

# Import LangChain components
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Configure uploads and storage with absolute paths relative to the script
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"Created uploads directory at: {UPLOAD_FOLDER}")

# Create storage folder
STORAGE_FOLDER = os.path.join(BASE_DIR, 'storage')
if not os.path.exists(STORAGE_FOLDER):
    os.makedirs(STORAGE_FOLDER)
    print(f"Created storage directory at: {STORAGE_FOLDER}")

# Initialize models
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
chat_model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Translation-related
LANGUAGES = sorted([language.name for language in pycountry.languages if hasattr(language, 'name')])

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['pdf', 'txt', 'docx']

# Simple vector store implementation
class SimpleVectorStore:
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function
        self.documents = []
        self.embeddings = []
        
    def add_documents(self, documents):
        """Add documents to the vector store"""
        # Convert document content to embeddings
        texts = [doc.page_content for doc in documents]
        new_embeddings = self.embedding_function.embed_documents(texts)
        
        # Store documents and their embeddings
        self.documents.extend(documents)
        self.embeddings.extend(new_embeddings)
    
    def similarity_search(self, query, k=4):
        """Search for similar documents to the query"""
        # Get query embedding
        query_embedding = self.embedding_function.embed_query(query)
        
        # Convert to numpy for similarity calculation
        query_embedding_np = np.array([query_embedding])
        embeddings_np = np.array(self.embeddings)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding_np, embeddings_np)[0]
        
        # Get top k similar document indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return the top k documents
        return [self.documents[i] for i in top_k_indices]
    
    def save(self, file_path):
        """Save the vector store to disk"""
        data = {
            "documents": [document_to_dict(doc) for doc in self.documents],
            "embeddings": self.embeddings
        }
        with open(file_path, "w") as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, file_path, embedding_function):
        """Load the vector store from disk"""
        with open(file_path, "r") as f:
            data = json.load(f)
        
        vector_store = cls(embedding_function)
        vector_store.embeddings = data["embeddings"]
        vector_store.documents = [
            Document(
                page_content=doc["page_content"],
                metadata=doc["metadata"]
            ) for doc in data["documents"]
        ]
        
        return vector_store
    
    def as_retriever(self, search_kwargs=None):
        """Return a retriever interface compatible with LangChain"""
        if search_kwargs is None:
            search_kwargs = {"k": 4}
            
        # Define a retriever function
        def retrieve_func(query):
            return self.similarity_search(query, k=search_kwargs.get("k", 4))
        
        # Create a simple object with an invoke method
        class SimpleRetriever:
            def __init__(self, retrieve_function):
                self.retrieve = retrieve_function
                
            def invoke(self, query):
                return self.retrieve(query)
            
            def get_relevant_documents(self, query):
                return self.retrieve(query)
                
        return SimpleRetriever(retrieve_func)

# Document handling functions
def document_to_dict(doc):
    """Convert a LangChain Document object to a serializable dictionary."""
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata
    }

def save_document_metadata(file_id, metadata):
    """Save document metadata to disk"""
    metadata_file = os.path.join(STORAGE_FOLDER, f"{file_id}_metadata.json")
    
    # Create a serializable copy of the metadata
    serializable_metadata = {}
    
    # Save document_id - for retrieval
    if "document_id" in metadata:
        serializable_metadata["document_id"] = metadata["document_id"]
    
    # Handle chunks (list of Document objects)
    if "chunks" in metadata:
        serializable_metadata["chunks"] = [document_to_dict(doc) for doc in metadata["chunks"]]
    
    # Handle raw_text
    if "raw_text" in metadata:
        serializable_metadata["raw_text"] = metadata["raw_text"]
    
    # Handle paper_metadata
    if "paper_metadata" in metadata:
        serializable_metadata["paper_metadata"] = metadata["paper_metadata"]
    
    # Print what we're saving for debugging
    print(f"Saving metadata for {file_id} with keys: {list(serializable_metadata.keys())}")
    
    # Save serializable metadata to JSON
    with open(metadata_file, 'w') as f:
        json.dump(serializable_metadata, f)
    
    # Return the original metadata with retriever intact
    return metadata

def load_document_metadata(file_id):
    """Load document metadata from disk"""
    metadata_file = os.path.join(STORAGE_FOLDER, f"{file_id}_metadata.json")
    
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
    
    return metadata

def process_document(file_path):
    """Process the uploaded document"""
    if file_path.lower().endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = UnstructuredFileLoader(file_path)
    
    documents = loader.load()
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create a unique ID for this document
    document_id = str(uuid.uuid4())
    print(f"Generated document ID: {document_id}")
    
    # Create vector store using our simple implementation
    vector_store = SimpleVectorStore(embeddings)
    vector_store.add_documents(chunks)
    
    # Save the vector store
    vector_store_path = os.path.join(STORAGE_FOLDER, f"{document_id}_vectors.json")
    vector_store.save(vector_store_path)
    
    # Extract metadata
    try:
        extracted_metadata = extract_paper_metadata(documents)
    except Exception as e:
        print(f"Error extracting metadata: {str(e)}")
        extracted_metadata = {"error": "Failed to extract metadata"}
    
    result = {
        "chunks": chunks,
        "document_id": document_id,
        "raw_text": "\n\n".join([doc.page_content for doc in documents]),
        "paper_metadata": extracted_metadata,
        "vector_store": vector_store  # Include the vector store in the result
    }
    
    print(f"Document processed with {len(chunks)} chunks, document ID: {document_id}")
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
    # Get relevant documents
    documents = retriever.get_relevant_documents(question)
    
    # Create a context from the retrieved documents
    context = "\n\n".join([doc.page_content for doc in documents])
    
    qa_prompt = ChatPromptTemplate.from_template("""
    You are a helpful research assistant. Answer the question based only on the following context:
    
    Context:
    {context}
    
    Question: {question}
    
    If you don't have enough information to answer this question based on the context, say "I don't have enough information to answer this question."
    """)
    
    qa_chain = qa_prompt | chat_model | StrOutputParser()
    
    return {
        "answer": qa_chain.invoke({"context": context, "question": question}),
        "source_documents": documents
    }

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
        return json.loads(result)
    except:
        return {"error": "Failed to parse metadata", "raw_result": result}

# Streamlit UI
st.set_page_config(
    page_title="PDF Analyzer",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'file_id' not in st.session_state:
    st.session_state.file_id = None
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'insights' not in st.session_state:
    st.session_state.insights = None
if 'answers' not in st.session_state:
    st.session_state.answers = []
if 'paper_metadata' not in st.session_state:
    st.session_state.paper_metadata = None

# App title
st.title("📄 PDF Analyzer")
st.markdown("Upload a PDF document to analyze and ask questions about it.")

# Sidebar for file upload and settings
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf', 'txt', 'docx'])
    
    if uploaded_file is not None:
        # Create a unique file ID
        if st.session_state.file_id is None or st.session_state.filename != uploaded_file.name:
            file_id = f"doc_{uuid.uuid4()}"
            st.session_state.file_id = file_id
            st.session_state.filename = uploaded_file.name
            
            # Save the uploaded file temporarily
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process the document with progress bar
            with st.spinner("Processing document..."):
                try:
                    document_data = process_document(file_path)
                    save_document_metadata(file_id, document_data)
                    
                    # Generate summary
                    st.session_state.summary = generate_summary(document_data["chunks"])
                    
                    # Generate insights
                    st.session_state.insights = generate_insights(st.session_state.summary)
                    
                    # Store paper metadata
                    if "paper_metadata" in document_data:
                        st.session_state.paper_metadata = document_data["paper_metadata"]
                    
                    st.success(f"Document processed: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                    st.code(traceback.format_exc())
    
    # Translation section
    st.header("Translation")
    if st.session_state.summary:
        source_language = "English"
        target_language = st.selectbox("Select target language", LANGUAGES)
        text_to_translate = st.text_area("Text to translate", 
                                         value=st.session_state.summary[:500] + "..." 
                                         if len(st.session_state.summary) > 500 
                                         else st.session_state.summary)
        
        if st.button("Translate"):
            with st.spinner("Translating..."):
                try:
                    translation = translate_text(text_to_translate, target_language, source_language)
                    st.markdown("### Translation")
                    st.markdown(translation)
                except Exception as e:
                    st.error(f"Translation error: {str(e)}")

# Main content area
if st.session_state.file_id and st.session_state.summary:
    
    # Add tabs for different views
    tab1, tab2, tab3 = st.tabs(["Summary", "Ask Questions", "Paper Metadata"])
    
    # Tab 1: Summary and insights
    with tab1:
        st.header("Document Summary")
        st.markdown(st.session_state.summary)
        
        st.header("Document Insights")
        st.markdown(st.session_state.insights)
    
    # Tab 2: Q&A
    with tab2:
        st.header("Ask Questions About the Document")
        question = st.text_input("Enter your question:")
        
        if st.button("Ask"):
            if question:
                with st.spinner("Finding answer..."):
                    try:
                        # Load document metadata
                        document_data = load_document_metadata(st.session_state.file_id)
                        
                        if document_data and "document_id" in document_data:
                            document_id = document_data["document_id"]
                            vector_store_path = os.path.join(STORAGE_FOLDER, f"{document_id}_vectors.json")
                            
                            # Load the vector store
                            vector_store = SimpleVectorStore.load(vector_store_path, embeddings)
                            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
                            
                            # Get answer
                            response = answer_question(question, retriever)
                            answer = response["answer"]
                            
                            # Store in session state
                            st.session_state.answers.append({"question": question, "answer": answer})
                        else:
                            st.error("Failed to load document data. Please try reuploading the document.")
                    except Exception as e:
                        st.error(f"Error answering question: {str(e)}")
                        st.code(traceback.format_exc())
            else:
                st.warning("Please enter a question.")
        
        # Display previous Q&A
        if st.session_state.answers:
            st.subheader("Previous Questions & Answers")
            for qa in st.session_state.answers:
                with st.expander(f"Q: {qa['question']}"):
                    st.markdown(qa['answer'])
    
    # Tab 3: Paper Metadata
    with tab3:
        st.header("Paper Metadata")
        if st.session_state.paper_metadata:
            metadata = st.session_state.paper_metadata
            if "error" in metadata:
                st.error("Could not extract metadata properly")
                st.json(metadata)
            else:
                col1, col2 = st.columns(2)
                with col1:
                    if "Title" in metadata:
                        st.markdown(f"**Title:** {metadata.get('Title', 'N/A')}")
                    if "Authors" in metadata:
                        st.markdown(f"**Authors:** {metadata.get('Authors', 'N/A')}")
                    if "Publication year" in metadata:
                        st.markdown(f"**Year:** {metadata.get('Publication year', 'N/A')}")
                with col2:
                    if "Journal/Conference" in metadata:
                        st.markdown(f"**Journal/Conference:** {metadata.get('Journal/Conference', 'N/A')}")
                    if "DOI/identifier" in metadata:
                        st.markdown(f"**DOI:** {metadata.get('DOI/identifier', 'N/A')}")
        else:
            st.info("No metadata available for this document.")
else:
    st.info("Please upload a document to get started.")
