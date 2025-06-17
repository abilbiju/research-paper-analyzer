import os
import json
import uuid
import streamlit as st
from dotenv import load_dotenv
import pycountry
import traceback
from PIL import Image
import fitz  # PyMuPDF
import google.generativeai as genai

# Get the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables from .env file
load_dotenv()

# Check if API key exists
if os.getenv("GEMINI_API_KEY") is not None:
    api_key = os.getenv("GEMINI_API_KEY")
    os.environ["GOOGLE_API_KEY"] = api_key
    # Configure Google Generative AI with the same API key
    genai.configure(api_key=api_key)
else:
    st.error("GEMINI_API_KEY not found in .env file")
    st.stop()

# Import LangChain components
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
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
    try:
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
        
        # Extract metadata using the new function - wrapped in try-except
        try:
            extracted_metadata = extract_paper_metadata(documents)
        except Exception as e:
            print(f"Error extracting metadata: {str(e)}")
            extracted_metadata = {"error": str(e)}
        
        result = {
            "chunks": chunks,
            "collection_name": collection_name,
            "raw_text": "\n\n".join([doc.page_content for doc in documents]),
            "paper_metadata": extracted_metadata
        }
        
        print(f"Document processed with {len(chunks)} chunks, collection name: {collection_name}")
        return result
    except Exception as e:
        print(f"Error in process_document: {str(e)}")
        raise

def generate_summary(chunks):
    """Generate a summary from document chunks using direct Gemini API"""
    # Join the first few chunks to avoid token limits
    combined_text = "\n\n".join([chunk.page_content for chunk in chunks[:10]])
    
    try:
        # Use direct Gemini API instead of LangChain wrapper
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(
            """
            You are a helpful research assistant that creates concise yet comprehensive summaries of academic papers.
            
            Please summarize the following research paper chunks:
            
            """ + combined_text + """
            
            Provide the summary in the following format:
            
            ## Title (if identified)
            ## Abstract
            ## Key Findings
            ## Methodology
            ## Conclusions
            ## Limitations (if any)
            
            Focus on the most important information and ensure the summary is coherent.
            """
        )
        
        # Return the generated text
        return response.text
    except Exception as e:
        print(f"Error generating summary: {e}")
        return f"Failed to generate summary: {str(e)}"

def generate_insights(summary):
    """Generate insights based on the summary using direct Gemini API"""
    try:
        # Use direct Gemini API instead of LangChain wrapper
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(
            """
            Based on the following research paper summary, please identify:
            1. The key innovations or contributions
            2. Potential applications or implications
            3. Connections to other research areas
            4. Future research directions
            
            Research summary:
            """ + summary + """
            
            Provide your analysis in a structured format.
            """
        )
        
        # Return the generated text
        return response.text
    except Exception as e:
        print(f"Error generating insights: {e}")
        return f"Failed to generate insights: {str(e)}"

def answer_question(question, retriever):
    """Answer a question using retrieval QA"""
    qa_prompt = ChatPromptTemplate.from_template("""
    You are a helpful research assistant. Answer the question based only on the following context:
    
    {context}
    
    Question: {input}
    
    If you don't have enough information to answer this question based on the context, say "I don't have enough information to answer this question."
    """)
    
    qa_chain = create_stuff_documents_chain(chat_model, qa_prompt)
    retrieval_chain = create_retrieval_chain(retriever, qa_chain)
    
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
    
    try:
        # Use direct Gemini API instead of LangChain wrapper for this function
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(
            f"""
            Extract the following metadata from this academic paper header, if present:
            - Title
            - Authors (full names)
            - Publication year
            - Journal/Conference
            - DOI/identifier
            
            Paper header text:
            {header_text}
            
            Respond in JSON format with these fields. Use null for missing information.
            """
        )
        
        # Extract text from response
        result = response.text
        
        # Try to parse as JSON but handle errors gracefully
        try:
            # Clean the result to handle potential markdown code blocks
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()
                
            return json.loads(result)
        except:
            return {"error": "Failed to parse metadata", "raw_result": result}
            
    except Exception as e:
        return {"error": f"Failed to extract metadata: {str(e)}"}

def process_document_visually(file_path):
    """Process document at the page level maintaining visual structure"""
    try:
        doc = fitz.open(file_path)
        pages = []
        
        for page_num in range(len(doc)):
            # Load the page
            page_obj = doc.load_page(page_num)
            
            # Extract text with position information
            text_blocks = page_obj.get_text("blocks")  # Gets text in reading order with coordinates
            
            # Extract images if present
            images = page_obj.get_images(full=True)
            
            # Render page to image
            pix = page_obj.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better resolution
            img_data = pix.samples
            page_image = Image.frombytes("RGB", [pix.width, pix.height], img_data)
            
            # Analyze page layout
            layout = analyze_page_layout(page_obj)
            
            # Store page data
            pages.append({
                "page_num": page_num,
                "text_blocks": text_blocks,
                "images": images,
                "layout": layout,
                "page_image": page_image,
                "page_obj": page_obj  # Store reference to page object
            })
        
        # Don't close the document here - we need it for further processing
        # We'll need to explicitly close it later
        
        return pages, doc
    except Exception as e:
        print(f"Error in document visual processing: {e}")
        return [], None

def analyze_page_layout(page):
    """Analyze the layout structure of a PDF page"""
    # Get page dimensions
    page_rect = page.rect
    width, height = page_rect.width, page_rect.height
    
    # Extract text blocks with their positions
    blocks = page.get_text("dict")["blocks"]
    
    # Identify different sections based on position and formatting
    headers = []
    paragraphs = []
    lists = []
    tables = []
    
    # Iterate through blocks to identify structure
    for block in blocks:
        if block["type"] == 0:  # Text block
            lines = block["lines"]
            block_bbox = block["bbox"]  # (x0, y0, x1, y1)
            block_text = ""
            
            # Get the full text from all spans in all lines
            for line in lines:
                for span in line["spans"]:
                    block_text += span["text"] + " "
            
            block_text = block_text.strip()
            
            # Check if this is a header (based on font size and position)
            is_header = False
            font_sizes = []
            for line in lines:
                for span in line["spans"]:
                    font_sizes.append(span["size"])
            
            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
            
            # Headers typically have larger font sizes
            if avg_font_size > 12:  # This threshold might need adjustment
                headers.append({
                    "text": block_text,
                    "bbox": block_bbox,
                    "font_size": avg_font_size,
                    "level": 1 if avg_font_size > 16 else 2  # Estimate header level
                })
                is_header = True
            
            # Check if this is a list item
            if not is_header and block_text.startswith(("•", "-", "*")) or (len(block_text) > 2 and block_text[0:2].isdigit() and block_text[2:4] in [". ", ") "]):
                lists.append({
                    "text": block_text,
                    "bbox": block_bbox
                })
            
            # Otherwise, consider it a paragraph
            elif not is_header:
                paragraphs.append({
                    "text": block_text,
                    "bbox": block_bbox
                })
                
        elif block["type"] == 1:  # Image block
            # Record the image position
            image_bbox = block["bbox"]
    
    # Try to detect tables by analyzing shapes - simplified version without drawing analysis
    # Instead, we'll use a heuristic approach based on text alignment
    
    # Get all paragraphs and sort them by y-position
    sorted_paras = sorted(paragraphs, key=lambda p: p["bbox"][1])
    
    # Look for grid-like structures in text positioning
    row_positions = []
    for para in sorted_paras:
        y_pos = para["bbox"][1]  # Top y position
        # Check if this is a new row (not close to any existing row position)
        is_new_row = all(abs(y_pos - existing) > 10 for existing in row_positions)
        if is_new_row:
            row_positions.append(y_pos)
    
    col_positions = []
    for para in sorted(paragraphs, key=lambda p: p["bbox"][0]):
        x_pos = para["bbox"][0]  # Left x position
        # Check if this is a new column
        is_new_col = all(abs(x_pos - existing) > 20 for existing in col_positions)
        if is_new_col:
            col_positions.append(x_pos)
    
    # If we have multiple rows and columns with similar spacing, it might be a table
    if len(row_positions) >= 3 and len(col_positions) >= 2:
        # Find the bounding box
        table_x0 = min(col_positions)
        table_y0 = min(row_positions)
        table_x1 = max([p["bbox"][2] for p in paragraphs if abs(p["bbox"][0] - table_x0) < 20])
        table_y1 = max([p["bbox"][3] for p in paragraphs if abs(p["bbox"][1] - row_positions[-1]) < 10])
        
        tables.append({
            "bbox": (table_x0, table_y0, table_x1, table_y1),
            "rows": len(row_positions),
            "columns": len(col_positions)
        })
            
    # Identify columns through text block alignment
    x_clusters = {}
    for p in paragraphs:
        x_pos = int(p["bbox"][0] / 20) * 20  # Group in 20-point clusters
        if x_pos not in x_clusters:
            x_clusters[x_pos] = 0
        x_clusters[x_pos] += 1
    
    # If we have multiple strong x-clusters, we probably have columns
    strong_clusters = [x for x, count in x_clusters.items() if count > len(paragraphs) / 10]
    if len(strong_clusters) > 1:
        columns = [{"x": x, "count": x_clusters[x]} for x in sorted(strong_clusters)]
    else:
        columns = []
    
    return {
        "width": width,
        "height": height,
        "headers": headers,
        "paragraphs": paragraphs,
        "lists": lists,
        "tables": tables,
        "columns": columns
    }

def analyze_page_with_vlm(page_data):
    """Use a Visual Language Model to understand document content"""
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
    except ImportError:
        return {
            "error": "Transformers library not installed. Run 'pip install transformers'",
            "Identify all sections in this document page.": "Analysis unavailable - missing dependencies",
            "Describe any diagrams, tables or figures and their purpose.": "Analysis unavailable - missing dependencies",
            "Explain the relationship between text and visual elements.": "Analysis unavailable - missing dependencies", 
            "Identify any special notations or symbols used.": "Analysis unavailable - missing dependencies"
        }
    
    # Check if we have a valid page_image
    if "page_image" not in page_data or page_data["page_image"] is None:
        # We need to render the page to an image first
        try:
            if "page_obj" in page_data:  # If we have a PyMuPDF page object
                pix = page_data["page_obj"].get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.samples
                page_image = Image.frombytes("RGB", [pix.width, pix.height], img_data)
            else:
                # Create a blank image if we don't have a page object
                print("Warning: No page image available for VLM analysis")
                page_image = Image.new('RGB', (800, 1000), color='white')
        except Exception as e:
            print(f"Error rendering page image: {e}")
            # Create a blank image as fallback
            page_image = Image.new('RGB', (800, 1000), color='white')
    else:
        page_image = page_data["page_image"]
    
    # Load VLM model (e.g., BLIP-2)
    try:
        processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = AutoModelForVision2Seq.from_pretrained("Salesforce/blip2-opt-2.7b")
        
        # Process with VLM
        inputs = processor(images=page_image, return_tensors="pt")
        
        # Generate different types of analysis
        prompts = [
            "Identify all sections in this document page.",
            "Describe any diagrams, tables or figures and their purpose.",
            "Explain the relationship between text and visual elements.",
            "Identify any special notations or symbols used."
        ]
        
        results = {}
        for prompt in prompts:
            try:
                # Add the text prompt to the processor inputs
                text_inputs = processor(text=prompt, return_tensors="pt")
                inputs.update(text_inputs)
                
                # Generate analysis
                outputs = model.generate(**inputs, max_length=100)
                results[prompt] = processor.decode(outputs[0], skip_special_tokens=True)
            except Exception as e:
                print(f"Error generating analysis for prompt '{prompt}': {e}")
                results[prompt] = f"Analysis failed: {str(e)}"
        
        return results
    except Exception as e:
        print(f"Error in VLM analysis: {e}")
        return {
            "error": f"VLM analysis failed: {str(e)}",
            "Identify all sections in this document page.": "Analysis unavailable",
            "Describe any diagrams, tables or figures and their purpose.": "Analysis unavailable",
            "Explain the relationship between text and visual elements.": "Analysis unavailable", 
            "Identify any special notations or symbols used.": "Analysis unavailable"
        }

def process_handwritten_content(page_image):
    """Process handwritten content using specialized OCR"""
    try:
        import pytesseract
    except ImportError:
        return "Pytesseract not installed. Run 'pip install pytesseract' and install Tesseract OCR."
    
    # Configure Tesseract for handwritten text
    custom_config = r'--oem 3 --psm 6 -l eng'
    
    # For better results with handwriting
    handwritten_text = pytesseract.image_to_string(
        page_image, 
        config=custom_config
    )
    
    return handwritten_text

def analyze_diagrams_and_connections(page):
    """Analyze diagrams and their connections in the page"""
    try:
        # Extract layout information
        layout = page.get("layout", {})
        
        # Look for potential diagrams based on visual features
        diagrams = []
        
        # Check if we have a page image to work with
        if "page_image" in page and page["page_image"]:
            # Here we could implement diagram detection using CV
            # For now, we'll use a simple placeholder implementation
            diagrams.append({
                "type": "placeholder",
                "from": "element1",
                "to": "element2",
                "relationship": "connects to"
            })
        
        # Look for text blocks that might indicate diagram labels or relationships
        if "text_blocks" in page:
            for block in page["text_blocks"]:
                # Look for arrow characters or connection terms
                text = "".join(block) if isinstance(block, list) else str(block)
                if "->" in text or "→" in text or "relationship" in text.lower():
                    parts = text.replace("->", "→").split("→")
                    if len(parts) >= 2:
                        diagrams.append({
                            "from": parts[0].strip(),
                            "to": parts[1].strip(),
                            "relationship": "connects to"
                        })
        
        return diagrams
    except Exception as e:
        print(f"Error analyzing diagrams: {e}")
        return []

def build_document_knowledge_graph(processed_pages):
    """Build a knowledge graph from processed document pages"""
    # Initialize knowledge graph structure
    knowledge_graph = {
        "nodes": [],
        "edges": []
    }
    
    node_ids = set()  # Track unique node IDs
    
    # Extract entities and relationships from pages
    for page in processed_pages:
        # Process diagrams and their connections
        if "diagrams" in page:
            for diagram in page.get("diagrams", []):
                if "from" in diagram and "to" in diagram:
                    # Add source node if new
                    if diagram["from"] not in node_ids:
                        knowledge_graph["nodes"].append({
                            "id": diagram["from"],
                            "label": diagram["from"],
                            "type": "entity"
                        })
                        node_ids.add(diagram["from"])
                    
                    # Add target node if new
                    if diagram["to"] not in node_ids:
                        knowledge_graph["nodes"].append({
                            "id": diagram["to"],
                            "label": diagram["to"],
                            "type": "entity"
                        })
                        node_ids.add(diagram["to"])
                    
                    # Add edge between nodes
                    knowledge_graph["edges"].append({
                        "source": diagram["from"],
                        "target": diagram["to"],
                        "label": diagram.get("relationship", "connects to")
                    })
        
        # Extract entities from VLM analysis if available
        if "vlm_analysis" in page:
            for section_text in page["vlm_analysis"].values():
                # Simple named entity extraction (could be enhanced with NER)
                # For now, just look for capitalized terms as potential entities
                words = section_text.split()
                for i, word in enumerate(words):
                    if (len(word) > 3 and word[0].isupper() and 
                        word.lower() not in ["this", "that", "these", "those", "there", "their"]):
                        
                        entity = word.strip(".,():;\"'")
                        if entity and entity not in node_ids and len(entity) > 3:
                            knowledge_graph["nodes"].append({
                                "id": entity,
                                "label": entity,
                                "type": "concept",
                                "page": page.get("page_num", 0)
                            })
                            node_ids.add(entity)
    
    return knowledge_graph
def extract_images_from_pdf(pdf_path):
    """Extract images from PDF without VLM analysis"""
    import fitz  # PyMuPDF
    import io
    from PIL import Image
    
    # Open the PDF
    doc = fitz.open(pdf_path)
    images = []
    
    try:
        # Process PDF pages
        for page_num in range(len(doc)):
            # Limit to 20 pages for performance
            if page_num >= 20:
                break
                
            page = doc.load_page(page_num)
            
            # Method 1: Get image references
            image_list = page.get_images(full=True)
            
            if image_list:
                # Process image references
                for img_index, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]  # image xref number
                        
                        # Extract image bytes
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Load image using PIL
                        img = Image.open(io.BytesIO(image_bytes))
                        
                        # Filter out very small images (likely icons, bullets, etc.)
                        if img.width < 100 or img.height < 100:
                            continue
                        
                        images.append({
                            "image": img,
                            "page_num": page_num,
                            "type": "embedded",
                            "size": (img.width, img.height),
                            "xref": xref
                        })
                    except Exception as e:
                        print(f"Error extracting embedded image: {e}")
            
            # Method 2: If few images found, capture whole page as image
            if len(images) < 5 or page_num < 5:  # Only for first few pages or if few images
                try:
                    # Render page at higher resolution
                    zoom = 2  # Adjust as needed for quality
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Convert to PIL Image
                    img_data = pix.tobytes("png")
                    page_img = Image.open(io.BytesIO(img_data))
                    
                    # Save the page image
                    images.append({
                        "image": page_img,
                        "page_num": page_num,
                        "type": "page_render",
                        "size": (page_img.width, page_img.height)
                    })
                except Exception as e:
                    print(f"Error rendering page {page_num+1}: {e}")
        
        # Close the document
        doc.close()
        
        return images
    except Exception as e:
        print(f"Error in image extraction: {e}")
        try:
            doc.close()
        except:
            pass
        return []
    
def enhanced_pdf_processing(file_path):
    """Complete pipeline for enhanced PDF processing"""
    
    # 1. Extract document structure and pages
    pages, doc = process_document_visually(file_path)
    
    try:
        # 2. For each page, process content
        processed_pages = []
        for page in pages:
            try:
                # 3. Visual analysis with VLM
                page["vlm_analysis"] = analyze_page_with_vlm(page)
                
                # 4. Process diagrams and connections
                page["diagrams"] = analyze_diagrams_and_connections(page)
                
                # 5. Process any handwritten elements
                page["handwritten"] = process_handwritten_content(page["page_image"])
                
                processed_pages.append(page)
            except Exception as e:
                print(f"Error processing page {page['page_num']}: {e}")
                # Add the page with error information
                page["error"] = str(e)
                processed_pages.append(page)
        
        # 6. Create document knowledge graph from all relationships
        try:
            knowledge_graph = build_document_knowledge_graph(processed_pages)
        except Exception as e:
            print(f"Error building knowledge graph: {e}")
            knowledge_graph = None
        
        # Close the document
        if doc:
            doc.close()
        
        return {
            "processed_pages": processed_pages,
            "knowledge_graph": knowledge_graph
        }
    except Exception as e:
        print(f"Error in enhanced PDF processing: {e}")
        # Make sure to close the document even on error
        if doc:
            doc.close()
        return {"error": str(e)}

def enhanced_pdf_processing_light(file_path):
    """Lighter PDF processing without intensive image analysis"""
    import time
    
    start_time = time.time()
    
    try:
        doc = fitz.open(file_path)
        processed_pages = []
        
        # Get document metadata
        metadata = {
            "title": doc.metadata.get("title", "Unknown"),
            "author": doc.metadata.get("author", "Unknown"),
            "subject": doc.metadata.get("subject", ""),
            "keywords": doc.metadata.get("keywords", ""),
            "page_count": len(doc),
            "file_size_kb": os.path.getsize(file_path) / 1024
        }
        
        # Process a maximum of 20 pages to keep performance reasonable
        max_pages = min(len(doc), 20)
        
        # Process pages
        for page_num in range(max_pages):
            page = doc.load_page(page_num)
            
            # Extract text
            text = page.get_text("text")
            blocks = page.get_text("blocks")
            
            # Simple layout analysis
            text_layout = {
                "blocks_count": len(blocks),
                "text_length": len(text),
                "has_images": len(page.get_images()) > 0
            }
            
            # Find structural elements
            structural_elements = extract_document_structure(text)
            
            processed_pages.append({
                "page_num": page_num + 1,
                "text": text[:500] + "..." if len(text) > 500 else text,  # Truncate long text
                "layout": text_layout,
                "structure": structural_elements
            })
        
        # Close the document
        doc.close()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            "processed_pages": processed_pages,
            "metadata": metadata,
            "processing_time_sec": processing_time,
            "processing_info": "Light processing mode (without image analysis)"
        }
    except Exception as e:
        print(f"Error processing document: {e}")
        # Make sure to close the document even on error
        try:
            if 'doc' in locals():
                doc.close()
        except:
            pass
        return {"error": str(e)}
def analyze_image_with_gemini(image):
    """Use Google's Gemini model to analyze an image"""
    import io
    import google.generativeai as genai
    
    try:
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()
        
        # Initialize the model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Create content parts
        image_part = {"mime_type": "image/png", "data": image_bytes}
        prompt = "Analyze this figure from an academic document. Describe what it shows, its purpose, any data visualizations, and the key information it's conveying."
        
        # Generate response
        response = model.generate_content([image_part, prompt])
        
        return response.text
    except Exception as e:
        return f"Image analysis failed: {str(e)}"
def get_image_brightness(image):
    """Calculate average brightness of an image"""
    from PIL import ImageStat
    
    # Convert to grayscale for brightness calculation
    gray_image = image.convert('L')
    stat = ImageStat.Stat(gray_image)
    return stat.mean[0] / 255.0  # Normalize to 0-1

def brightness_level(value):
    """Convert brightness value to description"""
    if value < 0.3:
        return "Dark"
    elif value < 0.6:
        return "Medium"
    else:
        return "Bright"

def is_colorful(image):
    """Detect if an image has multiple colors or is mostly monochrome"""
    # Convert to RGB
    rgb_image = image.convert('RGB')
    
    # Sample pixels for color variety (for performance)
    width, height = rgb_image.size
    sample_size = min(100, width * height)
    
    # Get unique colors from sampled pixels
    unique_colors = set()
    import random
    for _ in range(sample_size):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        pixel = rgb_image.getpixel((x, y))
        unique_colors.add(pixel)
    
    # If there are more than 10 unique colors, consider it colorful
    return len(unique_colors) > 10

def describe_image_simple(image):
    """Provide a simple description of image content without using VLM"""
    # Get image properties
    width, height = image.size
    mode = image.mode
    format_type = image.format if hasattr(image, 'format') else "Unknown"
    
    # Analyze image characteristics
    brightness = get_image_brightness(image)
    colorfulness = is_colorful(image)
    
    # Create description
    description = f"Image type: {format_type if format_type else 'Image'}\n"
    description += f"Size: {width}x{height} pixels\n"
    description += f"Color mode: {mode}\n"
    
    # Add qualitative descriptions
    if width / height > 1.5:
        description += "Orientation: Wide/Landscape (likely a figure, chart, or panorama)\n"
    elif height / width > 1.5:
        description += "Orientation: Tall/Portrait (likely a diagram or vertical illustration)\n"
    else:
        description += "Orientation: Balanced aspect ratio\n"
        
    description += f"Brightness: {brightness_level(brightness)}\n"
    
    if colorfulness:
        description += "Contains multiple colors (likely a photograph, illustration or color chart)\n"
    else:
        description += "Limited color range (likely a diagram, chart, or grayscale image)\n"
    
    return description



def extract_document_structure(text):
    """Extract structural elements from document text without heavy processing"""
    import re
    
    # Find potential headers
    header_pattern = re.compile(r'^([0-9]+\.)+\s+(.+)$|^(Chapter|Section)\s+([0-9]+)[\.\:]?\s*(.+)$|^([IVX]+)\.\s+(.+)$', 
                                re.MULTILINE)
    headers = [match.group(0) for match in header_pattern.finditer(text)]
    
    # Find potential list items
    list_pattern = re.compile(r'^\s*[•\-\*]\s+.+$|^\s*[0-9]+[\.\)]\s+.+$', re.MULTILINE)
    list_items = [match.group(0) for match in list_pattern.finditer(text)]
    
    # Find references or citations
    citation_pattern = re.compile(r'\[\d+\]|\(\w+\s+et al\.\s*,\s*\d{4}\)')
    citations = [match.group(0) for match in citation_pattern.finditer(text)]
    
    # Check for tables (simplistic approach)
    has_tables = '|' in text and '+--' in text or '+--' in text
    
    return {
        "headers": headers[:10],  # Limit to first 10 for performance
        "list_items_count": len(list_items),
        "citations_count": len(citations),
        "has_tables": has_tables
    }

# Streamlit UI
st.set_page_config(
    page_title="PDF Analyzer",
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
if 'translation_result' not in st.session_state:
    st.session_state.translation_result = None

# App title
st.title("📄 PDF Analyzer")
st.markdown("Upload a PDF document to analyze and ask questions about it.")

# Sidebar for file upload and settings
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf', 'txt', 'docx'])
    
    st.header("Processing Options")
    light_mode = st.checkbox("Use Light Mode (Faster)", value=True, 
                            help="Process documents without intensive image analysis")
    
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
                    # Main document processing for embeddings and QA
                    document_data = process_document(file_path)
                    save_document_metadata(file_id, document_data)
                    
                    # Generate summary
                    st.session_state.summary = generate_summary(document_data["chunks"])
                    
                    # Generate insights
                    st.session_state.insights = generate_insights(st.session_state.summary)
                    
                    # Store paper metadata
                    if "paper_metadata" in document_data:
                        st.session_state.paper_metadata = document_data["paper_metadata"]
                    
                    # Visual analysis - only if light mode is off
                    if not light_mode:
                        st.session_state.visual_analysis = enhanced_pdf_processing(file_path)
                    else:
                        # Use the lighter processing
                        st.session_state.light_analysis = enhanced_pdf_processing_light(file_path)
                    
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
                    st.session_state.translation_result = translation
                    st.markdown("### Translation")
                    st.markdown(translation)
                except Exception as e:
                    st.error(f"Translation error: {str(e)}")

# Main content area
if st.session_state.file_id and st.session_state.summary:
    
    # Add tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "Ask Questions", "Paper Metadata", "Document Structure", "Figures & Images"])
    
    # Tab 1: Summary and insights
    with tab1:
        st.header("Document Summary")
        st.markdown(st.session_state.summary)
        
        st.header("Document Insights")
        st.markdown(st.session_state.insights)
        
        # Show translation if available
        if st.session_state.translation_result:
            st.header("Translation")
            st.markdown(st.session_state.translation_result)
    
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
                        
                        if document_data and "collection_name" in document_data:
                            collection_name = document_data["collection_name"]
                            persist_directory = os.path.join(STORAGE_FOLDER, collection_name)
                            
                            # Load the vector store
                            vectorstore = Chroma(
                                persist_directory=persist_directory,
                                embedding_function=embeddings
                            )
                            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                            
                            # Get answer
                            response = answer_question(question, retriever)
                            answer = response["answer"]
                            
                            # Store in session state
                            st.session_state.answers.append({"question": question, "answer": answer})
                        else:
                            st.error("Failed to load document data. Please try reuploading the document.")
                    except Exception as e:
                        st.error(f"Error answering question: {str(e)}")
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
    
    # Tab 4: Document Structure
    with tab4:
        st.header("Document Structure")
        
        if st.session_state.file_id:
            if "light_analysis" in st.session_state:
                # Show the light analysis
                light_data = st.session_state.light_analysis
                
                # Show document metadata
                if "metadata" in light_data:
                    st.subheader("Document Metadata")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"Title: {light_data['metadata']['title']}")
                        st.write(f"Author: {light_data['metadata']['author']}")
                        st.write(f"Subject: {light_data['metadata']['subject']}")
                    with col2:
                        st.write(f"Pages: {light_data['metadata']['page_count']}")
                        st.write(f"File Size: {light_data['metadata']['file_size_kb']:.1f} KB")
                        st.write(f"Processing Time: {light_data['processing_time_sec']:.2f} seconds")
                
                # Initialize a session state for text previews if not exists
                if "text_previews" not in st.session_state:
                    st.session_state.text_previews = {}
                
                # Display pages
                st.subheader("Page Analysis")
                for page in light_data.get("processed_pages", []):
                    # Create unique key for this page
                    page_key = f"page_{page['page_num']}"
                    
                    with st.expander(f"Page {page['page_num']}"):
                        # Show structure
                        structure = page.get("structure", {})
                        
                        # Headers
                        if structure.get("headers"):
                            st.write("**Headers:**")
                            for header in structure["headers"]:
                                st.write(f"- {header}")
                        
                        # Lists and citations
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"List Items: {structure.get('list_items_count', 0)}")
                        with col2:
                            st.write(f"Citations: {structure.get('citations_count', 0)}")
                        
                        # Tables
                        if structure.get("has_tables"):
                            st.write("**Contains tables**")
                        
                        # Show text preview with a button toggle instead of nested expander
                        if st.button(f"Show/Hide Text Preview", key=f"text_btn_{page['page_num']}"):
                            # Toggle the visibility state in session_state
                            if page_key in st.session_state.text_previews:
                                st.session_state.text_previews.pop(page_key)
                            else:
                                st.session_state.text_previews[page_key] = True
                        
                        # Show text area if toggled on
                        if page_key in st.session_state.text_previews:
                            st.text_area("Text Content", page.get("text", ""), height=200)
            else:
                st.info("Document structure analysis not available. Please process a document first.")
        else:
            st.info("Please upload a document to analyze its structure.")

    # Tab 5: Figures & Images
    with tab5:
        st.header("Figures & Images")
        
        if st.session_state.file_id:
            # Add button to extract images (rather than automatic)
            if "extracted_images" not in st.session_state:
                if st.button("Extract Images and Figures"):
                    with st.spinner("Extracting images from document..."):
                        try:
                            file_path = os.path.join(UPLOAD_FOLDER, st.session_state.filename)
                            st.session_state.extracted_images = extract_images_from_pdf(file_path)
                        except Exception as e:
                            st.error(f"Error extracting images: {str(e)}")
                            st.session_state.extracted_images = []
            
            # Initialize image details toggle state if not exists
            if "image_details_visible" not in st.session_state:
                st.session_state.image_details_visible = {}
                
            # Display extraction status
            if "extracted_images" in st.session_state:
                images = st.session_state.extracted_images
                if images:
                    st.success(f"Found {len(images)} images/figures in the document")
                    
                    # Filter options
                    st.sidebar.header("Image Filters")
                    show_embedded = st.sidebar.checkbox("Show Embedded Images", value=True)
                    show_page_renders = st.sidebar.checkbox("Show Page Renders", value=True)
                    
                    # Filter images based on type
                    filtered_images = [img for img in images if 
                                    (img["type"] == "embedded" and show_embedded) or
                                    (img["type"] == "page_render" and show_page_renders)]
                    
                    if filtered_images:
                        # Display images in a grid
                        for i, img_data in enumerate(filtered_images):
                            img_key = f"img_{i}"
                            with st.expander(f"{img_data['type'].title()} on Page {img_data['page_num']+1}"):
                                col1, col2 = st.columns([1, 1.5])
                                with col1:
                                    st.image(img_data["image"], caption=f"Image #{i+1}")
                                
                                with col2:
                                    # Show image details
                                    st.write(f"Size: {img_data['size'][0]} x {img_data['size'][1]} pixels")
                                    
                                    # Add analyze button for each image
                                    if st.button(f"Analyze this image", key=f"analyze_btn_{i}"):
                                        with st.spinner("Analyzing image..."):
                                            analysis = analyze_image_with_gemini(img_data["image"])
                                            st.markdown("### AI Analysis")
                                            st.markdown(analysis)
                                    
                                    # Toggle button for image details instead of nested expander
                                    if st.button(f"Show/Hide Image Details", key=f"detail_btn_{i}"):
                                        # Toggle visibility state
                                        if img_key in st.session_state.image_details_visible:
                                            st.session_state.image_details_visible.pop(img_key)
                                        else:
                                            st.session_state.image_details_visible[img_key] = True
                                    
                                    # Show details if toggled on
                                    if img_key in st.session_state.image_details_visible:
                                        simple_desc = describe_image_simple(img_data["image"])
                                        st.text_area("Basic Image Details", simple_desc, height=150)
                    else:
                        st.info("No images match the current filters.")
                else:
                    st.info("No images found in the document.")
            else:
                st.info("Click 'Extract Images and Figures' to process the document.")
        else:
            st.info("Please upload a document to analyze figures and images.")

# Helper functions for image processing - moving these earlier in the file

