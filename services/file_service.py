# stelle_backend/services/file_service.py
import os 
import re
import ast
import logging
import asyncio
from io import BytesIO
from typing import List, Dict, Any
from datetime import datetime, timezone # <-- ADDED for consistency

from fastapi import UploadFile
import docx2txt
import fitz # PyMuPDF

from config import logger
from database import FAISS_DIM, doc_index, code_index, file_doc_memory_map, code_memory_map, uploads_collection
# CRITICAL FIX: REMOVE TOP-LEVEL IMPORT TO BREAK CIRCULAR DEPENDENCY
# from services.ai_service import generate_text_embedding 
import numpy as np

# --- Text Splitting Utilities (Section 1) ---
def extract_code_segments(code: str) -> List[Dict[str, str]]:
    """Extracts function, class, or general code chunks using AST or fallback."""
    # ... (body remains the same) ...
    segments = []
    try:
        tree = ast.parse(code)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno - 1
                end = getattr(node, 'end_lineno', node.body[-1].lineno if node.body else node.lineno)
                code_lines = code.splitlines()[start:end]
                segment_code = "\n".join(code_lines)
                segments.append({"name": node.name, "code": segment_code})
    except Exception as e:
        logger.warning(f"AST parse failed: {e}. Falling back to manual chunking.")
    
    if not segments:
        lines = code.splitlines()
        chunk_size = 300
        for i in range(0, len(lines), chunk_size):
            chunk = "\n".join(lines[i : i + chunk_size])
            segments.append({"name": f"chunk_{i//chunk_size+1}", "code": chunk})
    
    return segments

def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Splits general text content into overlapping chunks."""
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [" ".join(words)]
        
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


# -----------------------------------------------
# --- File Extraction Functions (Section 2 - MOVED UP TO BE DEFINED) --- 
# -----------------------------------------------

async def extract_text_from_pdf(file: UploadFile) -> str:
    """Extracts text content from a PDF file."""
    try:
        file.file.seek(0)
        contents = await file.read()
        doc = fitz.open(stream=contents, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        logger.info(f"PDF text extracted (first 200 chars): {text[:200]}...")
        return text
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return ""

async def extract_text_from_docx(file: UploadFile) -> str:
    """Extracts text content from a DOCX file."""
    try:
        file.file.seek(0)
        contents = await file.read()
        text = await asyncio.to_thread(docx2txt.process, BytesIO(contents))
        logger.info(f"DOCX text extracted (first 200 chars): {text[:200]}...")
        return text
    except Exception as e:
        logger.error(f"DOCX extraction error: {e}")
        return ""

async def extract_text_from_txt(file: UploadFile) -> str:
    """Extracts text content from a generic text file."""
    try:
        file.file.seek(0)
        contents = await file.read()
        text = contents.decode("utf-8")
        logger.info(f"TXT/Code text extracted (first 200 chars): {text[:200]}...")
        return text
    except Exception as e:
        logger.error(f"TXT extraction error: {e}")
        return ""

# -----------------------------------------------
# --- Index Management Functions (Section 3 - Now Calls Defined Functions) ---
# -----------------------------------------------

async def process_and_index_file(user_id: str, session_id: str, file: UploadFile) -> Dict[str, Any]:
    """Master function to process, embed, and index a single uploaded file."""
    
    # CRITICAL FIX: Import locally to break the AI service loop
    from services.ai_service import generate_text_embedding 
    
    filename = file.filename
    filename_lower = filename.lower()
    ext = os.path.splitext(filename_lower)[1]
    
    allowed_text_types = [
        "text/plain", "application/pdf", "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]
    allowed_code_extensions = [".py", ".js", ".java", ".cpp", ".c", ".ts"]

    modality = None
    extracted_text = None

    if ext in allowed_code_extensions:
        modality = "code"
        extracted_text = await extract_text_from_txt(file) # <-- NOW DEFINED
    elif file.content_type == "application/pdf":
        modality = "document"
        extracted_text = await extract_text_from_pdf(file) # <-- NOW DEFINED
    elif file.content_type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        modality = "document"
        extracted_text = await extract_text_from_docx(file) # <-- NOW DEFINED
    elif file.content_type == "text/plain":
        modality = "document" 
        extracted_text = await extract_text_from_txt(file) # <-- NOW DEFINED
    else:
        return {
            "filename": filename, "success": False, "message": "Format not allowed"
        }

    if not extracted_text:
        return {"filename": filename, "success": False, "message": "Parsing Failed"}
        
    embeddings_added = 0
    
    # ... (rest of the embedding and indexing logic is correct) ...
    if modality == "code":
        segments = extract_code_segments(extracted_text)
        if not segments:
             return {"filename": filename, "success": False, "message": "Code segmentation failed"}
             
        for segment in segments:
            segment_text = segment["code"]
            embedding_vector = await generate_text_embedding(segment_text)
            
            if embedding_vector and len(embedding_vector) == FAISS_DIM:
                new_vector = np.array(embedding_vector, dtype="float32").reshape(1, -1)
                new_id = code_index.ntotal
                code_index.add(new_vector)
                
                code_memory_map[new_id] = {
                    "user_id": user_id, "session_id": session_id, "filename": filename,
                    "modality": modality, "segment_name": segment["name"],
                    "text_snippet": segment_text[:300], "usage_count": 0,
                }
                
                await uploads_collection.insert_one({
                    "user_id": user_id, "session_id": session_id, "filename": filename,
                    "modality": modality, "segment_name": segment["name"],
                    "text_snippet": segment_text[:300], "embedding": embedding_vector,
                    "timestamp": datetime.now(timezone.utc), "query_count": 0,
                })
                embeddings_added += 1
        
        logger.info(f"Code file '{filename}' segmented into {embeddings_added}/{len(segments)} segments.")
        return {"filename": filename, "success": True, "segments": embeddings_added}
        
    elif modality == "document":
        chunks = split_text_into_chunks(extracted_text)
        
        for i, chunk in enumerate(chunks):
            embedding_vector = await generate_text_embedding(chunk)
            
            if embedding_vector and len(embedding_vector) == FAISS_DIM:
                new_vector = np.array(embedding_vector, dtype="float32").reshape(1, -1)
                new_id = doc_index.ntotal
                doc_index.add(new_vector)
                
                snippet = chunk[:300]
                file_doc_memory_map[new_id] = {
                    "user_id": user_id, "session_id": session_id, "filename": filename,
                    "modality": modality, "chunk_index": i,
                    "text_snippet": snippet, "usage_count": 0,
                }
                
                await uploads_collection.insert_one({
                    "user_id": user_id, "session_id": session_id, "filename": filename,
                    "modality": modality, "chunk_index": i,
                    "text_snippet": snippet, "embedding": embedding_vector,
                    "timestamp": datetime.now(timezone.utc), "query_count": 0,
                })
                embeddings_added += 1
                
        logger.info(f"Document file '{filename}' chunked into {embeddings_added}/{len(chunks)} chunks.")
        return {"filename": filename, "success": True, "chunks": embeddings_added}

    return {"filename": filename, "success": False, "message": "No content was processed."}