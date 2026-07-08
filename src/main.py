import os
import shutil
import pypdf
from fastapi import FastAPI, HTTPException, Query, File, UploadFile
from pydantic import BaseModel
from typing import Optional, Dict, Any
from src.pipeline import Pipeline

app = FastAPI(
    title="IntelliDoc IDP Pipeline API",
    description="Intelligent Document Processing (IDP) and RAG synthesis API using FastAPI.",
    version="1.0.0"
)

# Initialize RAG Pipeline
# FastAPI instantiates this on startup
pipeline = Pipeline()

# Setup local uploads directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class IngestRequest(BaseModel):
    pdf_path: str
    page_number: int


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    top_k: Optional[int] = 5
    ground_truth: Optional[str] = None


@app.get("/")
def read_root():
    return {
        "message": "Welcome to IntelliDoc IDP Pipeline API",
        "swagger_docs": "/docs",
        "redoc_docs": "/redoc"
    }


@app.get("/health")
def health_check():
    """Verify that the API and its dependent databases are reachable."""
    try:
        # Check connection status using vector store initialization or a simple flag
        db_status = "healthy" if pipeline.vector_store else "unreachable"
        return {
            "status": "healthy",
            "database": db_status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/index-page")
def index_document_page(payload: IngestRequest):
    """
    Ingest, segment, and index a single page of a PDF document.
    Automatically routes based on structural layout (Text vs OCR vs Tables).
    """
    try:
        result = pipeline.process_pdf_page(payload.pdf_path, payload.page_number)
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-pdf")
def upload_and_index_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF document and index all of its pages automatically.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Get page count
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            total_pages = len(reader.pages)
            
        # Index all pages
        indexed_pages = []
        errors = []
        total_chunks = 0
        
        for i in range(total_pages):
            page_result = pipeline.process_pdf_page(file_path, i)
            if page_result.get("status") == "success":
                total_chunks += page_result.get("chunks_indexed", 0)
                indexed_pages.append(i)
            else:
                errors.append({"page": i, "error": page_result.get("error", "Unknown error")})
                
        return {
            "status": "success" if not errors else "partial_success",
            "filename": file.filename,
            "saved_path": file_path,
            "total_pages": total_pages,
            "indexed_pages": indexed_pages,
            "total_chunks_indexed": total_chunks,
            "errors": errors
        }
        
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process uploaded PDF: {str(e)}")


@app.post("/query")
def query_rag_pipeline(payload: QueryRequest):
    """
    Submit a query to perform hybrid document retrieval, 
    re-rank candidates, synthesize an answer, and optionally calculate lexical metrics.
    """
    try:
        result = pipeline.query_document(
            query=payload.query,
            top_k=payload.top_k,
            ground_truth=payload.ground_truth,
            session_id=payload.session_id
        )
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error"))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
