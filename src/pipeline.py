import time
import os
import logging
from typing import Dict, List, Optional
from embedder import Embedder
from chunker import Chunker
from vector_store import VectorStore
from router import DocumentRouter
import pdfplumber
from extractors.OCR import OCRExtractor
from pdf2image import convert_from_path


class Pipeline:
    """IntelliDoc document processing pipeline"""
    
    def __init__(self):
        self.embedder = Embedder()
        self.chunker = Chunker()
        self.vector_store = VectorStore()
        self.router = DocumentRouter()
        self.ocr_extractor = OCRExtractor()  
    
    def process_pdf_page(self, pdf_path: str, page_number: int) -> Dict:
        """
        Process a single PDF page.
        
        Args:
            pdf_path: Path to PDF file
            page_number: Zero-indexed page number
        
        Returns:
            Status dict with results or error info
        """
        if not pdf_path:
            raise ValueError("pdf_path cannot be empty")
        if page_number < 0:
            raise ValueError("page_number must be >= 0")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        metadata = {
            "pdf_path": pdf_path,
            "page_number": page_number
        }
        
        try:
            decision = self.router.route_page(pdf_path, page_number)
            route = decision.get("route")
            metadata["route"] = route
            
            
            extracted_text = self._extract_text(pdf_path, page_number, route)
            
            if not extracted_text or not extracted_text.strip():
                return {
                    "status": "empty",
                    "reason": "no_text_extracted",
                    "metadata": metadata
                }
            
            result = self.process(extracted_text, metadata)
            
            return {
                **result,
                "metadata": metadata
            }
        
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "metadata": metadata
            }
    
    def _extract_text(self, pdf_path: str, page_number: int, route: str) -> str:
        """Extract text based on route type"""
        
        if route == "tables":
            return self._extract_tables(pdf_path, page_number)
        elif route == "text":
            return self._extract_text_native(pdf_path, page_number)
        elif route == "ocr":
            return self._extract_ocr(pdf_path, page_number)
        else:
            raise ValueError(f"Unknown route: {route}")
    
    def _extract_tables(self, pdf_path: str, page_number: int) -> str:
        """Extract tables from page"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_number >= len(pdf.pages):
                    raise IndexError(f"Page {page_number} out of range (PDF has {len(pdf.pages)} pages)")
                
                raw_tables = pdf.pages[page_number].extract_tables()
                
                if not raw_tables:
                    return ""
                
                formatted_tables = []
                for table in raw_tables:
                    table_str = "\n".join([
                        ", ".join([str(c) if c else "" for c in row]) 
                        for row in table
                    ])
                    formatted_tables.append(table_str)
                
                return "\n\n".join(formatted_tables)
        
        except Exception as e:
            logger.error(f"Table extraction failed for {pdf_path} page {page_number}: {e}")
            raise
    
    def _extract_text_native(self, pdf_path: str, page_number: int) -> str:
        """Extract text directly from PDF"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_number >= len(pdf.pages):
                    raise IndexError(f"Page {page_number} out of range")
                
                text = pdf.pages[page_number].extract_text()
                return text or ""
        
        except Exception as e:
            raise ValueError(f"Text extraction failed for {pdf_path} page {page_number}: {e}")
    
    def _extract_ocr(self, pdf_path: str, page_number: int) -> str:
        """Extract text using OCR"""
        try:
            images = convert_from_path(
                pdf_path,
                first_page=page_number + 1,
                last_page=page_number + 1
            )
            
            if not images:
                return ""
            
            pil_image = images[0]
            ocr_result = self.ocr_extractor.extract(image=pil_image)
            
            return ocr_result.get("extracted_text", "")
        
        except Exception as e:
            raise
    
    def process(self, text: str, metadata: Dict) -> Dict:
        """
        Process extracted text: chunk → embed → index
        
        Args:
            text: Extracted text from document
            metadata: Document metadata (pdf_path, page_number, etc.)
        
        Returns:
            Status dict with processing results
        """
        start_time = time.time()
        
        try:
            
            # Chunk text
            chunks = self.chunker.chunk(text)
            if not chunks:
                return {
                    "status": "empty",
                    "reason": "no_chunks",
                    "chunks_indexed": 0,
                    "time_taken_ms": (time.time() - start_time) * 1000
                }
            
            
            embeddings = self.embedder.embed(chunks)
            
            if embeddings is None or len(embeddings) == 0:
                raise ValueError("Embedder returned no embeddings")
            
            
            indexed_documents = [
                {
                    "text": chunk,
                    "embedding": embeddings[i],
                    "metadata": {
                        **metadata,
                        "chunk_id": i,
                        "chunk_text": chunk[:100]  
                    }
                }
                for i, chunk in enumerate(chunks)
            ]

            self.vector_store.index_documents(indexed_documents)
            
            end_time = time.time()
            elapsed_ms = (end_time - start_time) * 1000
            
            result = {
                "status": "success",
                "chunks_indexed": len(chunks),
                "time_taken_ms": elapsed_ms,
                "avg_chunk_size": len(text) / len(chunks)
            }
            
            return result
        
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "chunks_indexed": 0,
                "time_taken_ms": (time.time() - start_time) * 1000
            }


if __name__ == "__main__":
    import os

    pipeline = Pipeline()

    # Resolve path relative to this script's location (src/), not cwd
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(base_dir, "..", "sample_data", "sample_text.pdf")

    result = pipeline.process_pdf_page(pdf_path, page_number=0)
    print("\n\n")
    print(result)