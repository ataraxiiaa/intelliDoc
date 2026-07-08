import time
import os
import logging
from typing import Dict, List, Optional
from embedder import Embedder
from chunker import Chunker
from vector_store import VectorStore
from router import DocumentRouter
from retriever import Retriever
from synthesizer import Synthesizer
import pdfplumber
from extractors.OCR import OCRExtractor
from pdf2image import convert_from_path
from evaluator import Evaluator

class Pipeline:
    """IntelliDoc document processing pipeline"""
    
    def __init__(self):
        self.embedder = Embedder()
        self.chunker = Chunker()
        self.vector_store = VectorStore()
        self.router = DocumentRouter()
        self.ocr_extractor = OCRExtractor()  
        self.retriever = Retriever()
        self.synthesizer = Synthesizer()
        self.Evaluator = Evaluator()
        self.memory: Dict[str, List[Dict]] = {}

    def get_chat_history(self, session_id: str) -> List[Dict]:
        if not session_id:
            return []
        if session_id not in self.memory:
            self.memory[session_id] = []
        return self.memory[session_id]

    def add_to_history(self, session_id: str, role: str, content: str):
        if not session_id:
            return
        history = self.get_chat_history(session_id)
        history.append({"role": role, "content": content})
        self.memory[session_id] = history[-10:] # retain last 10 messages

    
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

    def query_document(self, query: str, top_k: int = 5, ground_truth: Optional[str] = None, session_id: Optional[str] = None) -> Dict:
        """
        Process a query against the indexed documents using hybrid search, synthesize an answer,
        and optionally evaluate the result against a ground truth.
        
        Args:
            query: User query string
            top_k: Number of results to return as context
            ground_truth: Optional reference answer to evaluate the prediction against
            session_id: Optional session ID to enable conversational memory
        
        Returns:
            Dict containing synthesized answer object, source results, and evaluation metrics if evaluated.
        """
        start_time = time.time()
        try:
            results = self.retriever.process_query(query, top_k)
            if not results:
                return {
                    "status": "error",
                    "query": query,
                    "error": "No results found in the database.",
                    "answer": None,
                    "results": [],
                    "eval_results": None,
                    "time_taken_ms": (time.time() - start_time) * 1000
                }
            
            # Fetch conversation memory
            history = self.get_chat_history(session_id) if session_id else None

            # Synthesize answer using OpenAI/Qwen LLM passing the chat history
            answer_obj = self.synthesizer.answer_question(query, results, history)
            
            # Save new turn to conversational memory
            if session_id:
                self.add_to_history(session_id, role="user", content=query)
                self.add_to_history(session_id, role="assistant", content=answer_obj.answer)

            eval_results = None
            if ground_truth:
                # Calculate evaluation metrics
                eval_results = self.Evaluator.compute_metrics_batch([{
                    "predicted": answer_obj.answer,
                    "ground_truth": ground_truth
                }])
                
                # Save the results to evals/results/eval_results_2025.json
                base_dir = os.path.dirname(os.path.abspath(__file__))
                eval_file_path = os.path.join(base_dir, "..", "evals", "results", "eval_results_2025.json")
                
                # Load existing runs
                eval_data = []
                if os.path.exists(eval_file_path):
                    try:
                        with open(eval_file_path, "r", encoding="utf-8") as f:
                            import json as json_lib
                            eval_data = json_lib.load(f)
                            if not isinstance(eval_data, list):
                                eval_data = []
                    except Exception:
                        eval_data = []
                
                # Append new run
                eval_data.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "query": query,
                    "predicted": answer_obj.answer,
                    "ground_truth": ground_truth,
                    "metrics": eval_results
                })
                
                # Save back to file
                os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
                with open(eval_file_path, "w", encoding="utf-8") as f:
                    import json as json_lib
                    json_lib.dump(eval_data, f, indent=4, ensure_ascii=False)
            
            return {
                "status": "success",
                "query": query,
                "answer": answer_obj.dict(),
                "results": results,
                "eval_results": eval_results,
                "total_results": len(results),
                "time_taken_ms": (time.time() - start_time) * 1000
            }
            
        except Exception as e:
            return {
                "status": "error",
                "query": query,
                "error": str(e),
                "answer": None,
                "results": [],
                "eval_results": None,
                "time_taken_ms": (time.time() - start_time) * 1000
            }