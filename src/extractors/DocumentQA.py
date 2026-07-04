from os import stat
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from typing import List, Optional, Dict, Any

class DocumentQAExtractor:
    def __init__(self, model_name="deepset/roberta-base-squad2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    def _validate_input(self,input: str) -> bool:
        if not input or not input.strip():
            return False
        
        return True
    def _extract(self,question: str, text: str) -> dict:
        if not self._validate_input(question):
            raise ValueError("Question is Empty.")
        
        if not text or not text.strip():
            raise ValueError("Text is Empty.")
        
        if len(text) > 3000:
            print("Extracting from chunks...")
            return self._extract_from_long_text(question,text)
        else:
            print("Extracting from single text...")
            return self._extract_answer(question,text)

    def _extract_answer(self, question: str, context: str) -> dict:
                
        try:
            inputs = self.tokenizer(
                question, 
                context, 
                return_tensors="pt"
            )
        
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            answer_start = outputs.start_logits.argmax()
            answer_end = outputs.end_logits.argmax() + 1

            if answer_end < answer_start:
                return {"answer": "", "score" : 0.0, "error" : "invalid span end < start"}
            

            answer = self.tokenizer.decode(
                inputs["input_ids"][0][answer_start:answer_end],
                skip_special_tokens=True,
            )
            score = float(
                outputs.start_logits.softmax(-1)[0, answer_start]
                * outputs.end_logits.softmax(-1)[0, answer_end - 1]
            )
            if not answer:
                return {"answer": "", "score": 0.0, "error": "No answer found"}

            return {"answer": answer,"score":round(score,2),"error": None}
        except Exception as e:
            raise ValueError(f"Error extracting answer: {str(e)}")

    def _chunk_text(self, text: str, chunk_size: int = 1500, overlap: int = 100) -> List[str]:
        if not self._validate_input(text):
            raise ValueError("Text is Empty.")
        chunks = []
        step = chunk_size - overlap
 
        for i in range(0, len(text), step):
            chunk = text[i : i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
 
        return chunks
 


    def _extract_from_long_text(self,question: str,text: str) -> dict:
        best_result = None
        best_confidence = 0.0

        try:
            if len(text) > 3000:
                chunks = self._chunk_text(text)

            for chunk in chunks:
                try:
                    result = self._extract_answer(question,chunk)
                    confidence = result["score"]

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_result = result
                    
                except Exception as e:
                    raise ValueError(f"Error extracting answer:{str(e)}")

        except Exception as e:
            raise ValueError(f"Error: {str(e)}")
        
        if best_result is None:
            return {"answer": "", "score": 0.0, "error": "No answer found in any chunk"}
        return best_result

    def batch_extract(self,texts: List[str], question: str) -> List[dict]:
        if not self._validate_input(question):
            raise ValueError("Question is Empty.")
        
        if not texts:
            raise ValueError("Texts are Empty.")
        
        results = []
        for text in texts:
            try:
                result = self._extract(question,text)
                results.append(result)
            except Exception as e:
                raise ValueError(f"Error: {str(e)}")
        
        return results

if __name__ == "__main__":
    import logging
 
    logging.basicConfig(level=logging.INFO)
 
    # Initialize extractor
    extractor = DocumentQAExtractor()
 
    # Example 1: Simple extraction
    text = """
    Q3 2024 was a strong quarter for our company. Total revenue reached $1.2 billion,
    with product revenue contributing $800 million and services contributing $400 million.
    This represents a 26% increase compared to Q2 2024. The growth was driven by
    expansion in our cloud services division.
    """
 
    question = "What was the Q3 revenue?"
    result = extractor._extract(question, text)
 
    print("\n=== Example 1: Simple Extraction ===")
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['score']:.3f}")
    print(f"Error: {result['error']}")
 
    # Example 2: Low confidence handling
    question_hard = "What was the Q5 revenue?"
    result_hard = extractor._extract(question_hard, text)
 
    print("\n=== Example 2: Unanswerable Question ===")
    print(f"Question: {question_hard}")
    print(f"Answer: {result_hard['answer']}")
    print(f"Confidence: {result_hard['score']:.3f}")
    print(f"Error: {result_hard['error']}")
 
    # Example 3: Long text handling
    long_text = text * 50  # Make it very long
    question_long = "What was the Q3 revenue?"
    result_long = extractor._extract(question_long, long_text)
 
    print("\n=== Example 3: Long Text ===")
    print(f"Text length: {len(long_text)} chars")
    print(f"Answer: {result_long['answer']}")
    print(f"Confidence: {result_long['score']:.3f}")
    print(f"Error: {result_long['error']}")