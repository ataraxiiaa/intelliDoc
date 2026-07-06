import os
import pypdf
import pdfplumber
from typing import Dict, Any, Literal

class DocumentRouter:
    def __init__(self):
        pass

    def route_page(self,pdf_path: str, page_number: int) -> Dict[str, Any]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found{pdf_path}")

        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_number<0 or page_number>len(pdf.pages)-1:
                    raise ValueError(f"page_number must be between 0 and {len(pdf.pages)-1}")
                
                page = pdf.pages[page_number]
                text=page.extract_text()

                
                if not text or text.strip()=="":
                    return {
                        "route": "ocr",
                        "reason": "no text found",
                        "metadata": {}
                    }

                tables = page.find_tables()

                if tables:
                    extracted_tables = page.extract_tables()
                    return {
                        "route": "tables",
                        "reason": "tables found",
                        "metadata": {
                            "num_tables": len(extracted_tables)
                        }
                    }
                return {
                    "route": "text",
                    "reason": "text found",
                    "metadata": {
                        "num_chars": len(text)
                    }
                }
        except Exception as e:
            return {
                "route": "error",
                "reason": str(e)
            }

                

                
                