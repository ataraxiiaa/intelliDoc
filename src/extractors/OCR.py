import logging
from typing import Union, Optional, List, Dict, Any
from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np

# imgaug (used by paddleocr 2.9.1) uses np.sctypes which was removed in NumPy 2.0.
# Monkey-patch it so paddleocr can import without crashing.
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [bool, object, bytes, str, np.void],
    }

from paddleocr import PaddleOCR


class OCRExtractor:
    def __init__(self,lang: str = 'en',use_angle_cls: bool = True):
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.model = PaddleOCR(
            use_angle_cls=self.use_angle_cls,
            lang=self.lang,
        )

    @staticmethod
    def _load_image(image_path: Union[str,Path] = None, image: Image.Image = None) -> np.ndarray:
        if image_path:
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found at {image_path}")
            
            image = Image.open(image_path)

        return np.array(image)
    @staticmethod
    def _parse_ocr_results(ocr_result: List[List[Any]], min_confidence: float) -> tuple:
        """Parse PaddleOCR 2.x result format: List[List[[bbox, [text, conf]]]]"""
        if not ocr_result:
            return "", 0.0, []

        words = []
        confidences = []
        word_details = []

        for line in ocr_result:
            for item in line:
                bbox = item[0]
                text, confidence = item[1][0], item[1][1]
                if confidence >= min_confidence:
                    words.append(text)
                    confidences.append(confidence)
                    word_details.append({
                        "text": text,
                        "confidence": round(confidence, 4),
                        "bbox": bbox,
                    })

        extracted_text = " ".join(words)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return extracted_text, avg_confidence, word_details
        
        
    def _extract(self, image_path: Union[str, Path] = None, image: Image.Image = None):
        if image_path is None and image is None:
            raise ValueError("Both image_path and image are None")
        
        try:
            if image_path:
                img_array = self._load_image(image_path)
            else:
                img_array = np.array(image)
            
            result = self.model.ocr(img_array, cls=self.use_angle_cls)

            text, confidence, details = OCRExtractor._parse_ocr_results(
                result,
                min_confidence=0.5  
            )

            return {
                "extracted_text": text,
                "confidence": confidence,
                "details": details
            }

        except Exception as e:
            raise ValueError(f"Error extracting text from {image_path}: {str(e)}")
        


if __name__ == "__main__":
    import logging
 
    logging.basicConfig(level=logging.INFO)
 
    # Initialize extractor
    extractor = OCRExtractor(lang="en", use_angle_cls=True)
 
    # Example: Create a synthetic image with text and run OCR on it
    print("\n=== Demo: Synthetic Image OCR ===")
    try:
        from PIL import ImageDraw

        test_image = Image.new("RGB", (400, 80), color="white")
        draw = ImageDraw.Draw(test_image)
        draw.text((10, 20), "Q3 Revenue: $1.2 Billion", fill="black")

        result = extractor._extract(image=test_image)
        print(f"Extracted text: {result['extracted_text']}")
        print(f"Confidence:     {result['confidence']:.3f}")

    except Exception as e:
        print(f"Demo failed: {e}")
        print("PaddleOCR extractor is installed and ready to use.")
