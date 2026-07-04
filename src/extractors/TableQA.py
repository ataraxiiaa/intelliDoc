import logging
from typing import List, Optional, Dict, Any, Union
import pandas as pd
from transformers import pipeline, Pipeline


class TableQAExtractor:
    def __init__(self,model_name="google/tapas-base-finetuned-wtq"):
        self.model = pipeline(
            "table-question-answering",
            model=model_name
        )


    def _validate_input(self,question: str, table: Union[pd.DataFrame, List[List[Any]]]) -> bool:
        if not question or not question.strip():
            return False
        
        if isinstance(table,pd.DataFrame):
            if table.empty:
                return False
        else:
            if not table:
                return False
        return True

    @staticmethod
    def _list_to_dataframe(table: List[List[Any]]) -> pd.DataFrame:
        headers = table[0]
        rows = table[1]
        return pd.DataFrame(rows,columns=headers)

    def _extract(self,question: str,table: Union[pd.DataFrame, List[List[str]]]) -> dict:
        if not self._validate_input(question,table):
            raise ValueError("Invalid Input.")
        
        try:
            if isinstance(table, list):
                df = self._list_to_dataframe(table)
            else:
                df = table.copy()

            df = df.astype(str)
            result = self.model(table=df,query=question)
            confidence = result.get("score",0.0)

            answer = result["answer"]    
            return {"answer": answer,"score":round(confidence,2),}
                
        except Exception as e:
            raise ValueError(f"Error: {str(e)}")

        

        
if __name__ == "__main__":
    import logging
 
    logging.basicConfig(level=logging.INFO)
 
    # Initialize extractor
    extractor = TableQAExtractor()
 
    # Example 1: Simple table extraction
    df = pd.DataFrame({
        "Quarter": ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"],
        "Product Revenue": ["$600M", "$700M", "$800M", "$850M"],
        "Service Revenue": ["$300M", "$350M", "$400M", "$420M"],
        "Total": ["$900M", "$1.05B", "$1.2B", "$1.27B"],
    })
 
    question = "What was the Q3 total revenue?"
    result = extractor._extract(table=df, question=question)
    print(result)
    print("\n=== Example 1: Simple Table ===")
    print(f"Question: {question}")
    print(f"Answer: {result["answer"]}")
    print(f"Confidence: {result["score"]:.3f}")
 
    # Example 2: Wide table (should auto-reduce)
    wide_df = pd.DataFrame({
        f"Col{i}": [f"Value{i}{j}" for j in range(5)]
        for i in range(15)
    })
    wide_df.insert(0, "ID", ["Row1", "Row2", "Row3", "Row4", "Row5"])
 
    question_wide = "What is Row1 value?"
    result_wide = extractor._extract(table=wide_df, question=question_wide)
 
    print("\n=== Example 2: Wide Table (15 cols, auto-reduced) ===")
    print(f"Original shape: {wide_df.shape}")
    print(f"Question: {question_wide}")
    print(f"Answer: {result_wide["answer"]}")
 
    # Example 3: Empty table (should fail gracefully)
    empty_df = pd.DataFrame()
    result_empty = extractor._extract(table=empty_df, question="Any question")
 
    print("\n=== Example 3: Empty Table (Error Handling) ===")
    print(f"Success: {result_empty}")
    print(f"Error: {result_empty}")
                