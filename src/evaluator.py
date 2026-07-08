import evaluate
from typing import List, Dict

class Evaluator:
    def __init__(self):
        self.bleu_metric = evaluate.load("sacrebleu")
        self.rouge_metric = evaluate.load("rouge")

    def compute_metrics_batch(self, results: List[Dict]) -> Dict:
        """
        Computes batch BLEU and ROUGE scores using the Hugging Face evaluate library.
        
        results: List of dicts, e.g., [{"predicted": "...", "ground_truth": "..."}]
        """
        if not results:
            return {}
            
        predictions = [item["predicted"] for item in results]
        references = [[item["ground_truth"]] for item in results]  # References must be a list of lists
        
        # Compute BLEU
        bleu_results = self.bleu_metric.compute(predictions=predictions, references=references)
        
        # Compute ROUGE
        # Flatten references for ROUGE
        flat_references = [item["ground_truth"] for item in results]
        rouge_results = self.rouge_metric.compute(predictions=predictions, references=flat_references)
        
        return {
            "bleu_score": round(bleu_results["score"], 4),  
            "rouge1": round(rouge_results["rouge1"], 4),
            "rouge2": round(rouge_results["rouge2"], 4),
            "rougeL": round(rouge_results["rougeL"], 4),
            "total_evaluated": len(results)
        }