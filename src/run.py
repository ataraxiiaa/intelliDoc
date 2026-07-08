import os
from pipeline import Pipeline

# 1. Initialize the pipeline
pipeline = Pipeline()

# Use absolute paths relative to this script
base_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(base_dir, "..", "sample_data", "sample_text.pdf")

print("--- Step 1: Processing & Indexing PDF ---")
# Process and index page 0 of the sample PDF
indexing_result = pipeline.process_pdf_page(pdf_path, page_number=0)
print("Indexing Result:", indexing_result)

print("\n--- Step 2: Querying & RAG Synthesis ---")
# Query the document, providing a ground truth answer to trigger lexical evaluation
query = "What was the revenue and technical achievements in Q3?"
ground_truth = "Total revenue reached $1.2 billion and core databases were migrated to a distributed multi-region cluster with under 15ms latency."

query_result = pipeline.query_document(query, top_k=3, ground_truth=ground_truth)

if query_result["status"] == "success":
    print(f"Question: {query_result['query']}")
    print(f"Answer:   {query_result['answer']['answer']}")
    print(f"Confidence: {query_result['answer']['confidence']}")
    print(f"BLEU/ROUGE Metrics: {query_result['eval_results']}")
else:
    print("Query failed:", query_result["error"])
