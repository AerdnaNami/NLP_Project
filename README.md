# NLP701_QA_Paper
This repository contains the code for NLP701 project

## Retrieval Comparison
In this step, we compare two paragraph retrievers: TF-IDF and BM25. 
Required packages:
- pandas
- numpy
- scikit-learn
- rank_bm25
- matplotlib
- datasets
- pickle

### Running Retrieval Experiment Script
1. Run the main script inside the experiments folder:
```bash
python retrieval_comparison.py
```

2. The script will:
   - Load the QASPER dataset
   - Process the test split into a suitable format
   - Train and save a TF-IDF vectorizer
   - Compare TF-IDF and BM25 retrieval methods
   - Generate performance plots

### Code Structure

- `create_qa_dataframe()`: Processes QASPER dataset into a pandas DataFrame
- `retrieve_from_paper_tfidf()`: Implements TF-IDF based retrieval
- `retrieve_from_paper_bm25()`: Implements BM25 based retrieval
- `calculate_metrics_for_k_range_*()`: Calculate performance metrics for each method
- `plot_combined_metrics()`: Visualizes comparison results
- `main()`: Orchestrates the complete workflow

### Metrics

The code evaluates retrieval performance using three metrics:
1. Precision@k: Proportion of retrieved paragraphs that are relevant
2. Recall@k: Proportion of relevant paragraphs that are retrieved
3. Accuracy@k: Whether at least one relevant paragraph is retrieved

### Visualization

The script generates a plot comparing TF-IDF and BM25 performance across different k values (number of retrieved paragraphs). The plot shows:
- Precision, Recall, and Accuracy curves for both methods
- Performance trends as k increases
- Direct comparison between TF-IDF and BM25

# BM25 + QA Pipeline with Model Comparison

This script implements a QA pipeline using BM25 for paragraph retrieval and transformer-based models for answer extraction. It evaluates the performance of three models: **RoBERTa**, **BERT**, and **DistilBERT**, using the **Qasper dataset**.

---
## Required Packages

- **`datasets`**: For loading the Qasper dataset.
- **`rank_bm25`**: Implements BM25 retrieval.
- **`transformers`**: HuggingFace library for QA pipelines.
- **`pandas`**: Data manipulation.
- **`nltk`**: Tokenization and preprocessing.
## Code Structure

### Functions:

- **`create_qa_dataframe()`**: Extracts questions, answers, and evidence from the Qasper dataset into a DataFrame.

- **`extract_full_papers()`**: Extracts and organizes full-text paragraphs from research papers.

- **`combine_questions_and_papers()`**: Merges the QA DataFrame with full paper content.

- **`process_bm25_qa()`**: Implements BM25 retrieval to rank paragraphs by relevance.

- **`process_qa()`**: Generates answers using a specified QA model and evaluates its confidence.

- **`compute_exact_match()`**: Calculates the proportion of perfect matches between predictions and ground truth.

- **`compute_f1()`**: Computes F1 scores to evaluate the overlap of predicted and true answers.

- **`evaluate_results()`**: Merges results and ground truth, computes evaluation metrics, and generates a detailed evaluation DataFrame.
## Model Comparison

The script compares three transformer-based models using the following metrics:

- **Exact Match (EM)**:
  - Measures the percentage of perfect matches between predictions and ground truth.

- **F1 Score**:
  - Evaluates the harmonic mean of precision and recall.

### Output:
- A summary DataFrame containing EM and F1 scores for each model.

## Results Visualization

The script prints a concise performance summary for the three models:

| **Model**      | **Exact Match (%)** | **F1 Score (%)** |
|-----------------|---------------------|------------------|
| **RoBERTa**    | X                   | X                |
| **BERT**       | Y                   | Y                |
| **DistilBERT** | Z                   | Z                |



