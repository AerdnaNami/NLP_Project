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
