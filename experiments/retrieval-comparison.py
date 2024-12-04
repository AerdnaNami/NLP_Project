import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pickle
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def create_qa_dataframe(ds, split):
    """
    Create a DataFrame from the QASPER dataset containing questions, answers, and relevant paragraphs.
    
    Args:
        ds: QASPER dataset object
        split: Dataset split to use ('train', 'validation', or 'test')
    
    Returns:
        DataFrame containing processed QA pairs and relevant paragraphs
    """
    titles, questions, answers = [], [], []
    list_of_full_papers, relevant_paragraphs = [], []

    for id in range(len(ds[split])):
        # Extract paragraphs from each section
        paragraphs = []
        title = ds[split]['title'][id]
        for section in ds[split][id]['full_text']['paragraphs']:
            for paragraph in section:
                paragraphs.append(paragraph)

        # Process each question and its answers
        for i in range(len(ds[split][id]['qas']['question'])):
            for j in range(len(ds[split][id]['qas']['answers'][i]['answer'])):
                if len(ds[split][id]['qas']['answers'][i]['answer'][j]['extractive_spans']) > 0:
                    titles.append(title)
                    questions.append(ds[split][id]['qas']['question'][i])
                    answers.append(ds[split][id]['qas']['answers'][i]['answer'][j]['extractive_spans'])
                    list_of_full_papers.append(paragraphs)
                    relevant_paragraphs.append(ds[split][id]['qas']['answers'][i]['answer'][j]['evidence'])

    return pd.DataFrame({
        'title': titles,
        'question': questions,
        'answer': answers,
        'full_paper': list_of_full_papers,
        'relevant_paragraphs': relevant_paragraphs
    })

def retrieve_from_paper_tfidf(question, paper_text, tfidf_vectorizer, k):
    """
    Retrieve relevant paragraphs using TF-IDF and cosine similarity.
    """
    paragraphs = [p for p in paper_text if p.strip()]
    paragraph_tfidf = tfidf_vectorizer.transform(paragraphs)
    question_vec = tfidf_vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, paragraph_tfidf).flatten()
    ranked_indices = similarities.argsort()[::-1]
    return [(paragraphs[i], similarities[i]) for i in ranked_indices[:k]]

def retrieve_from_paper_bm25(question, paper_paragraphs, k=5):
    """
    Retrieve relevant paragraphs using BM25 ranking.
    """
    tokenized_paragraphs = [paragraph.split() for paragraph in paper_paragraphs]
    bm25 = BM25Okapi(tokenized_paragraphs)
    tokenized_question = question.split()
    scores = bm25.get_scores(tokenized_question)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(paper_paragraphs[i], scores[i]) for i in ranked_indices]

def calculate_metrics_for_k_range_tfidf(test_data, tfidf_vectorizer, k_values):
    """
    Calculate precision, recall, and accuracy metrics for TF-IDF retrieval.
    """
    metrics = {metric: [] for metric in ['precision', 'recall', 'accuracy']}
    total_questions = len(test_data)
    
    for k in k_values:
        k_metrics = {metric: 0 for metric in ['precision', 'recall', 'accuracy']}
        
        for _, row in test_data.iterrows():
            ranked_paragraphs = retrieve_from_paper_tfidf(row['question'], row['full_paper'], 
                                                        tfidf_vectorizer, k)
            retrieved_paragraphs = [para[0] for para in ranked_paragraphs]
            relevant_retrieved = [para for para in retrieved_paragraphs 
                                if para in row['relevant_paragraphs']]
            
            k_metrics['precision'] += len(relevant_retrieved) / k
            k_metrics['recall'] += (len(relevant_retrieved) / len(row['relevant_paragraphs']) 
                                  if len(row['relevant_paragraphs']) > 0 else 0)
            k_metrics['accuracy'] += int(any(para in row['relevant_paragraphs'] 
                                           for para in retrieved_paragraphs))
        
        for metric in k_metrics:
            metrics[metric].append(k_metrics[metric] / total_questions)
    
    return metrics['precision'], metrics['recall'], metrics['accuracy']

def calculate_metrics_for_k_range_bm25(test_data, k_values):
    """
    Calculate precision, recall, and accuracy metrics for BM25 retrieval.
    """
    metrics = {metric: [] for metric in ['precision', 'recall', 'accuracy']}
    total_questions = len(test_data)
    
    for k in k_values:
        k_metrics = {metric: 0 for metric in ['precision', 'recall', 'accuracy']}
        
        for _, row in test_data.iterrows():
            try:
                ranked_paragraphs = retrieve_from_paper_bm25(row['question'], row['full_paper'], k)
                retrieved_paragraphs = [para[0] for para in ranked_paragraphs]
                relevant_retrieved = [para for para in retrieved_paragraphs 
                                    if para in row['relevant_paragraphs']]
                
                k_metrics['precision'] += len(relevant_retrieved) / k
                k_metrics['recall'] += (len(relevant_retrieved) / len(row['relevant_paragraphs']) 
                                      if len(row['relevant_paragraphs']) > 0 else 0)
                k_metrics['accuracy'] += int(any(para in row['relevant_paragraphs'] 
                                               for para in retrieved_paragraphs))
            except ValueError as e:
                print(f"Error processing row: {e}")
                continue
        
        for metric in k_metrics:
            metrics[metric].append(k_metrics[metric] / total_questions)
    
    return metrics['precision'], metrics['recall'], metrics['accuracy']

def plot_combined_metrics(test_data, tfidf_vectorizer, k_values):
    """
    Plot comparison of TF-IDF and BM25 retrieval metrics.
    """
    # Calculate metrics
    tfidf_metrics = calculate_metrics_for_k_range_tfidf(test_data, tfidf_vectorizer, k_values)
    bm25_metrics = calculate_metrics_for_k_range_bm25(test_data, k_values)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'orange', 'purple']
    metrics = ['Precision', 'Recall', 'Accuracy']
    
    for i, (tfidf_metric, bm25_metric) in enumerate(zip(tfidf_metrics, bm25_metrics)):
        plt.plot(k_values, tfidf_metric, label=f'TF-IDF {metrics[i]}@k',
                marker='o', color=colors[i])
        plt.plot(k_values, bm25_metric, label=f'BM25 {metrics[i]}@k',
                marker='s', linestyle='--', color=colors[i])
    
    plt.xlabel('k (Number of Retrieved Paragraphs)')
    plt.ylabel('Score')
    plt.title('Comparison of TF-IDF and BM25 Retrieval Metrics')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Load QASPER dataset
    print("Loading QASPER dataset...")
    ds = load_dataset("allenai/qasper")
    
    # Create test DataFrame
    print("Processing test split...")
    test_data = create_qa_dataframe(ds, 'test')
    
    # Initialize and fit TF-IDF vectorizer
    print("Training TF-IDF vectorizer...")
    tfidf_vectorizer = TfidfVectorizer()
    all_paragraphs = [p for paper in test_data['full_paper'] for p in paper if p.strip()]
    tfidf_vectorizer.fit(all_paragraphs)
    
    # Save vectorizer
    print("Saving TF-IDF vectorizer...")
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    
    # Compare methods
    print("Comparing retrieval methods...")
    k_values = list(range(1, 11))
    plot_combined_metrics(test_data, tfidf_vectorizer, k_values)

if __name__ == "__main__":
    main()
