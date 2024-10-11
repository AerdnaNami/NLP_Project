import numpy as np
from datasets import load_metric
from transformers import pipeline
import pandas as pd
from ast import literal_eval

# Load the F1 score metric from Hugging Face's datasets
metric = load_metric("squad")
model_checkpoint = 'ImanAndrea/roberta-finetuned-paperQA' # make public
question_answerer = pipeline("question-answering", model=model_checkpoint, device=0)
test = pd.read_csv('test_explicit_extractive.csv', converters={'answer': literal_eval, 'evidence': literal_eval})

# Function to compute the F1 score between two texts
def compute_f1(prediction, ground_truth):
    # Normalize by removing extra whitespace and lowercasing
    pred_tokens = prediction.strip().lower().split()
    truth_tokens = ground_truth.strip().lower().split()
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    if len(common_tokens) == 0:
        return 0
    
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(truth_tokens)
    
    if precision + recall == 0:
        return 0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

#list of predictions and ground truth answers
ground_truths = []

for index, row in test.iterrows():
    ground_truths.append({'id': row['id'],'question': row['question'], 'answer': row['answer'][0], 'evidence': row['evidence'][0]})

predictions = []

for index, row in test.iterrows():
    answer = question_answerer(question=row['question'], context=row['evidence'][0])
    predictions.append({'id': row['id'], 'question': row['question'], 'pred_answer': answer['answer'], 'evidence': row['evidence'][0]})

for pred, truth in zip(predictions, ground_truths):
    print('predicted answer:', pred['pred_answer'])
    print('ground truth answer:', truth['answer'])


# Extract the prediction text and ground truth for each sample
predicted_answers = [pred["pred_answer"] for pred in predictions]
true_answers = [truth["answer"] for truth in ground_truths]

# Compute F1 scores for each prediction
f1_scores = []
for pred, true in zip(predicted_answers, true_answers):
    f1 = compute_f1(pred, true)
    f1_scores.append(f1)

# Print F1 scores for each prediction
# for i, f1 in enumerate(f1_scores):
#     print(f"Sample {i+1} - F1 Score: {f1}")

# Optionally, compute the average F1 score across all predictions
average_f1 = np.mean(f1_scores)
print(f"Average F1 Score: {average_f1}")


# Exact match score
def compute_exact_match(prediction, ground_truth):
    return int(prediction.strip() == ground_truth.strip())

predicted_answers = [pred["pred_answer"] for pred in predictions]
true_answers = [truth["answer"] for truth in ground_truths]

em_scores = []
for pred, true in zip(predicted_answers, true_answers):
    em = compute_exact_match(pred, true)
    em_scores.append(em)

# Compute the overall Exact Match score as a percentage
exact_match_percentage = np.mean(em_scores) * 100
print(f"Exact Match Score: {exact_match_percentage}%")