from datasets import load_dataset
from transformers import AutoTokenizer
import collections
import torch
from transformers import AutoModelForQuestionAnswering
import numpy as np
from tqdm.auto import tqdm
import evaluate
from transformers import TrainingArguments
from transformers import Trainer
import pandas as pd
from ast import literal_eval
from datasets import Dataset
from datasets import DatasetDict

bert_model = 'deepset/bert-large-uncased-whole-word-masking-squad2'
roberta_model = 'deepset/roberta-base-squad2'
distilbert_model = "distilbert/distilbert-base-cased-distilled-squad"
model_checkpoint = roberta_model

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
data_path = '/mnt/c/Users/imana/Desktop/Masters/NLP_project/NLP701_QA_Paper/data/extractive_dataset.csv'
data = pd.read_csv(data_path,  converters={'answer': literal_eval, 'evidence': literal_eval})

evidence_list = []
for rows in data.itertuples():
    evidence = rows.evidence[0]
    evidence_list.append(evidence)

data['evidence'] = evidence_list

def preprocess_answers(df):
    ans_list = []
    to_delete = []
    
    # Loop through the DataFrame to find answer locations and prepare to-delete list
    for index, row in df.iterrows():
        loc = row['evidence'].lower().find(row['answer'][0].lower())

        if loc != -1:
            ans_list.append({'answer_start': [loc], 'text': [row['answer'][0]]})
        else:
            to_delete.append(index)  # Use index for rows to be deleted
    
    return ans_list, to_delete

# Preprocess the answers and get rows to delete
answer_list, del_q = preprocess_answers(data)

# Drop the rows only after processing
data = data.drop(del_q).reset_index(drop=True)  # Drop and reset index

# Ensure the lengths of the DataFrame and answer_list are now aligned
if len(answer_list) == len(data):
    data['answer'] = answer_list
else:
    raise ValueError(f"Length mismatch: {len(answer_list)} answers and {len(data)} rows in data.")

data_train = Dataset.from_pandas(data).train_test_split(test_size=0.2)
data_dev = data_train["test"].train_test_split(test_size=0.5)

data = DatasetDict({
    'train': data_train["train"],
    'test': data_dev["test"],
    'validation': data_dev["train"]
})
print(data["train"][0])

context = data["train"][0]['evidence']
question = data["train"][0]['question']
# print(context)
# print(question)

# %%
inputs = tokenizer(
    data["train"][:]["question"],
    data["train"][:]["evidence"],
    question,
    context,
    max_length=100,
    truncation="only_second",       # truncate the context when the question is too long 
    stride=50,                      # set the number of overlapping tokens to 50 between two successive chunks  
    return_overflowing_tokens=True, 
    return_offsets_mapping=True,
)

# for ids in inputs["input_ids"]:
#     print(tokenizer.decode(ids))

answers = data["train"][:]["answer"]
start_positions = []
end_positions = []

for i, offset in enumerate(inputs["offset_mapping"]):
    sample_idx = inputs["overflow_to_sample_mapping"][i]
    answer = answers[sample_idx]
    start_char = answer["answer_start"][0]
    end_char = answer["answer_start"][0] + len(answer["text"][0])
    sequence_ids = inputs.sequence_ids(i)

    # Find the start and end of the context
    idx = 0
    while sequence_ids[idx] != 1:
        idx += 1
    context_start = idx
    while sequence_ids[idx] == 1:
        idx += 1
    context_end = idx - 1

    # If the answer is not fully inside the context, label is (0, 0)
    if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
        start_positions.append(0)
        end_positions.append(0)
    else:
        # Otherwise it's the start and end token positions
        idx = context_start
        while idx <= context_end and offset[idx][0] <= start_char:
            idx += 1
        start_positions.append(idx - 1)

        idx = context_end
        while idx >= context_start and offset[idx][1] >= end_char:
            idx -= 1
        end_positions.append(idx + 1)

# print(start_positions, end_positions)

max_length = 386
stride = 128

def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["evidence"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answer"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

train_dataset = data["train"].map(
    preprocess_training_examples,
    batched=True,
    remove_columns=data["train"].column_names,
)
len(data["train"]), len(train_dataset)

def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["evidence"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

validation_dataset = data["validation"].map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=data["validation"].column_names,
)

print(len(data["validation"]), len(validation_dataset))

small_eval_set = data["validation"].select(range(2))
trained_checkpoint = "deepset/roberta-base-squad2"

tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint)
eval_set = small_eval_set.map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=data["validation"].column_names,
)

small_eval_set = data["validation"].select(range(2))
trained_checkpoint = "deepset/roberta-base-squad2"

tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint)
eval_set = small_eval_set.map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=data["validation"].column_names,
)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

eval_set_for_model = eval_set.remove_columns(["example_id", "offset_mapping"])
eval_set_for_model.set_format("torch")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
batch = {k: eval_set_for_model[k].to(device) for k in eval_set_for_model.column_names}
trained_model = AutoModelForQuestionAnswering.from_pretrained(trained_checkpoint).to(
    device
)

with torch.no_grad():
    outputs = trained_model(**batch)

start_logits = outputs.start_logits.cpu().numpy()
end_logits = outputs.end_logits.cpu().numpy()


example_to_features = collections.defaultdict(list)
for idx, feature in enumerate(eval_set):
    example_to_features[feature["example_id"]].append(idx)

n_best = 20
max_answer_length = 100
predicted_answers = []

for example in small_eval_set:
    example_id = example["id"]
    context = example["evidence"]
    answers = []

    for feature_index in example_to_features[example_id]:
        start_logit = start_logits[feature_index]
        end_logit = end_logits[feature_index]
        offsets = eval_set["offset_mapping"][feature_index]

        start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
        end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
        for start_index in start_indexes:
            for end_index in end_indexes:
                # Skip answers that are not fully in the context
                if offsets[start_index] is None or offsets[end_index] is None:
                    continue
                # Skip answers with a length that is either < 0 or > max_answer_length.
                if (
                    end_index < start_index
                    or end_index - start_index + 1 > max_answer_length
                ):
                    continue

                answers.append(
                    {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                )

    best_answer = max(answers, key=lambda x: x["logit_score"])
    predicted_answers.append({"id": str(example_id), "prediction_text": best_answer["text"]})

metric = evaluate.load("squad")

theoretical_answers = [
    {"id": str(ex["id"]), "answers": ex["answer"]} for ex in small_eval_set
]

print(metric.compute(predictions=predicted_answers, references=theoretical_answers))

def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["evidence"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": str(example_id), "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": str(example_id), "prediction_text": ""})

    theoretical_answers = [{"id": str(ex["id"]), "answers": ex["answer"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

compute_metrics(start_logits, end_logits, eval_set, small_eval_set)

args = TrainingArguments(
    output_dir="roberta-finetuned-paperQA",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    logging_dir='/mnt/c/Users/imana/Desktop/Masters/NLP_project/NLP701_QA_Paper/train/logs',
    logging_steps=1,
    report_to="tensorboard",
    warmup_steps=500
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
)
trainer.train()

predictions, _, _ = trainer.predict(validation_dataset)
start_logits, end_logits = predictions
compute_metrics(start_logits, end_logits, validation_dataset, data["validation"])