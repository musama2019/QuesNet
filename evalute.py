

from datasets import load_metric
import torch
from transformers import AutoTokenizer, pipeline
from sklearn.metrics import f1_score, accuracy_score
from pathlib import Path
import json  # Add
from sklearn.model_selection import train_test_split


# Define the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Load the fine-tuned model
model_path = "finetunedmodel.pth"  # Assuming the model file is in the same directory
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

# Load the validation data
validation_data_path = "squad/dev-v2.0.json"  # Assuming the validation data file is in the same directory
path = Path(validation_data_path)

with open(path, 'rb') as f:
    squad_dict = json.load(f)

texts, queries, answers = [], [], []

# Extract passages, questions, and answers from the validation data
for group in squad_dict['data']:
    for passage in group['paragraphs']:
        context = passage['context']
        for qa in passage['qas']:
            question = qa['question']
            for answer in qa['answers']:
                texts.append(context)
                queries.append(question)
                answers.append(answer)

# Split the data into validation and test sets
val_texts, test_texts, val_queries, test_queries, val_answers, test_answers = train_test_split(
    texts, queries, answers, test_size=0.1, random_state=42
)

# Create the question-answering pipeline
qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)

# Evaluate on the test set
results = []

for text, question, answer in zip(test_texts, test_queries, test_answers):
    prediction = qa_pipeline({'context': text, 'question': question})
    results.append({'predicted_answer': prediction['answer'], 'true_answer': answer['text']})

# Compute metrics
def compute_metrics(predictions):
    true_answers = [item['true_answer'] for item in predictions]
    predicted_answers = [item['predicted_answer'] for item in predictions]

    # Compute Exact Match (EM)
    em_score = accuracy_score(true_answers, predicted_answers)

    # Compute F1 score
    f1 = f1_score(true_answers, predicted_answers, average='macro')

    return {'EM': em_score, 'F1': f1}

# Compute metrics on the test set
evaluation_metrics = compute_metrics(results)
print(evaluation_metrics)
