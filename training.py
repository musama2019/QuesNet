import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import time
from transformers import AutoTokenizer, AdamW, BertForQuestionAnswering

# Load and preprocess training data
path_train = Path('squad/train-v2.0.json')
with open(path_train, 'rb') as f:
    squad_dict_train = json.load(f)

train_texts, train_queries, train_answers = [], [], []
for group in squad_dict_train['data']:
    for passage in group['paragraphs']:
        context = passage['context']
        for qa in passage['qas']:
            question = qa['question']
            for answer in qa['answers']:
                train_texts.append(context)
                train_queries.append(question)
                train_answers.append(answer)

# Load and preprocess validation data
path_val = Path('squad/dev-v2.0.json')
with open(path_val, 'rb') as f:
    squad_dict_val = json.load(f)

val_texts, val_queries, val_answers = [], [], []
for group in squad_dict_val['data']:
    for passage in group['paragraphs']:
        context = passage['context']
        for qa in passage['qas']:
            question = qa['question']
            for answer in qa['answers']:
                val_texts.append(context)
                val_queries.append(question)
                val_answers.append(answer)

# Adjust answer positions
def adjust_answer_positions(answers, texts):
    for answer, text in zip(answers, texts):
        real_answer = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(real_answer)

        if text[start_idx:end_idx] == real_answer:
            answer['answer_end'] = end_idx
        elif text[start_idx-1:end_idx-1] == real_answer:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1
        elif text[start_idx-2:end_idx-2] == real_answer:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2

adjust_answer_positions(train_answers, train_texts)
adjust_answer_positions(val_answers, val_texts)

# Tokenize using BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(train_texts, train_queries, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, val_queries, truncation=True, padding=True)

# Add token positions
def add_token_positions(encodings, answers):
    start_positions, end_positions = [], []
    count = 0

    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length

        if end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - 1)
            if end_positions[-1] is None:
                count += 1
                end_positions[-1] = tokenizer.model_max_length

    print(count)
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

add_token_positions(train_encodings, train_answers)
add_token_positions(val_encodings, val_answers)

# Create datasets and dataloaders
class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = SquadDataset(train_encodings)
val_dataset = SquadDataset(val_encodings)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

# Define model and optimizer
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)
optim = AdamW(model.parameters(), lr=5e-5)

# Train and evaluate
epochs = 3
whole_train_eval_time = time.time()
train_losses = []
val_losses = []
print_every = 1000

for epoch in range(epochs):
    epoch_time = time.time()
    
    model.train()
    loss_of_epoch = 0

    print("############ Train ############")

    for batch_idx, batch in enumerate(train_loader): 
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        loss.backward()
        optim.step()
        loss_of_epoch += loss.item()

        if (batch_idx+1) % print_every == 0:
            print("Batch {:} / {:}".format(batch_idx+1, len(train_loader)), "\nLoss:", round(loss.item(), 1), "\n")

    loss_of_epoch /= len(train_loader)
    train_losses.append(loss_of_epoch)

    # Evaluation
    model.eval()
    print("############ Evaluate ############")
    loss_of_epoch = 0

    for batch_idx, batch in enumerate(val_loader):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            loss = outputs[0]
            loss_of_epoch += loss.item()

        if (batch_idx+1) % print_every == 0:
            print("Batch {:} / {:}".format(batch_idx+1, len(val_loader)), "\nLoss:", round(loss.item(), 1), "\n")

    loss_of_epoch /= len(val_loader)
    val_losses.append(loss_of_epoch)

    print("\n------- Epoch ", epoch+1,
          " -------"
          "\nTraining Loss:", train_losses[-1],
          "\nValidation Loss:", val_losses[-1],
          "\nTime: ", (time.time() - epoch_time),
          "\n-----------------------",
          "\n\n")

print("Total training and evaluation time: ", (time.time() - whole_train_eval_time))

# Save model
torch.save(model, "finetunedmodel.pth")

