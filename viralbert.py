

!pip install datasets
!pip install transformers
!huggingface-cli login     

import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

# Loading the dataset
dataset = load_dataset('yummy456/viral_news_pairs')['train'].train_test_split(train_size=7000, test_size=3000)

# Loading the BERT tokenizer and the model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

def softmax(logits):
    return F.softmax(torch.tensor(logits), dim=-1).numpy()

# function to tokenize input
def tokenize_function(examples):
    return tokenizer(
        text=examples['title1'],
        text_pair=examples['title2'],
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

# Mapping the tokenization function to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Setting the format of the dataset for PyTorch
tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Spliting the dataset into train and test
train_dataset = tokenized_datasets['train']
test_dataset = tokenized_datasets['test']

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    # Retrieve predictions and true labels
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)

    # Calculate precision, recall, and F1-score (average='binary' assumes binary classification)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')

    # Return metrics as a dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Defining training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    learning_rate=1e-5,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Initializing the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Training the model
trainer.train()

# Evaluating the model
eval_results = trainer.evaluate()

print(f"Evaluation results: {eval_results}")