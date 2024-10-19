# Viral News Article Prediction Model

## Project Overview
The project aims to predict the virality of news article titles using a binary classification model based on BERT (Bidirectional Encoder Representations from Transformers). Given two article titles, the model determines which is more likely to go viral. This task is performed by training a BERT-based model using pairs of article titles with a corresponding binary label indicating which is more viral. The project uses Hugging Face's transformers library for model training and evaluation, leveraging pre-trained BERT models for sequence classification.

## Dataset
The dataset, named viral_news_pairs, contains pairs of article titles and a binary label. Each data entry includes two titles (title1 and title2) and a corresponding label:

title1: The first news article title.
title2: The second news article title.
label: A binary value where 0 means that the first title (title1) is more viral, and 1 means that the second title (title2) is more viral.
The dataset used for this project was loaded from Hugging Face's datasets library using the path 'yummy456/viral_news_pairs'.

Train-test split: The dataset is split into 7,000 examples for training and 3,000 examples for testing, using the train_test_split function.

## Model Architecture
The project utilizes a pretrained BERT model for sequence classification:

Model: bert-base-uncased from Hugging Face’s transformers library.
Classification Task: A sequence classification task with 2 output labels (binary classification), determining which of the two article titles is more viral.
The architecture is based on BERT, which includes:

BERT Encoder: Pretrained on large corpora of text and fine-tuned in this project to classify pairs of article titles.
Classification Head: A linear layer on top of the BERT model outputs that maps the contextualized token representations to two output logits (one for each class).

## Model Training and Evaluation
### Training Arguments
The training is controlled by specific arguments defined using the TrainingArguments class. These arguments include:

Output Directory: Where the model checkpoints and results will be saved.
Evaluation Strategy: Set to evaluate the model at the end of every epoch.
Batch Sizes: A batch size of 8 for both training and evaluation.
Epochs: The model is trained for 10 epochs.
Learning Rate: A learning rate of 1e-5 for fine-tuning the BERT model.
Weight Decay: Regularization factor to avoid overfitting, set to 0.01.

### Trainer Setup
The Hugging Face Trainer is used to simplify training, evaluation, and logging. It handles all aspects of the training process, including forward passes, backpropagation, and evaluation.

### Evaluation Metrics
The primary metric used for evaluation is accuracy. The compute_metrics function calculates the accuracy based on the predictions:

Logits (raw model outputs) are converted to probabilities using the softmax function.
The predicted class is selected based on the maximum probability.
Accuracy is computed by comparing the predicted class with the true labels.

### Model Training
The Trainer starts the training process using the train() method. The model is fine-tuned on the training dataset for 10 epochs.

### Model Evaluation
After training, the model is evaluated on the test dataset using the evaluate() method, which outputs the final accuracy.
