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
### 1. Training Arguments
The training is controlled by specific arguments defined using the TrainingArguments class. These arguments include:

Output Directory: Where the model checkpoints and results will be saved.
Evaluation Strategy: Set to evaluate the model at the end of every epoch.
Batch Sizes: A batch size of 8 for both training and evaluation.
Epochs: The model is trained for 10 epochs.
Learning Rate: A learning rate of 1e-5 for fine-tuning the BERT model.
Weight Decay: Regularization factor to avoid overfitting, set to 0.01.

### 2. Trainer Setup
The Hugging Face Trainer is used to simplify training, evaluation, and logging. It handles all aspects of the training process, including forward passes, backpropagation, and evaluation.

### 3. Evaluation Metrics
The primary metric used for evaluation is accuracy. The compute_metrics function calculates the accuracy based on the predictions:
Logits (raw model outputs) are converted to probabilities using the softmax function.
The predicted class is selected based on the maximum probability.
Accuracy is computed by comparing the predicted class with the true labels.

### 4. Model Training
The Trainer starts the training process using the train() method. The model is fine-tuned on the training dataset for 10 epochs.

### 5. Model Evaluation
After training, the model is evaluated on the test dataset using the evaluate() method, which outputs the final accuracy.


## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/basic-features/font-optimization) to automatically optimize and load Inter, a custom Google Font.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js/) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/deployment) for more details.
