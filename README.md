# Naive Bayes Spam Classifier

A spam classifier built from scratch using Bayes' theorem — no ML libraries.

## How it works
Uses Naive Bayes: calculates the probability that a message is spam vs ham based on the words it contains. Built on the same conditional probability and Bayes' theorem concepts from high school mathematics.

## Dataset
SMS Spam Collection — 5,574 SMS messages labelled spam or ham.

## Usage
```bash
python train.py     # train and evaluate
python predict.py   # classify new messages
```

## Results
~98% accuracy on test set.

## Key concepts
- Bayes' theorem
- Laplace smoothing
- Log probabilities to prevent underflow