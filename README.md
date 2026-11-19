# Sentiment Analysis AI

A Machine Learning project that classifies text opinions
into three sentiment categories: Positive, Negative and Neutral.

## Project Overview

This project uses Natural Language Processing (NLP)
and Machine Learning to analyze opinions written in Portuguese.

The model predicts the sentiment of a given sentence.

## Technologies

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Joblib

## Project Structure

sentiment-analysis-ai/
- data/
  - dataset.csv
- models/
- src/
  - train.py
  - predict.py
- requirements.txt
- README.md
- .gitignore

## Installation

Clone the repository:

    git clone https://github.com/ruben42/sentiment-analysis-ai.git

Enter the project folder:

    cd sentiment-analysis-ai

Install dependencies:

    pip install -r requirements.txt

## Train the Model

    python src/train.py

## Run Sentiment Analysis

    python src/predict.py

## Sentiment Categories

- Positive: expresses satisfaction or happiness.
- Negative: expresses dissatisfaction or frustration.
- Neutral: expresses no clear positive or negative opinion.

## Machine Learning Approach

The project uses:

1. TF-IDF for text vectorization.
2. Logistic Regression for sentiment classification.
3. Train/Test Split for model evaluation.
4. Accuracy and Classification Report for evaluation.

## Dataset

A small Portuguese dataset created for educational purposes.

The dataset can be expanded to improve the model's
ability to understand different opinions and expressions.

## Future Improvements

- Add a graphical user interface.
- Support English and other languages.
- Use a larger Portuguese sentiment dataset.
- Experiment with other Machine Learning algorithms.
- Deploy the application as a web service.

## Author

Rúben Pádua

## License

MIT License