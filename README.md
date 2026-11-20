# 🧠 Sentiment Analysis AI

### Portuguese Natural Language Processing | Machine Learning

A Machine Learning project that automatically classifies Portuguese text into three sentiment categories:

- 😊 Positive
- 😐 Neutral
- 😞 Negative

The application uses Natural Language Processing (NLP), TF-IDF and Machine Learning algorithms to analyze opinions and predict their sentiment.

---

## 🚀 Features

- Sentiment classification in Portuguese.
- Logistic Regression and Multinomial Naive Bayes comparison.
- Automatic model selection using Macro F1-score.
- Individual sentiment prediction.
- Probability visualization.
- Batch sentiment analysis using CSV files.
- Interactive Streamlit dashboard.
- Model evaluation and confusion matrix.
- Automated tests with Pytest.

## 🛠️ Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| TF-IDF | Text vectorization |
| Logistic Regression | Classification |
| Multinomial Naive Bayes | Classification |
| Streamlit | Web interface |
| Plotly | Interactive charts |
| Joblib | Model persistence |
| Pytest | Automated testing |

## 📂 Project Structure

```text
sentiment-analysis-ai/
│
├── app.py
├── data/
│   └── dataset.csv
├── models/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
├── tests/
│   ├── __init__.py
│   └── test_predict.py
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/ruben42/sentiment-analysis-ai.git
```

Enter the project directory:

```bash
cd sentiment-analysis-ai
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🧠 Train the Models

Run:

```bash
python -m src.train
```

The training pipeline:

1. Loads the Portuguese dataset.
2. Splits the data into training and testing sets.
3. Trains Logistic Regression and Multinomial Naive Bayes.
4. Evaluates both models.
5. Selects the best model using Macro F1-score.
6. Saves the trained model.
7. Generates evaluation reports.

## 💻 Run the Application

After training:

```bash
streamlit run app.py
```

Open the local URL provided by Streamlit.

## 📊 Model Evaluation

The project evaluates models using:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion Matrix

Evaluation results are saved in:

```text
reports/metrics.json
```

## 🧪 Run Automated Tests

```bash
pytest -v
```

## 📁 Dataset

The initial dataset contains Portuguese example sentences classified into:

- Positive
- Negative
- Neutral

It is a small educational dataset and should not be considered representative of real-world Portuguese language usage.

## 🔮 Future Improvements

- Expand the Portuguese training dataset.
- Add real-world labeled data.
- Improve sarcasm and irony detection.
- Explore transformer-based NLP models.
- Add multilingual support.
- Deploy the application online.
- Add continuous integration with GitHub Actions.

## ⚠️ Limitations

The model uses a small educational dataset.

Predictions may be inaccurate when dealing with:
- Sarcasm
- Irony
- Complex expressions
- Ambiguous sentences
- Context-dependent opinions

Predicted probabilities are model estimates and should not be interpreted as guaranteed confidence.

## 👨‍💻 Author

Rúben Pádua

GitHub: https://github.com/ruben42

## 📄 License

This project is licensed under the MIT License.