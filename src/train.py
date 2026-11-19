import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# Paths do projeto
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"


def train_model():

    print("\n=== SENTIMENT ANALYSIS AI ===")
    print("A carregar os dados...\n")

    # Carregar dataset
    data = pd.read_csv(DATA_PATH)

    X = data["text"]
    y = data["sentiment"]

    # Dividir dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # Criar pipeline de Machine Learning
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2)
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    # Treinar modelo
    print("A treinar a inteligência artificial...")

    model.fit(X_train, y_train)

    # Fazer previsões
    predictions = model.predict(X_test)

    # Avaliar modelo
    accuracy = accuracy_score(y_test, predictions)

    print("\n=== RESULTADOS ===")

    print(f"Accuracy: {accuracy:.2%}")

    print("\nRelatório de classificação:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # Guardar modelo
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print("\nModelo guardado com sucesso!")
    print(f"Localização: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()