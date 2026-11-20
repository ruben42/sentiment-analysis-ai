import json
import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.model import create_models


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "dataset.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"
METRICS_PATH = REPORT_DIR / "metrics.json"


def train():

    print("\n=== SENTIMENT ANALYSIS AI ===")

    # Carregar dados
    data = pd.read_csv(DATA_PATH)

    if data.empty or data["text"].isna().any():
        raise ValueError("O dataset contém dados inválidos.")

    X = data["text"].astype(str)
    y = data["sentiment"]

    # Separar treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    models = create_models()

    results = {}
    trained_models = {}

    print("\nA comparar algoritmos...\n")

    # Treinar e avaliar os modelos
    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        metrics = {
            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),
            "precision": float(
                precision_score(
                    y_test,
                    predictions,
                    average="macro",
                    zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_test,
                    predictions,
                    average="macro",
                    zero_division=0
                )
            ),
            "f1_score": float(
                f1_score(
                    y_test,
                    predictions,
                    average="macro",
                    zero_division=0
                )
            )
        }

        results[name] = metrics
        trained_models[name] = model

        print(f"\nModelo: {name}")
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        print(f"F1-score: {metrics['f1_score']:.2%}")

    # Selecionar melhor modelo pelo F1-score
    best_name = max(
        results,
        key=lambda name: results[name]["f1_score"]
    )

    best_model = trained_models[best_name]

    predictions = best_model.predict(X_test)

    # Relatório detalhado
    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    labels = ["negative", "neutral", "positive"]

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    # Guardar modelo
    joblib.dump(best_model, MODEL_PATH)

    # Guardar resultados
    output = {
        "best_model": best_name,
        "dataset_size": len(data),
        "training_size": len(X_train),
        "test_size": len(X_test),
        "models": results,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "labels": labels
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4, ensure_ascii=False)

    print("\n==============================")
    print(f"Melhor modelo: {best_name}")
    print("==============================")

    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    print(f"\nModelo guardado: {MODEL_PATH}")
    print(f"Relatório guardado: {METRICS_PATH}")


if __name__ == "__main__":
    train()