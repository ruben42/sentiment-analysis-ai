import joblib

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "sentiment_model.pkl"


def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Modelo não encontrado. Executa python -m src.train"
        )

    return joblib.load(MODEL_PATH)


def predict_sentiment(text, model):

    if not isinstance(text, str) or not text.strip():
        raise ValueError("O texto não pode estar vazio.")

    prediction = model.predict([text])[0]

    probabilities = model.predict_proba([text])[0]

    classes = model.classes_

    scores = {
        str(label): float(probability)
        for label, probability in zip(classes, probabilities)
    }

    confidence = max(scores.values())

    return {
        "sentiment": str(prediction),
        "confidence": confidence,
        "probabilities": scores
    }


if __name__ == "__main__":

    model = load_model()

    print("\n=== SENTIMENT ANALYSIS AI ===")
    print("Escreve 'sair' para terminar.\n")

    while True:

        text = input("A tua opinião: ").strip()

        if text.lower() == "sair":
            break

        try:

            result = predict_sentiment(text, model)

            print("\nSentimento:", result["sentiment"])
            print(
                "Confiança:",
                f"{result['confidence']:.2%}"
            )

        except ValueError as error:
            print(error)