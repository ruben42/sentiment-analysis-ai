import joblib

from pathlib import Path


# Localização do modelo
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "sentiment_model.pkl"


def predict_sentiment(text, model):

    prediction = model.predict([text])[0]

    probabilities = model.predict_proba([text])[0]

    confidence = max(probabilities) * 100

    return prediction, confidence


def main():

    print("\n================================")
    print("       SENTIMENT ANALYSIS AI")
    print("================================")

    # Verificar se o modelo existe
    if not MODEL_PATH.exists():

        print("\nModelo não encontrado!")

        print("Executa primeiro: python src/train.py")

        return

    # Carregar modelo
    model = joblib.load(MODEL_PATH)

    print("\nEscreve uma opinião para analisar.")

    print("Escreve 'sair' para terminar.\n")

    while True:

        text = input("A tua opinião: ").strip()

        if text.lower() == "sair":
            print("\nObrigado por utilizares a IA!")
            break

        if not text:
            print("Por favor, escreve uma opinião.")
            continue

        sentiment, confidence = predict_sentiment(
            text,
            model
        )

        print("\n--- RESULTADO ---")

        if sentiment == "positive":
            print("Sentimento: POSITIVO")

        elif sentiment == "negative":
            print("Sentimento: NEGATIVO")

        else:
            print("Sentimento: NEUTRO")

        print(f"Confiança do modelo: {confidence:.2f}%")

        print("-----------------\n")


if __name__ == "__main__":
    main()