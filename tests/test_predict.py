import pytest

from src.model import create_models
from src.predict import predict_sentiment


@pytest.fixture
def trained_model():

    data = [
        "Adorei este produto",
        "Excelente experiência",
        "Estou muito satisfeito",
        "Foi fantástico",
        "Recomendo muito",

        "Produto horrível",
        "Experiência péssima",
        "Estou muito desiludido",
        "Não gostei nada",
        "Foi terrível",

        "O produto chegou ontem",
        "A reunião é amanhã",
        "A caixa é azul",
        "O equipamento tem duas versões",
        "Recebi uma encomenda"
    ]

    labels = [
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",

        "negative",
        "negative",
        "negative",
        "negative",
        "negative",

        "neutral",
        "neutral",
        "neutral",
        "neutral",
        "neutral"
    ]

    model = create_models()["Logistic Regression"]

    model.fit(data, labels)

    return model


def test_prediction_returns_valid_sentiment(trained_model):

    result = predict_sentiment(
        "Adorei este produto",
        trained_model
    )

    assert result["sentiment"] in [
        "positive",
        "negative",
        "neutral"
    ]


def test_probabilities_sum_to_one(trained_model):

    result = predict_sentiment(
        "O produto chegou ontem",
        trained_model
    )

    total = sum(result["probabilities"].values())

    assert total == pytest.approx(1.0)


def test_empty_text_raises_error(trained_model):

    with pytest.raises(ValueError):

        predict_sentiment("", trained_model)


def test_confidence_is_valid(trained_model):

    result = predict_sentiment(
        "Experiência fantástica",
        trained_model
    )

    assert 0 <= result["confidence"] <= 1