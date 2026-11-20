import json
import pandas as pd
import streamlit as st
import plotly.express as px

from pathlib import Path

from src.predict import load_model, predict_sentiment


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "sentiment_model.pkl"
METRICS_PATH = BASE_DIR / "reports" / "metrics.json"


# Configuração da página
st.set_page_config(
    page_title="Sentiment Analysis AI",
    page_icon="🧠",
    layout="wide"
)


# Estilo visual
st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #e2e8f0;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# Carregar modelo
@st.cache_resource
def get_model():
    return load_model()


# Carregar métricas
@st.cache_data
def get_metrics():

    if not METRICS_PATH.exists():
        return None

    with open(METRICS_PATH, encoding="utf-8") as file:
        return json.load(file)


# Título
st.title("🧠 Sentiment Analysis AI")

st.caption(
    "Portuguese Natural Language Processing | Machine Learning"
)

st.markdown("""
Esta aplicação utiliza Inteligência Artificial para analisar
opiniões escritas em português e classificá-las em três categorias.
""")

st.divider()


# Verificar existência do modelo
if not MODEL_PATH.exists():

    st.error(
        "Modelo não encontrado. Executa primeiro: "
        "`python -m src.train`"
    )

    st.stop()


model = get_model()
metrics = get_metrics()


# Separadores
tab1, tab2, tab3 = st.tabs([
    "🔍 Analisar opinião",
    "📊 Estatísticas",
    "📁 Análise em lote"
])


# ======================================
# TAB 1 - ANÁLISE INDIVIDUAL
# ======================================

with tab1:

    st.subheader("Analisa uma opinião")

    text = st.text_area(
        "Escreve aqui a tua opinião:",
        placeholder="Exemplo: Adorei este produto!",
        height=130
    )

    if st.button("Analisar sentimento", type="primary"):

        if not text.strip():

            st.warning("Introduz uma opinião primeiro.")

        else:

            result = predict_sentiment(text, model)

            sentiment = result["sentiment"]

            colors = {
                "positive": "green",
                "negative": "red",
                "neutral": "gray"
            }

            icons = {
                "positive": "😊",
                "negative": "😞",
                "neutral": "😐"
            }

            labels = {
                "positive": "POSITIVO",
                "negative": "NEGATIVO",
                "neutral": "NEUTRO"
            }

            st.divider()

            st.subheader("Resultado")

            st.markdown(
                f"## {icons[sentiment]} {labels[sentiment]}"
            )

            st.metric(
                "Probabilidade estimada da classe prevista",
                f"{result['confidence']:.2%}"
            )

            st.caption(
                "A confiança é uma estimativa do modelo, "
                "não uma garantia de correção."
            )

            st.subheader("Probabilidades por sentimento")

            probability_data = pd.DataFrame({
                "Sentimento": [
                    "Positivo",
                    "Negativo",
                    "Neutro"
                ],
                "Probabilidade": [
                    result["probabilities"].get("positive", 0),
                    result["probabilities"].get("negative", 0),
                    result["probabilities"].get("neutral", 0)
                ]
            })

            fig = px.bar(
                probability_data,
                x="Sentimento",
                y="Probabilidade",
                color="Sentimento",
                color_discrete_map={
                    "Positivo": "#16a34a",
                    "Negativo": "#dc2626",
                    "Neutro": "#64748b"
                },
                range_y=[0, 1],
                title="Distribuição das probabilidades"
            )

            st.plotly_chart(fig, use_container_width=True)


# ======================================
# TAB 2 - ESTATÍSTICAS
# ======================================

with tab2:

    st.subheader("Avaliação do Machine Learning")

    if metrics is None:

        st.warning("Ainda não existem métricas. Treina o modelo.")

    else:

        best_model = metrics["best_model"]

        best_metrics = metrics["models"][best_model]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Accuracy",
            f"{best_metrics['accuracy']:.2%}"
        )

        col2.metric(
            "Precision",
            f"{best_metrics['precision']:.2%}"
        )

        col3.metric(
            "Recall",
            f"{best_metrics['recall']:.2%}"
        )

        col4.metric(
            "F1-score",
            f"{best_metrics['f1_score']:.2%}"
        )

        st.info(f"Melhor modelo: {best_model}")

        st.caption(
            f"Dataset: {metrics['dataset_size']} exemplos | "
            f"Treino: {metrics['training_size']} | "
            f"Teste: {metrics['test_size']}"
        )

        # Comparação dos algoritmos
        comparison = pd.DataFrame(
            metrics["models"]
        ).T.reset_index()

        comparison = comparison.rename(
            columns={"index": "Modelo"}
        )

        st.subheader("Comparação dos modelos")

        fig = px.bar(
            comparison,
            x="Modelo",
            y="f1_score",
            color="Modelo",
            title="Comparação de F1-score"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Matriz de confusão
        st.subheader("Matriz de confusão")

        matrix = pd.DataFrame(
            metrics["confusion_matrix"],
            index=metrics["labels"],
            columns=metrics["labels"]
        )

        fig = px.imshow(
            matrix,
            text_auto=True,
            color_continuous_scale="Blues",
            labels={
                "x": "Previsão",
                "y": "Valor real",
                "color": "Quantidade"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

        # Classification report
        st.subheader("Relatório de classificação")

        report = pd.DataFrame(
            metrics["classification_report"]
        ).T

        st.dataframe(
            report.round(3),
            use_container_width=True
        )


# ======================================
# TAB 3 - ANÁLISE EM LOTE
# ======================================

with tab3:

    st.subheader("Analisar várias opiniões")

    st.write(
        "Carrega um ficheiro CSV com uma coluna chamada `text`."
    )

    uploaded_file = st.file_uploader(
        "Seleciona um ficheiro CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            batch_data = pd.read_csv(uploaded_file)

            if "text" not in batch_data.columns:

                st.error(
                    "O ficheiro deve conter uma coluna chamada text."
                )

            elif batch_data.empty:

                st.warning("O ficheiro está vazio.")

            elif batch_data["text"].isna().any():

                st.error(
                    "Existem opiniões vazias. Corrige o CSV."
                )

            else:

                if len(batch_data) > 5000:

                    st.error(
                        "O limite desta demonstração é de 5000 opiniões."
                    )

                elif st.button("Analisar todas as opiniões"):

                    batch_data["text"] = (
                        batch_data["text"].astype(str)
                    )

                    predictions = model.predict(
                        batch_data["text"]
                    )

                    probabilities = model.predict_proba(
                        batch_data["text"]
                    )

                    batch_data["sentiment"] = predictions

                    batch_data["confidence"] = (
                        probabilities.max(axis=1)
                    )

                    st.success("Análise concluída!")

                    st.dataframe(
                        batch_data,
                        use_container_width=True
                    )

                    # Distribuição dos sentimentos
                    counts = (
                        batch_data["sentiment"]
                        .value_counts()
                        .reset_index()
                    )

                    counts.columns = [
                        "Sentimento",
                        "Quantidade"
                    ]

                    fig = px.pie(
                        counts,
                        names="Sentimento",
                        values="Quantidade",
                        title="Distribuição dos sentimentos",
                        color="Sentimento",
                        color_discrete_map={
                            "positive": "#16a34a",
                            "negative": "#dc2626",
                            "neutral": "#64748b"
                        }
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    # Download dos resultados
                    csv = batch_data.to_csv(
                        index=False
                    ).encode("utf-8-sig")

                    st.download_button(
                        "Descarregar resultados CSV",
                        data=csv,
                        file_name="sentiment_results.csv",
                        mime="text/csv"
                    )

        except Exception as error:

            st.error(f"Erro ao processar o ficheiro: {error}")


# Rodapé
st.divider()

st.caption(
    "Sentiment Analysis AI | Python · Scikit-learn · Streamlit"
)