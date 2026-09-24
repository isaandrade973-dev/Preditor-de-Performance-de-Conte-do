"""
============================================================================
 Preditor de Performance de Conteúdo — Categoria x Hashtags -> Alcance
============================================================================
Aplicação Streamlit com modelo de regressão em TensorFlow/Keras que estima
o alcance esperado de um vídeo com base na categoria de conteúdo e na
quantidade de hashtags utilizadas, sinalizando visualmente se a publicação
tende a performar como "Viral", "Flopado" ou "Performance Média".

------------------------------------------------------------------------
COMO TESTAR LOCALMENTE
------------------------------------------------------------------------
1) Crie e ative um ambiente virtual (recomendado):
   python -m venv venv
   venv\\Scripts\\activate        # Windows
   source venv/bin/activate      # macOS/Linux

2) Instale as dependências (mesmo arquivo usado no deploy):
   pip install -r requirements.txt

3) Rode a aplicação:
   streamlit run app.py

4) Acesse http://localhost:8501

------------------------------------------------------------------------
COMO IMPLANTAR NO RENDER
------------------------------------------------------------------------
1) Suba este arquivo (app.py) e o requirements.txt em um repositório GitHub.
2) No Render, crie um "Web Service" apontando para o repositório.
3) Build Command:
   pip install -r requirements.txt
4) Start Command:
   streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
============================================================================
"""

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Preditor de Performance de Conteúdo",
    page_icon="📊",
    layout="centered",
)

CATEGORIAS = ["Dança", "Humor", "Beleza", "Games", "Educação", "Lifestyle"]

# ---------------------------------------------------------------------------
# 1) Geração dos dados de treino (base histórica simulada)
#    Cada categoria tem um alcance-base e uma sensibilidade diferente ao
#    número de hashtags. Excesso de hashtags (> 20) penaliza o alcance,
#    refletindo o efeito de "spam" observado na prática.
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

PERFIL_CATEGORIA = {
    "Dança":      {"base": 15_000, "efeito_hashtag": 1800},
    "Humor":      {"base": 20_000, "efeito_hashtag": 2200},
    "Beleza":     {"base": 12_000, "efeito_hashtag": 1500},
    "Games":      {"base": 9_000,  "efeito_hashtag": 1200},
    "Educação":   {"base": 7_000,  "efeito_hashtag": 900},
    "Lifestyle":  {"base": 10_000, "efeito_hashtag": 1100},
}


@st.cache_data
def gerar_dados_historicos(n_por_categoria=300):
    registros = []
    for categoria in CATEGORIAS:
        perfil = PERFIL_CATEGORIA[categoria]
        hashtags = np.random.randint(0, 31, size=n_por_categoria)
        penalidade = np.where(hashtags > 20, (hashtags - 20) * 400, 0)
        ruido = np.random.normal(0, 2500, size=n_por_categoria)
        alcance = (
            perfil["base"]
            + perfil["efeito_hashtag"] * hashtags
            - penalidade
            + ruido
        )
        alcance = np.clip(alcance, 500, None)
        for h, a in zip(hashtags, alcance):
            registros.append({"categoria": categoria, "hashtags": h, "alcance": a})
    return pd.DataFrame(registros)


df = gerar_dados_historicos()

# ---------------------------------------------------------------------------
# 2) Pré-processamento: one-hot da categoria + normalização das features
# ---------------------------------------------------------------------------
df_encoded = pd.get_dummies(df, columns=["categoria"])
colunas_categoria = [c for c in df_encoded.columns if c.startswith("categoria_")]
feature_cols = ["hashtags"] + colunas_categoria

hashtags_mean, hashtags_std = df["hashtags"].mean(), df["hashtags"].std()
alcance_mean, alcance_std = df["alcance"].mean(), df["alcance"].std()

df_encoded["hashtags"] = (df_encoded["hashtags"] - hashtags_mean) / hashtags_std
X = df_encoded[feature_cols].values.astype("float32")
y = ((df["alcance"] - alcance_mean) / alcance_std).values.astype("float32")

# Limiares de classificação (percentis da base histórica)
LIMIAR_VIRAL = float(np.percentile(df["alcance"], 75))
LIMIAR_FLOPADO = float(np.percentile(df["alcance"], 25))


# ---------------------------------------------------------------------------
# 3) Treinamento do modelo (cacheado — treina uma única vez por sessão)
# ---------------------------------------------------------------------------
@st.cache_resource
def treinar_modelo(X, y, epochs=150, lr=0.01):
    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )
    historico = modelo.fit(X, y, epochs=epochs, verbose=0, validation_split=0.2)
    return modelo, historico.history


with st.spinner("Carregando modelo preditivo..."):
    modelo, historico = treinar_modelo(X, y)

# ---------------------------------------------------------------------------
# 4) Interface — cabeçalho institucional
# ---------------------------------------------------------------------------
st.title("📊 Preditor de Performance de Conteúdo")
st.caption(
    "Ferramenta interna para estimativa de alcance de publicações, "
    "com base em dados históricos por categoria e volume de hashtags."
)
st.divider()

with st.expander("Detalhes técnicos do modelo (uso interno)"):
    st.write(
        f"Modelo de regressão (TensorFlow/Keras), treinado sobre "
        f"{len(df)} registros históricos simulados. "
        f"MAE final (escala normalizada): {historico['mae'][-1]:.3f}"
    )
    st.write(
        f"Limiar de classificação — Viral: alcance ≥ {LIMIAR_VIRAL:,.0f} | "
        f"Flopado: alcance ≤ {LIMIAR_FLOPADO:,.0f}"
    )

# ---------------------------------------------------------------------------
# 5) Formulário de simulação
# ---------------------------------------------------------------------------
st.subheader("Simulação de Publicação")

col1, col2 = st.columns(2)
with col1:
    categoria_input = st.selectbox("Categoria do vídeo", CATEGORIAS)
with col2:
    hashtags_input = st.slider("Quantidade de hashtags", min_value=0, max_value=30, value=10)

# Monta o vetor de entrada no mesmo formato do treino
entrada = {col: 0.0 for col in feature_cols}
entrada["hashtags"] = (hashtags_input - hashtags_mean) / hashtags_std
coluna_categoria = f"categoria_{categoria_input}"
if coluna_categoria in entrada:
    entrada[coluna_categoria] = 1.0

X_input = np.array([[entrada[col] for col in feature_cols]], dtype="float32")

# ---------------------------------------------------------------------------
# 6) Predição e classificação de resultado
# ---------------------------------------------------------------------------
alcance_pred_norm = modelo.predict(X_input, verbose=0)[0][0]
alcance_pred = max(alcance_pred_norm * alcance_std + alcance_mean, 0)

if alcance_pred >= LIMIAR_VIRAL:
    status, cor, icone = "VIRAL", "#0F9D58", "🔥"
elif alcance_pred <= LIMIAR_FLOPADO:
    status, cor, icone = "FLOPADO", "#D93025", "❄️"
else:
    status, cor, icone = "PERFORMANCE MÉDIA", "#F9AB00", "📈"

st.divider()
st.markdown(
    f"""
    <div style="
        background-color:{cor}1A;
        border-left: 6px solid {cor};
        padding: 20px;
        border-radius: 8px;
    ">
        <span style="font-size:28px; font-weight:700; color:{cor};">
            {icone} {status}
        </span>
        <br><br>
        <span style="font-size:16px;">
            Alcance estimado: <b>{alcance_pred:,.0f} visualizações</b>
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Estimativa gerada por modelo estatístico com base em dados históricos "
    "simulados. Recomenda-se validar com dados reais da plataforma antes "
    "de decisões de investimento em mídia."
)
