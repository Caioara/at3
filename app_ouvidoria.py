"""Ouvidoria Inteligente — interface Streamlit para análise semântica."""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from ouvidoria_core import (
        MODEL_OPTIONS,
        cosine_matrix,
        embed_texts,
        load_manifestacoes,
        reduce_embeddings,
        split_text,
    )
    CORE_ERROR: str | None = None
except Exception as exc:  # The app should still render if setup is incomplete.
    MODEL_OPTIONS = {}
    CORE_ERROR = f"{type(exc).__name__}: {exc}"


st.set_page_config(page_title="Ouvidoria Inteligente", page_icon="◌", layout="wide")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');
      :root { --ink:#f2f8fa; --muted:#b4c8d0; --line:#294957; --paper:#102d3a; --ground:#061923; --teal:#68d8d0; --aqua:#30bdb5; --field:#0b2531; }
      .stApp { background:radial-gradient(circle at 88% 0%, #164456 0, transparent 30%), radial-gradient(circle at 0% 66%, #0a303c 0, transparent 28%), var(--ground); color:var(--ink); }
      html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
      h1, h2, h3 { font-family:'Fraunces', Georgia, serif !important; color:var(--ink); letter-spacing:-.035em; }
      h1 { font-size:clamp(2.1rem, 4vw, 3.5rem) !important; margin:0 !important; }
      section[data-testid="stSidebar"] { background:#09212c; border-right:1px solid var(--line); }
      section[data-testid="stSidebar"] > div { padding-top:2.1rem; }
      [data-testid="stSidebar"] .stMarkdown p { color:var(--muted); font-size:.86rem; }
      .block-container { max-width:1280px; padding-top:3rem; padding-bottom:3.5rem; }
      .eyebrow { color:var(--teal); font-size:.72rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; margin-bottom:.45rem; }
      .lede { color:var(--muted); max-width:690px; font-size:1.04rem; line-height:1.6; margin:.8rem 0 2rem; }
      .rule { height:1px; background:var(--line); margin:1.2rem 0 1.6rem; }
      .metric-strip { display:flex; gap:12px; flex-wrap:wrap; margin:0 0 1.35rem; }
      .metric-chip { background:var(--paper); border:1px solid var(--line); border-radius:8px; padding:.7rem .95rem; min-width:132px; box-shadow:0 8px 24px rgba(0,0,0,.16); }
      .metric-chip b { display:block; color:var(--teal); font-size:1.25rem; line-height:1.1; }
      .metric-chip span { color:var(--muted); font-size:.72rem; }
      .result-card { position:relative; overflow:hidden; background:var(--paper); border:1px solid var(--line); border-radius:9px; padding:1.15rem 1.25rem 1.15rem 1.5rem; margin:.7rem 0; box-shadow:0 8px 24px rgba(0,0,0,.18); }
      .score-bar { position:absolute; top:0; left:0; bottom:0; width:6px; background:var(--score); }
      .result-head { display:flex; justify-content:space-between; gap:1rem; align-items:baseline; }
      .result-title { color:var(--ink); font-weight:700; font-size:1rem; }
      .result-score { color:var(--score); font-size:.82rem; font-weight:700; white-space:nowrap; }
      .result-summary { color:var(--teal); margin:.6rem 0 .38rem; font-size:.91rem; font-weight:600; }
      .result-body { color:#d5e3e8; font-size:.91rem; line-height:1.55; }
      .empty-panel { background:rgba(16,45,58,.82); border:1px dashed #4c7481; border-radius:9px; padding:2.5rem 2rem; text-align:center; color:var(--muted); margin-top:.8rem; }
      .empty-panel strong { display:block; color:var(--ink); font-family:'Fraunces',serif; font-size:1.25rem; margin-bottom:.3rem; }
      .sidebar-label { color:var(--teal); font-size:.7rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
      [data-testid="stTabs"] [role="tablist"] { gap:1.25rem; border-bottom:1px solid var(--line); }
      [data-testid="stTabs"] [role="tab"] { height:44px; padding:0 .15rem; color:var(--muted); }
      [data-testid="stTabs"] [aria-selected="true"] { color:var(--teal); border-bottom:3px solid var(--aqua); }
      .stButton > button { background:var(--teal); color:#06212a; border:0; border-radius:6px; font-weight:700; padding:.58rem 1.1rem; }
      .stButton > button:hover { background:#99eee7; color:#041920; }
      .stButton > button:focus-visible { outline:3px solid #f6cf62; outline-offset:2px; }
      .stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div { border-color:#4c7481 !important; border-radius:6px !important; background:var(--field) !important; color:var(--ink) !important; }
      .stTextInput input::placeholder, .stTextArea textarea::placeholder { color:#9ab5bf !important; opacity:1; }
      [data-baseweb="select"] *, [data-baseweb="base-input"] input { color:var(--ink) !important; }
      [data-testid="stSidebar"], [data-testid="stSidebar"] *, .stMarkdown, .stMarkdown p, label, [data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color:var(--ink); }
      [data-testid="stSidebar"] .stMarkdown p, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color:var(--muted) !important; }
      [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:7px; overflow:hidden; }
      [data-testid="stDataFrame"] iframe { background:var(--paper); }
      [data-testid="stExpander"] { background:var(--paper); border:1px solid var(--line); color:var(--ink); }
      [data-testid="stExpander"] summary, [data-testid="stExpander"] p { color:var(--ink) !important; }
      [data-testid="stAlert"] { background:#10313d; color:var(--ink); }
      div[data-baseweb="slider"] div[role="slider"] { background:var(--aqua); border-color:#b7fffa; }
      div[data-baseweb="slider"] div[role="progressbar"] { background:var(--aqua); }
      ::selection { background:#2e8586; color:#fff; }
      @media (max-width: 700px) { .block-container { padding:1.5rem 1rem 2rem; } .result-head { align-items:flex-start; flex-direction:column; gap:.2rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def _as_frame(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        return pd.DataFrame(data)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return _as_frame(load_manifestacoes())


def text_column(frame: pd.DataFrame) -> str | None:
    preferred = ("texto", "manifestacao", "descricao", "mensagem", "conteudo", "assunto")
    lowered = {str(c).lower(): c for c in frame.columns}
    for name in preferred:
        if name in lowered:
            return lowered[name]
    candidates = frame.select_dtypes(include="object").columns.tolist()
    return candidates[0] if candidates else None


def label_for(row: pd.Series, index: int) -> str:
    for key in ("protocolo", "id", "titulo", "assunto", "categoria"):
        if key in row and pd.notna(row[key]):
            return str(row[key])
    return f"Manifestação {index + 1:03d}"


def confidence(score: float) -> tuple[str, str, str]:
    if score > .70:
        return "#138a68", "Alta relação", "O tema e a intenção estão muito próximos da sua busca."
    if score > .50:
        return "#d79718", "Relação possível", "Há pontos em comum; vale conferir o contexto da manifestação."
    return "#c85555", "Relação fraca", "Compartilha poucos elementos de significado com a consulta."


def fixed_character_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Divisão previsível por posição, disponível para comparação didática."""
    safe_overlap = min(overlap, max(chunk_size - 1, 0))
    step = max(chunk_size - safe_overlap, 1)
    return [text[start:start + chunk_size] for start in range(0, len(text), step) if text[start:start + chunk_size]]


def cached_embeddings(texts: tuple[str, ...], model: str) -> np.ndarray:
    key = hashlib.sha1((model + "\x1f" + "\x1f".join(texts)).encode()).hexdigest()
    if st.session_state.get("embedding_key") != key:
        st.session_state["embeddings"] = np.asarray(embed_texts(list(texts), model))
        st.session_state["embedding_key"] = key
    return st.session_state["embeddings"]


def select_model_options() -> list[str]:
    if isinstance(MODEL_OPTIONS, dict):
        return list(MODEL_OPTIONS.keys())
    return list(MODEL_OPTIONS) if MODEL_OPTIONS else []


@st.cache_resource(show_spinner=False)
def selected_model_id(label: str) -> str:
    """Mantém a seleção de backend estável entre as interações do Streamlit."""
    return MODEL_OPTIONS.get(label, label) if isinstance(MODEL_OPTIONS, dict) else label


with st.sidebar:
    st.markdown("<div class='sidebar-label'>Ouvidoria municipal</div>", unsafe_allow_html=True)
    st.markdown("### Painel de análise")
    st.caption("Pesquisa por significado para apoiar a escuta e a tomada de decisão.")
    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    choices = select_model_options()
    model = st.selectbox("Modelo de linguagem", choices, disabled=not bool(choices), placeholder="Indisponível")
    top_k = st.slider("Resultados por busca", min_value=3, max_value=20, value=5, step=1)
    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-label'>Como ler</div>", unsafe_allow_html=True)
    st.caption("A cor lateral indica a confiança semântica. Ela apoia a análise humana — não substitui a leitura integral.")

st.markdown("<div class='eyebrow'>Inteligência para a escuta pública</div>", unsafe_allow_html=True)
st.title("Ouvidoria Inteligente")
st.markdown("<p class='lede'>Encontre padrões, aproxime relatos e torne a base de manifestações mais legível — com uma busca que entende contexto, não só palavras.</p>", unsafe_allow_html=True)

if CORE_ERROR:
    st.error("A camada de análise ainda não está disponível.")
    st.info(f"Não foi possível importar `ouvidoria_core.py`: `{CORE_ERROR}`. Adicione o arquivo e recarregue a página.")
    st.stop()

try:
    data = get_data()
    load_error = None
except Exception as exc:
    data = pd.DataFrame()
    load_error = f"{type(exc).__name__}: {exc}"

if load_error:
    st.error("Não foi possível abrir a base de manifestações.")
    st.info(f"Verifique se `manifestacoes.json` está na pasta do projeto. Detalhe: `{load_error}`")

col = text_column(data) if not data.empty else None
count_label = f"{len(data):,}".replace(",", ".")
st.markdown(f"<div class='metric-strip'><div class='metric-chip'><b>{count_label}</b><span>manifestações disponíveis</span></div><div class='metric-chip'><b>{'pronta' if col else 'aguardando'}</b><span>base para análise semântica</span></div></div>", unsafe_allow_html=True)

tab_search, tab_base, tab_space, tab_chunks = st.tabs(["Busca semântica", "Base de manifestações", "Espaço vetorial", "Chunking"])

with tab_search:
    st.markdown("### Encontre relatos pelo que eles querem dizer")
    st.caption("Descreva uma situação com suas palavras. Os resultados são ordenados por proximidade de sentido.")
    query = st.text_input("O que você quer investigar?", placeholder="Ex.: demora no atendimento de saúde no bairro")
    run_search = st.button("Buscar manifestações", type="primary", disabled=not (query.strip() and col and model))
    if run_search:
        texts = tuple(data[col].fillna("").astype(str).tolist())
        try:
            with st.spinner("Lendo relações de significado na base..."):
                model_id = selected_model_id(model)
                corpus = cached_embeddings(texts, model_id)
                query_vector = np.asarray(embed_texts([query.strip()], model_id))
                scores = np.asarray(cosine_matrix(np.vstack([query_vector, corpus])))[0, 1:]
            order = np.argsort(scores)[::-1][:top_k]
            st.markdown(f"<div class='eyebrow'>{len(order)} sinais encontrados</div>", unsafe_allow_html=True)
            for rank, idx in enumerate(order):
                row = data.iloc[int(idx)]
                score = float(scores[idx])
                color, name, explanation = confidence(score)
                raw = str(row[col])
                excerpt = raw if len(raw) <= 360 else raw[:357].rstrip() + "…"
                st.markdown(f"""<article class='result-card' style='--score:{color}'><span class='score-bar'></span><div class='result-head'><span class='result-title'>{rank + 1:02d} · {label_for(row, int(idx))}</span><span class='result-score'>{score:.0%} de afinidade</span></div><div class='result-summary'>{name} · {explanation}</div><div class='result-body'>{excerpt}</div></article>""", unsafe_allow_html=True)
        except Exception as exc:
            st.error("A busca não pôde ser concluída com o modelo selecionado.")
            st.code(f"{type(exc).__name__}: {exc}", language=None)
    elif not query:
        st.markdown("<div class='empty-panel'><strong>Uma pergunta abre a escuta.</strong>Digite um tema acima para encontrar manifestações semelhantes, mesmo quando usam palavras diferentes.</div>", unsafe_allow_html=True)

with tab_base:
    st.markdown("### Conheça a matéria-prima da escuta")
    if data.empty:
        st.markdown("<div class='empty-panel'><strong>A base ainda não chegou.</strong>Quando `manifestacoes.json` estiver disponível, os registros aparecerão aqui para uma primeira leitura.</div>", unsafe_allow_html=True)
    else:
        search_base = st.text_input("Filtrar registros", placeholder="Filtre por qualquer palavra da base", key="base_filter")
        view = data.copy()
        if search_base:
            mask = view.astype(str).apply(lambda c: c.str.contains(re.escape(search_base), case=False, na=False)).any(axis=1)
            view = view[mask]
        st.caption(f"Exibindo {len(view):,} de {len(data):,} registros.".replace(",", "."))
        st.dataframe(view, use_container_width=True, hide_index=True, height=420)
        similarity_map = st.button("Gerar matriz de similaridade", disabled=not (col and model), key="similarity_matrix")
        if similarity_map:
            try:
                texts = tuple(data[col].fillna("").astype(str).tolist())
                with st.spinner("Calculando afinidades entre todas as manifestações..."):
                    embeddings = cached_embeddings(texts, selected_model_id(model))
                    matrix = np.asarray(cosine_matrix(embeddings))
                ids = [label_for(data.iloc[i], i) for i in range(len(data))]
                if PLOTLY_AVAILABLE:
                    heatmap = go.Figure(data=go.Heatmap(
                        z=matrix, x=ids, y=ids, colorscale="Teal", zmin=-1, zmax=1,
                        colorbar={"title": "Similaridade"},
                        hovertemplate="Origem: %{y}<br>Comparação: %{x}<br>Similaridade: %{z:.2f}<extra></extra>",
                    ))
                    heatmap.update_layout(
                        height=620, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0b2531",
                        font={"color": "#eaf4f7", "family": "DM Sans, sans-serif"},
                        margin={"l": 0, "r": 0, "t": 28, "b": 0},
                    )
                    heatmap.update_xaxes(tickfont={"color": "#b4c8d0", "size": 10}, tickangle=-45)
                    heatmap.update_yaxes(tickfont={"color": "#b4c8d0", "size": 10}, autorange="reversed")
                    st.plotly_chart(heatmap, use_container_width=True)
                else:
                    st.info("Instale Plotly para visualizar a matriz de similaridade.")
            except Exception as exc:
                st.error("Não foi possível gerar a matriz de similaridade.")
                st.code(f"{type(exc).__name__}: {exc}", language=None)

with tab_space:
    st.markdown("### Veja os temas se aproximarem")
    st.caption("Cada ponto representa uma manifestação. Pontos vizinhos tendem a compartilhar contexto e intenção.")
    method = st.selectbox("Método de redução", ["PCA", "t-SNE"], key="reduction")
    build_space = st.button("Mapear a base", disabled=not (col and model))
    if build_space:
        try:
            texts = tuple(data[col].fillna("").astype(str).tolist())
            with st.spinner("Organizando os relatos em um mapa de afinidades..."):
                emb = cached_embeddings(texts, selected_model_id(model))
                coords = np.asarray(reduce_embeddings(emb, method))
            if coords.ndim != 2 or coords.shape[1] < 2:
                raise ValueError("A redução precisa retornar uma matriz com duas dimensões.")
            category_col = "categoria_oficial" if "categoria_oficial" in data.columns else None
            plot_data = pd.DataFrame({"Eixo 1": coords[:, 0], "Eixo 2": coords[:, 1], "Registro": [label_for(data.iloc[i], i) for i in range(len(data))], "Resumo": data[col].astype(str).str.slice(0, 180), "Categoria": data[category_col].astype(str) if category_col else "Não informada"})
            if PLOTLY_AVAILABLE:
                fig = px.scatter(plot_data, x="Eixo 1", y="Eixo 2", color="Categoria", hover_name="Registro", hover_data={"Resumo": True, "Eixo 1": False, "Eixo 2": False})
                fig.update_traces(marker={"size": 8, "opacity": .72})
                fig.update_layout(
                    height=490,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#0b2531",
                    font={"color": "#eaf4f7", "family": "DM Sans, sans-serif"},
                    hoverlabel={"bgcolor": "#102d3a", "bordercolor": "#68d8d0", "font": {"color": "#f2f8fa"}},
                    legend={"font": {"color": "#eaf4f7"}},
                    margin={"l":0,"r":0,"t":10,"b":0},
                )
                fig.update_xaxes(gridcolor="#294957", zerolinecolor="#4c7481", title_font={"color": "#b4c8d0"}, tickfont={"color": "#b4c8d0"})
                fig.update_yaxes(gridcolor="#294957", zerolinecolor="#4c7481", title_font={"color": "#b4c8d0"}, tickfont={"color": "#b4c8d0"})
                st.plotly_chart(fig, use_container_width=True)
                st.info("Leitura sugerida: cores agrupadas em regiões próximas indicam aderência entre categorias oficiais e semântica. Pontos de cores diferentes lado a lado revelam temas transversais — como iluminação, que pode afetar infraestrutura e segurança.")
            else:
                st.info("Instale Plotly para visualizar o mapa interativo.")
        except Exception as exc:
            st.error("Não foi possível gerar o espaço vetorial.")
            st.code(f"{type(exc).__name__}: {exc}", language=None)
    else:
        st.markdown("<div class='empty-panel'><strong>O mapa começa com uma decisão.</strong>Escolha um método e gere a visualização para explorar agrupamentos de temas.</div>", unsafe_allow_html=True)

with tab_chunks:
    st.markdown("### Recorte um relato sem perder seu fio")
    st.caption("Use blocos menores para tornar documentos extensos pesquisáveis, preservando uma faixa de contexto entre cada trecho.")
    source = st.text_area("Texto para segmentar", placeholder="Cole aqui uma manifestação longa para ver como ela será dividida.", height=140)
    strategy = st.selectbox(
        "Estratégia de segmentação",
        ["RecursiveCharacterTextSplitter (recomendada)", "Caracteres fixos"],
        help="A estratégia recomendada preserva melhor limites de sentenças e parágrafos; caracteres fixos divide apenas pela posição no texto.",
    )
    left, right = st.columns(2)
    with left:
        chunk_size = st.slider("Tamanho do trecho", 100, 1500, 500, 50)
    with right:
        overlap = st.slider("Sobreposição de contexto", 0, 500, 80, 10)
    cut = st.button("Dividir texto", disabled=not source.strip())
    if cut:
        try:
            parts = (
                split_text(source.strip(), chunk_size, overlap)
                if strategy == "RecursiveCharacterTextSplitter (recomendada)"
                else fixed_character_chunks(source.strip(), chunk_size, overlap)
            )
            st.caption(f"{len(parts)} trechos produzidos · a sobreposição preserva continuidade entre blocos.")
            for i, part in enumerate(parts, 1):
                with st.expander(f"Trecho {i:02d} · {len(str(part))} caracteres", expanded=i == 1):
                    st.write(part)
            if len(parts) > 1 and PLOTLY_AVAILABLE:
                vectors = embed_texts(parts, selected_model_id(model)) if model else embed_texts(parts)
                xy = reduce_embeddings(vectors, "PCA")
                chart = pd.DataFrame({"PCA 1": xy[:, 0], "PCA 2": xy[:, 1], "Chunk": [f"Trecho {i}" for i in range(1, len(parts) + 1)], "Embedding": [str(v[:8].round(3).tolist()) + "…" for v in np.asarray(vectors)]})
                chunk_fig = px.scatter(chart, x="PCA 1", y="PCA 2", text="Chunk", hover_data=["Embedding"], title="Embeddings dos chunks (projeção PCA)")
                chunk_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0b2531",
                    font={"color": "#eaf4f7", "family": "DM Sans, sans-serif"},
                    title_font={"color": "#f2f8fa"},
                    hoverlabel={"bgcolor": "#102d3a", "bordercolor": "#68d8d0", "font": {"color": "#f2f8fa"}},
                )
                chunk_fig.update_xaxes(gridcolor="#294957", zerolinecolor="#4c7481", tickfont={"color": "#b4c8d0"})
                chunk_fig.update_yaxes(gridcolor="#294957", zerolinecolor="#4c7481", tickfont={"color": "#b4c8d0"})
                st.plotly_chart(chunk_fig, use_container_width=True)
        except Exception as exc:
            st.error("Não foi possível segmentar este texto.")
            st.code(f"{type(exc).__name__}: {exc}", language=None)
    else:
        st.markdown("<div class='empty-panel'><strong>Experimente com uma narrativa real.</strong>Cole um texto acima para conferir os limites de cada trecho antes de indexá-lo.</div>", unsafe_allow_html=True)
