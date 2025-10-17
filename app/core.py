import os, re, json
from io import BytesIO
from collections import Counter
from itertools import combinations
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

# ===== Optional deps (fallbacks) =====
_WORDCLOUD_AVAILABLE = True
try:
    from wordcloud import WordCloud
except Exception:
    _WORDCLOUD_AVAILABLE = False

_GRAPH_AVAILABLE = True
try:
    import networkx as nx
    from pyvis.network import Network
except Exception:
    _GRAPH_AVAILABLE = False

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
load_dotenv()
API_TOKEN = os.getenv("API_TOKEN", "")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY não encontrada no ambiente (.env).")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
TEMPERATURA_PADRAO = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
MAX_TOKENS_PADRAO = int(os.getenv("OPENAI_MAX_TOKENS", "400"))


SYSTEM_PROMPT = """
Você é o Assistente de Atendimento e Conciliação da empresa.
Missão: resolver solicitações de clientes com rapidez, cordialidade e foco em acordos justos.
Você responde apenas após a primeira mensagem do usuário. Não peça dados pessoais por padrão.
Princípios: clareza, opções de conciliação, prazos/steps, no jargão, proteção de dados.
Formato:
- Resumo do caso:
- Solução proposta:
- Próximos passos:
- Observações:
""".strip()

client = OpenAI(api_key=OPENAI_API_KEY)


def _is_nano(model_name: str) -> bool:
    return "nano" in (model_name or "").lower()


# -----------------------------------------------------------------------------
# LLM
# -----------------------------------------------------------------------------
def call_llm(
    user_message: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Chamada robusta: trata modelos 'nano' (sem temperature e usando max_completion_tokens).
    """
    model = model or OPENAI_MODEL
    temperature = TEMPERATURA_PADRAO if temperature is None else temperature
    max_tokens = MAX_TOKENS_PADRAO if max_tokens is None else max_tokens

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    if _is_nano(model):
        resp = client.chat.completions.create(
            model=model, messages=messages, max_completion_tokens=max_tokens
        )
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return (resp.choices[0].message.content or "").strip()


# -----------------------------------------------------------------------------
# Sentimento
# -----------------------------------------------------------------------------
def _formatar_prompt_sentimento(texto: str) -> str:
    return (
        "Você é um classificador de sentimento. Classifique a mensagem a seguir.\n"
        "Responda APENAS com JSON válido:\n"
        '{"label":"positivo|neutro|negativo","confidence":0.0-1.0,"emotions":["..."],"reason":"..."}\n'
        "Mensagem:\n"
        f"{texto.strip()}"
    )


def analisar_sentimento(
    texto: str, modelo_sentimento: Optional[str] = None
) -> Dict[str, Any]:
    modelo_sentimento = modelo_sentimento or OPENAI_MODEL
    try:
        resp = client.chat.completions.create(
            model=modelo_sentimento,
            messages=[
                {"role": "system", "content": "Retorne JSON estrito."},
                {"role": "user", "content": _formatar_prompt_sentimento(texto)},
            ],
            temperature=0.0,
            max_tokens=150,
            top_p=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        label = str(data.get("label", "neutro")).lower()
        if label not in {"positivo", "neutro", "negativo"}:
            label = "neutro"
        conf = float(data.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
        emotions = data.get("emotions", [])
        if not isinstance(emotions, list):
            emotions = [str(emotions)]
        reason = str(data.get("reason", "")).strip()
        return {
            "label": label,
            "confidence": conf,
            "emotions": emotions,
            "reason": reason,
        }
    except Exception as e:
        return {
            "label": "neutro",
            "confidence": 0.0,
            "emotions": [],
            "reason": f"Falha: {e}",
        }


def score_from_label(label: str, confidence: float) -> float:
    sgn = 1 if label == "positivo" else (-1 if label == "negativo" else 0)
    try:
        c = float(confidence)
    except Exception:
        c = 0.0
    c = max(0.0, min(1.0, c))
    return round(sgn * c, 3)


# -----------------------------------------------------------------------------
# Tokenização PT-BR
# -----------------------------------------------------------------------------
_PT_STOPWORDS = {
    "a",
    "à",
    "às",
    "ao",
    "aos",
    "as",
    "o",
    "os",
    "um",
    "uma",
    "uns",
    "umas",
    "de",
    "da",
    "do",
    "das",
    "dos",
    "dá",
    "dão",
    "em",
    "no",
    "na",
    "nos",
    "nas",
    "por",
    "para",
    "pra",
    "com",
    "sem",
    "entre",
    "sobre",
    "sob",
    "até",
    "após",
    "que",
    "se",
    "é",
    "ser",
    "são",
    "era",
    "eram",
    "foi",
    "fui",
    "vai",
    "vou",
    "e",
    "ou",
    "mas",
    "como",
    "quando",
    "onde",
    "qual",
    "quais",
    "porque",
    "porquê",
    "já",
    "não",
    "sim",
    "também",
    "mais",
    "menos",
    "muito",
    "muita",
    "muitos",
    "muitas",
    "meu",
    "minha",
    "meus",
    "minhas",
    "seu",
    "sua",
    "seus",
    "suas",
    "depois",
    "antes",
    "este",
    "esta",
    "estes",
    "estas",
    "isso",
    "isto",
    "aquele",
    "aquela",
    "aqueles",
    "aquelas",
    "lhe",
    "lhes",
    "ele",
    "ela",
    "eles",
    "elas",
    "você",
    "vocês",
    "nós",
    "nosso",
    "nossa",
    "nossos",
    "nossas",
}


def tokenize_pt(texto: str):
    texto = texto.lower()
    tokens = re.findall(r"[a-zA-ZÀ-ÿ]+", texto)
    tokens = [t for t in tokens if len(t) >= 3 and t not in _PT_STOPWORDS]
    return tokens


# -----------------------------------------------------------------------------
# WordCloud
# -----------------------------------------------------------------------------
def gerar_wordcloud(corpus_text: str, width: int = 450, height: int = 280):
    if not corpus_text.strip():
        raise ValueError("Texto vazio para wordcloud.")
    if not _WORDCLOUD_AVAILABLE:
        raise RuntimeError(
            "Pacote 'wordcloud' não encontrado. Instale: pip install wordcloud"
        )
    wc = WordCloud(
        width=width, height=height, background_color="white", collocations=False
    )
    wc.generate(corpus_text)
    img = wc.to_image()
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf  # BytesIO


# -----------------------------------------------------------------------------
# Grafo (NetworkX + PyVis)
# -----------------------------------------------------------------------------
def build_word_graph(token_sequences: List[List[str]], min_edge_weight: int = 1):
    if not _GRAPH_AVAILABLE:
        raise RuntimeError("Instale: pip install networkx pyvis")
    import networkx as nx

    G = nx.Graph()
    node_counts = Counter()
    edge_counts = Counter()

    for seq in token_sequences:
        node_counts.update(seq)
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            if a == b:
                continue
            edge = tuple(sorted((a, b)))
            edge_counts[edge] += 1

    for w, c in node_counts.items():
        G.add_node(w, count=int(c))
    for (a, b), w in edge_counts.items():
        if w >= max(1, int(min_edge_weight)):
            G.add_edge(a, b, weight=int(w))
    return G


def subgraph_paths_to_target(G, target: str, max_depth: int = 4):
    import networkx as nx

    if G is None or target not in G:
        return None
    visited = {target}
    frontier = {target}
    depth = 0
    while frontier and depth < max_depth:
        next_frontier = set()
        for u in frontier:
            for v in G.neighbors(u):
                if v not in visited:
                    visited.add(v)
                    next_frontier.add(v)
        frontier = next_frontier
        depth += 1
    return G.subgraph(visited).copy()


def render_graph_pyvis(
    G,
    highlight_target: Optional[str] = None,
    height_px: int = 600,
    dark_mode: bool = False,
) -> str:
    if not _GRAPH_AVAILABLE:
        raise RuntimeError("Instale: pip install networkx pyvis")
    if G is None or len(G) == 0:
        raise ValueError("Grafo vazio/inválido.")

    bg = "#0f172a" if dark_mode else "#ffffff"
    fg = "#e5e7eb" if dark_mode else "#333333"
    net = Network(
        height=f"{height_px}px",
        width="100%",
        bgcolor=bg,
        font_color=fg,
        notebook=False,
        directed=False,
    )
    net.barnes_hut(
        gravity=-2000,
        central_gravity=0.3,
        spring_length=160,
        spring_strength=0.01,
        damping=0.9,
    )

    import networkx as nx

    node_counts = nx.get_node_attributes(G, "count")
    max_count = max(node_counts.values()) if node_counts else 1

    for node, data in G.nodes(data=True):
        count = int(data.get("count", 1))
        size = 10 + (30 * (count / max_count))
        color_high = "#34d399" if dark_mode else "#10b981"
        color_norm = "#93c5fd" if dark_mode else "#60a5fa"
        color = color_high if node == highlight_target else color_norm
        title = f"{node}<br/>freq: {count}"
        net.add_node(node, label=node, size=size, color=color, title=title)

    for u, v, data in G.edges(data=True):
        w = int(data.get("weight", 1))
        width = 1 + min(10, w)
        title = f"{u} — {v}<br/>coocorrências: {w}"
        net.add_edge(u, v, value=w, width=width, title=title)

    return net.generate_html()
