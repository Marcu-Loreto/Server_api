from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.security.api_key import APIKeyHeader
import os
from io import BytesIO

from .models import (
    ChatIn,
    ChatOut,
    SentimentIn,
    SentimentOut,
    WordCloudIn,
    GraphIn,
    GraphOut,
)
from .core import (
    call_llm,
    analisar_sentimento,
    score_from_label,
    tokenize_pt,
    gerar_wordcloud,
    build_word_graph,
    subgraph_paths_to_target,
    render_graph_pyvis,
)


# uvicorn app.main:app --reload --port 8000
# ========= Segurança por API Key =========
API_TOKEN = os.getenv("API_TOKEN", "")  # defina no .env: API_TOKEN=uma-senha-forte
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


def require_api_key(api_key: str = Security(api_key_header)):
    # Se API_TOKEN estiver vazio, bloqueia tudo (evita expor sem querer)
    if API_TOKEN and api_key == API_TOKEN:
        return True
    raise HTTPException(status_code=401, detail="Unauthorized")


# ========================================

app = FastAPI(title="Conversas API", version="1.0.0")

# CORS (ajuste a ORIGEM em produção)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # troque por ["https://seu-frontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz():
    return {"status": "OK - API esta no AR"}


# ---------------- Chat ----------------
@app.post("/chat", response_model=ChatOut, dependencies=[Depends(require_api_key)])
def chat(payload: ChatIn):
    try:
        reply = call_llm(
            payload.message,
            model=payload.model,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
        )
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(500, f"LLM error: {e}")


# --------------- Sentimento -----------
@app.post(
    "/sentiment", response_model=SentimentOut, dependencies=[Depends(require_api_key)]
)
def sentiment(payload: SentimentIn):
    try:
        data = analisar_sentimento(payload.text, modelo_sentimento=payload.model)
        score = score_from_label(data["label"], data["confidence"])
        return {
            "label": data["label"],
            "confidence": data["confidence"],
            "emotions": data["emotions"],
            "reason": data["reason"],
            "score": score,
        }
    except Exception as e:
        raise HTTPException(500, f"Sentiment error: {e}")


# ---------------- WordCloud (PNG) -----
@app.post(
    "/wordcloud",
    response_class=StreamingResponse,
    dependencies=[Depends(require_api_key)],
)
def wordcloud(payload: WordCloudIn):
    try:
        buf: BytesIO = gerar_wordcloud(payload.text, payload.width, payload.height)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(400, f"WordCloud error: {e}")


# ---------------- Grafo (HTML) --------
@app.post("/graph", response_model=GraphOut, dependencies=[Depends(require_api_key)])
def graph(payload: GraphIn):
    try:
        token_sequences = [tokenize_pt(t or "") for t in payload.texts]
        G = build_word_graph(token_sequences, min_edge_weight=payload.min_edge_weight)

        if payload.paths_only and payload.target:
            Gp = subgraph_paths_to_target(
                G, payload.target, max_depth=payload.max_depth
            )
            if Gp is not None and len(Gp) > 0:
                G = Gp

        html = render_graph_pyvis(
            G,
            highlight_target=payload.target,
            height_px=520,
            dark_mode=payload.dark_mode,
        )
        return {"html": html}
    except Exception as e:
        raise HTTPException(400, f"Graph error: {e}")


# -------- Página HTML pronta (embed do grafo) -----------
@app.post(
    "/graph/html", response_class=HTMLResponse, dependencies=[Depends(require_api_key)]
)
def graph_html(payload: GraphIn):
    out = graph(payload)
    return HTMLResponse(content=out["html"])
