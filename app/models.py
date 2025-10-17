# app/models.py
# Com exemplos para testar no swegger UI
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


# ---------------------- /chat ----------------------
class ChatIn(BaseModel):
    message: str = Field(
        ..., min_length=1, description="Mensagem do usuário para o LLM."
    )
    model: Optional[str] = Field(None, description="Modelo a usar (opcional).")
    temperature: Optional[float] = Field(
        None, ge=0, le=2, description="Criatividade do modelo (0-2)."
    )
    max_tokens: Optional[int] = Field(
        None, ge=1, description="Limite de tokens da resposta."
    )

    # # Exemplo mostrado no Swagger
    # model_config = ConfigDict(
    #     json_schema_extra={
    #         "example": {
    #             "message": "Resuma em 2 linhas as vantagens do meu produto.",
    #             "temperature": 0.2,
    #             "max_tokens": 400
    #         }
    #     }
    # )


class ChatOut(BaseModel):
    reply: str = Field(..., description="Resposta textual gerada pelo LLM.")
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "reply": "Seu produto destaca-se pela facilidade de uso e atendimento ágil, reduzindo tempo de treinamento e aumentando a satisfação do cliente."
            }
        }
    )


# ------------------- /sentiment --------------------
class SentimentIn(BaseModel):
    text: str = Field(..., description="Texto a classificar.")
    model: Optional[str] = Field(None, description="Modelo a usar (opcional).")
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "O atendimento foi excelente, entregaram antes do prazo!"
            }
        }
    )


class SentimentOut(BaseModel):
    label: str = Field(..., description="positivo | neutro | negativo")
    confidence: float = Field(
        ..., ge=0, le=1, description="Confiança do classificador."
    )
    emotions: List[str] = Field(
        default_factory=list, description="Emoções detectadas (se houver)."
    )
    reason: str = Field(..., description="Justificativa do classificador.")
    score: float = Field(..., ge=-1, le=1, description="Score agregado (-1 a 1).")
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "label": "positivo",
                "confidence": 0.93,
                "emotions": ["alegria", "alívio"],
                "reason": "Expressões de satisfação e elogio ao atendimento.",
                "score": 0.93,
            }
        }
    )


# ------------------- /wordcloud --------------------
class WordCloudIn(BaseModel):
    text: str = Field(
        ..., description="Corpus de texto para gerar a nuvem de palavras."
    )
    width: int = Field(450, ge=100, le=2000, description="Largura em pixels.")
    height: int = Field(280, ge=100, le=2000, description="Altura em pixels.")
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "preço bom bom entrega rápida atendimento ótimo ótimo ótimo",
                "width": 600,
                "height": 300,
            }
        }
    )


# --------------------- /graph ----------------------
class GraphIn(BaseModel):
    texts: List[str] = Field(
        ..., description="Lista de mensagens do usuário (tokenizadas internamente)."
    )
    min_edge_weight: int = Field(
        1, ge=1, description="Mínimo de coocorrências para criar aresta."
    )
    target: Optional[str] = Field(
        None, description="Palavra alvo para destacar/filtrar (opcional)."
    )
    paths_only: bool = Field(True, description="Se True, retorna subgrafo até o alvo.")
    max_depth: int = Field(
        4, ge=1, le=10, description="Profundidade máxima do subgrafo."
    )
    dark_mode: bool = Field(True, description="Cores escuras (True) ou claras (False).")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texts": [
                    "gostei do preço e da entrega",
                    "entrega atrasou mas o preço é bom",
                    "ótimo atendimento e bom preço",
                ],
                "target": "preço",
                "min_edge_weight": 1,
                "paths_only": True,
                "max_depth": 4,
                "dark_mode": True,
            }
        }
    )


class GraphOut(BaseModel):
    html: str = Field(..., description="HTML PyVis para renderizar o grafo.")
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "html": "<!doctype html><html><head>...</head><body>...</body></html>"
            }
        }
    )
