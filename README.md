# AI Brasil - Server API

API REST construída com FastAPI para processamento de linguagem natural, análise de sentimento e visualização de texto, integrada à API da OpenAI.

## Requisitos
- Python 3.8+
- pip ou Poetry

## Dependências
Dependências principais:
- python-dotenv>=1.0.0
- openai>=1.0.0
- fastapi>=0.104.0
- uvicorn>=0.23.0

Dependências opcionais (para funcionalidades extras):
- wordcloud>=1.9.0    (geração de nuvem de palavras)
- networkx>=3.1      (manipulação de grafos)
- pyvis>=0.3.2       (visualização de grafos)

Recomendado criar um arquivo `requirements.txt` ou usar Poetry para gerenciar dependências.

## Configuração
1. Clone o repositório:
```bash
git clone <seu-repo-url>
cd Server_api
```

2. Instale dependências:
```bash
# com pip
pip install -r requirements.txt

# ou com Poetry
poetry install
```

3. Crie um arquivo `.env` na raiz do projeto com as variáveis:
```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4.1-mini
OPENAI_TEMPERATURE=0.2
OPENAI_MAX_TOKENS=400
API_TOKEN=senha_forte
```
Nota: não versionar o `.env` nem expor chaves em repositório público.

## Execução (desenvolvimento)
Inicie a API com uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
A API ficará disponível em: `http://localhost:8000`

## Endpoints (exemplos)
As rotas podem variar conforme implementação; abaixo seguem exemplos típicos.

- POST /analyze  
  Corpo JSON: { "text": "texto para análise" }  
  Header: `Authorization: Bearer {API_TOKEN}`

- POST /wordcloud  
  Corpo JSON: { "text": "texto para nuvem" }  
  Retorna imagem PNG (stream/bytes)

- POST /wordgraph  
  Corpo JSON: { "text": "texto para grafo" }  
  Retorna HTML com visualização do grafo

Exemplo curl:
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Authorization: Bearer senha_forte" \
  -H "Content-Type: application/json" \
  -d '{"text":"Exemplo de texto para análise"}'
```

## Boas práticas e segurança
- Proteja a chave `OPENAI_API_KEY`. Use variáveis de ambiente e secret manager em produção.
- Valide e sanitize entradas do usuário antes de enviar ao LLM.
- Limite uso por rate limiting e autenticação adequada (Bearer token, API Gateway).
- Trate erros da API da OpenAI (timeouts, quotas, respostas inválidas).
- Registre (log) requisições e respostas relevantes para monitoramento, sem armazenar dados sensíveis.

## Documentação interativa
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Desenvolvimento
Fluxo sugerido:
1. Fork -> branch de feature -> commits pequenos -> PR com descrição e testes.
2. Escrever testes unitários para funções críticas (ex.: tokenização, parser de sentimento).
3. Usar lint (flake8/black/isort) e CI para garantir qualidade.

## Observações
- Algumas funcionalidades (wordcloud, grafos) dependem de bibliotecas opcionais. O código faz detecção e degrada funcionalidade quando pacotes não estão instalados.
- Ajuste variáveis `OPENAI_MODEL`, `OPENAI_TEMPERATURE` e `OPENAI_MAX_TOKENS` conforme necessidade e custo.

## Licença
Projeto com licença permissiva — ajustar conforme necessidade (ex.: MIT).

## Autor
Projeto mantido por você. Contribuições são bem-vindas.
// ...existing code...