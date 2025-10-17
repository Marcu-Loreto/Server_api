#!/usr/bin/env bash
set -Eeuo pipefail

# Ir para a pasta do script
cd "$(dirname "$0")"

# Criar venv se não existir
if [ ! -d ".venv" ]; then
  uv venv .venv
fi

# Ativar venv
source .venv/bin/activate

# Instalar/ sincronizar deps
uv pip sync requirements.txt

# Porta e modo (DEV=1 usa --reload)
PORT="${PORT:-8000}"
DEV="${DEV:-1}"

# Garantir .env carregado (o app já chama load_dotenv)
export PYTHONUNBUFFERED=1

if [ "$DEV" = "1" ]; then
  exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --reload
else
  exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
fi
