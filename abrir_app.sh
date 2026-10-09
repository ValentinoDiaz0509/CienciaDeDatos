#!/usr/bin/env bash
# Abre la app del TPO (macOS / Linux):  ./abrir_app.sh
set -e
cd "$(dirname "$0")"

PY=""
for v in python3.12 python3.13 python3.11 python3.10 python3.14 python3 python; do
  if command -v "$v" >/dev/null 2>&1 && "$v" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PY="$v"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Falta instalar Python 3.10 o superior (recomendado: 3.12): https://www.python.org/downloads/"
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "Creando el entorno de Python (solo la primera vez)..."
  "$PY" -m venv .venv
fi
echo "Preparando la app. La primera vez tarda unos minutos..."
.venv/bin/python -m pip install -q --disable-pip-version-check -r app/requirements.txt

URL="http://localhost:8501"
( sleep 8; if command -v open >/dev/null 2>&1; then open "$URL"; elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"; fi ) >/dev/null 2>&1 &
echo "La app se abre en $URL (para cerrarla: Ctrl + C)."
exec .venv/bin/python -m streamlit run app/app.py
