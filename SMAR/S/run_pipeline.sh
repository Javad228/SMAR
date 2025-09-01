#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

# Resolve python interpreter (allow override via PYTHON_BIN, prefer conda env if present)
if [ -n "$PYTHON_BIN" ] && command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    : # Respect provided PYTHON_BIN
elif [ -x "$HOME/miniconda/envs/together_ft/bin/python" ]; then
    PYTHON_BIN="$HOME/miniconda/envs/together_ft/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "Python interpreter not found. Set PYTHON_BIN or install python."
    exit 1
fi

# Define directories and file paths
# Resolve script directory (location of this file) and repo root (two levels up)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Use tools that were moved with this script
GROBID_DIR="$SCRIPT_DIR/tools/grobid-0.8.2"

# Allow override of input PDF via env var; default to repo's pdfs/test_bio.pdf
if [ -z "$INPUT_PDF" ]; then
  INPUT_PDF="$REPO_DIR/pdfs/test_bio.pdf"
fi
PDF_BASENAME="$(basename "$INPUT_PDF")"
PDF_STEM="${PDF_BASENAME%.*}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"

# Use run-scoped temp and output directories to avoid collisions (kept under this script's dir)
OUTPUT_BASE_DIR="$SCRIPT_DIR/output"
TEMP_BASE_DIR="$OUTPUT_BASE_DIR/tmp"
TEMP_DIR="$TEMP_BASE_DIR/${PDF_STEM}_$RUN_ID"
# Write outputs into a stable directory (no date) so files are replaced each run
OUTPUT_DIR="$OUTPUT_BASE_DIR/${PDF_STEM}"

OUTPUT_JSON="$OUTPUT_DIR/${PDF_STEM}.json"
# Write sentences to a stable filename (no date) so it is replaced each run
SENTENCES_JSON="$OUTPUT_DIR/sentences.json"
FIGURES_JSON="$OUTPUT_DIR/figures.json"

# Ensure compatibility symlink in repo root if Grobid expects that path
if [ ! -e "$REPO_DIR/grobid-0.8.2" ] && [ ! -L "$REPO_DIR/grobid-0.8.2" ] && [ -d "$GROBID_DIR" ]; then
    ln -s "$GROBID_DIR" "$REPO_DIR/grobid-0.8.2" 2>/dev/null || true
fi

# Function to clean up Grobid process on exit
cleanup() {
    echo "Cleaning up..."
    # If KEEP_GROBID is set, do not stop the Grobid service (useful for batch runs)
    if [ -n "$KEEP_GROBID" ] && [ "$KEEP_GROBID" != "0" ]; then
        echo "Skipping Grobid shutdown due to KEEP_GROBID=$KEEP_GROBID"
        return
    fi
    if [ -n "$GROBID_PID" ] && ps -p $GROBID_PID > /dev/null; then
        echo "Stopping Grobid service (PID: $GROBID_PID)..."
        kill $GROBID_PID
    fi
}

# Register the cleanup function to be called on script exit
trap cleanup EXIT

# 1. Start Grobid service in the background (only if not already running)
STARTED_GROBID=0
if curl -s --fail "http://localhost:8070/api/isalive" > /dev/null; then
  echo "Grobid service already running."
else
  echo "Starting Grobid service..."
  cd "$GROBID_DIR"
  ./gradlew --no-daemon grobid-service:run &
  GROBID_PID=$!
  STARTED_GROBID=1
  cd "$SCRIPT_DIR"
fi

# Wait for Grobid service to be ready
if curl -s --fail "http://localhost:8070/api/isalive" > /dev/null; then
  echo "Grobid service is ready."
else
  echo "Waiting for Grobid service to be ready..."
  until curl -s --fail "http://localhost:8070/api/isalive" > /dev/null; do
    echo -n "."
    sleep 5
    if [ "$STARTED_GROBID" -eq 1 ] && ! ps -p $GROBID_PID > /dev/null; then
      echo "Grobid service failed to start."
      exit 1
    fi
  done
  echo
  echo "Grobid service is ready."
fi

# Ensure pip is available for the selected Python
if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
  echo "Error: pip is not available for $PYTHON_BIN. Please ensure you're using a Python with pip (e.g., activate your conda env or set PYTHON_BIN)."
  exit 1
fi

# Prepare isolated venv for doc2json to avoid polluting global env
DOC2JSON_DIR="$SCRIPT_DIR/tools/s2orc-doc2json"
DOC2JSON_VENV="$SCRIPT_DIR/.venv_doc2json"
DOC2JSON_PY="$DOC2JSON_VENV/bin/python"
if [ ! -x "$DOC2JSON_PY" ]; then
  echo "Creating virtual environment for doc2json at $DOC2JSON_VENV..."
  "$PYTHON_BIN" -m venv "$DOC2JSON_VENV"
fi
"$DOC2JSON_PY" -m pip install --upgrade pip setuptools wheel >/dev/null
if [ -d "$DOC2JSON_DIR" ]; then
  echo "Installing doc2json into isolated venv..."
  if [ -f "$DOC2JSON_DIR/requirements.txt" ]; then
    "$DOC2JSON_PY" -m pip install -r "$DOC2JSON_DIR/requirements.txt" >/dev/null
  fi
  "$DOC2JSON_PY" -m pip install -e "$DOC2JSON_DIR" >/dev/null
else
  echo "Error: local repo missing at $DOC2JSON_DIR"
  exit 1
fi

# 2. Run doc2json to process PDF
echo "Processing PDF: $INPUT_PDF"
# Ensure temp and output directories exist (run-scoped)
mkdir -p "$TEMP_DIR" "$OUTPUT_DIR"
# Retry doc2json in case Grobid needs a bit more time
MAX_TRIES=5
TRY=1
until [ $TRY -gt $MAX_TRIES ]
do
  if "$DOC2JSON_PY" -m doc2json.grobid2json.process_pdf \
    -i "$INPUT_PDF" \
    -t "$TEMP_DIR" \
    -o "$OUTPUT_DIR"; then
    break
  fi
  echo "doc2json failed (attempt $TRY/$MAX_TRIES). Retrying in 8s..."
  sleep 8
  TRY=$((TRY+1))
done

if [ ! -f "$OUTPUT_JSON" ]; then
  echo "Failed to produce JSON at $OUTPUT_JSON after $MAX_TRIES attempts."
  exit 1
fi

# 3. Run A_flatten_paper to create sentences file from your logic
echo "Flattening JSON output with custom logic..."
"$PYTHON_BIN" "$SCRIPT_DIR/A_flatten_paper.py" \
  -i "$OUTPUT_JSON" \
  -o "$SENTENCES_JSON"

echo "Script finished successfully."
