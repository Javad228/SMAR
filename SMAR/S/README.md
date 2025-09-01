## Process a single PDF with the SMAR S pipeline

This guide shows how to set up and run the pipeline on one PDF using `run_pipeline.sh`. It starts a local Grobid service, converts the PDF to structured JSON, and flattens it into sentence-level JSON using `A_flatten_paper.py`.

### Prerequisites

- **Java (JDK 11+)**: required by Grobid. Verify with:
```bash
java -version
```
- **Python 3.9+ with pip and venv**: the script creates a local virtualenv for doc2json and uses your system/conda Python for sentence flattening.
- **curl**: used to probe Grobid readiness.
- **SciSpaCy model in your Python** (used by `A_flatten_paper.py`): default model is `en_core_sci_lg`.

Install SciSpaCy and the model in the Python referenced by `PYTHON_BIN` (see below):
```bash
pip install -U pip
pip install spacy==3.7.4 scispacy==0.5.4
pip install "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz"
```

Tip: If you prefer a smaller model, install `en_core_sci_md` and pass `--model en_core_sci_md` when running `A_flatten_paper.py` manually.

### One-time repo layout

This directory expects the tools already vendored here:
- `tools/grobid-0.8.2/` (Grobid with Gradle wrapper)
- `tools/s2orc-doc2json/` (S2ORC doc2json)

Both are used directly by the script. No global installs are required beyond the prerequisites above.

### Installation (manual build)

If the tools are not already vendored under `SMAR/S/tools/`, you can build them manually.

Build GROBID 0.8.2 from source:
```bash
# Download and extract GROBID
wget https://github.com/kermitt2/grobid/archive/0.8.2.zip
unzip 0.8.2.zip
cd grobid-0.8.2

# Build with Gradle
./gradlew clean install
```

Place the resulting `grobid-0.8.2` directory at `SMAR/S/tools/grobid-0.8.2` (or create a symlink there) so `run_pipeline.sh` can find it.

Optionally, build `pdffigures2` (some figure extraction workflows use it):
```bash
# Clone the repo
git clone https://github.com/allenai/pdffigures2.git
cd pdffigures2

# Build using SBT
sbt assembly
```

Note: `pdffigures2` is not required by `run_pipeline.sh`. The doc2json step may still produce figure metadata without it. Install only if your use-case requires the Scala JAR.

### Quick start (single PDF)

From this directory:
```bash
cd SMAR/S
INPUT_PDF=/absolute/path/to/your.pdf bash run_pipeline.sh
```

Key environment variables you can set:
- `INPUT_PDF` (required unless you want the default `pdfs/test_bio.pdf`) — absolute path is recommended.
- `PYTHON_BIN` — which Python to use for `A_flatten_paper.py` (e.g. a conda env). If unset, the script looks for `$HOME/miniconda/envs/together_ft/bin/python`, then `python`, then `python3`.
- `KEEP_GROBID=1` — keep the Grobid service running between runs.

### What the script does

1) Starts Grobid (`tools/grobid-0.8.2`) locally (or reuses an already running instance).  
2) Creates an isolated venv `.venv_doc2json` and installs `tools/s2orc-doc2json` into it.  
3) Runs doc2json to convert your PDF into S2ORC-like JSON.  
4) Runs `A_flatten_paper.py` to split the paper into cleaned sentences using SciSpaCy and writes `sentences.json`.

### Outputs

Results are written under `output/<pdf_stem>/`:
- `<pdf_stem>.json` — structured S2ORC-like JSON from doc2json
- `sentences.json` — flattened sentence list produced by `A_flatten_paper.py`
- `figures.json` — figure metadata if produced by doc2json

Example:
```bash
ls -1 SMAR/S/output/my_paper/
# my_paper.json
# sentences.json
# figures.json
```

Preview the first few sentences:
```bash
jq '.[0:5]' SMAR/S/output/my_paper/sentences.json
```

### Running the flattener manually (optional)

If you already have a doc2json output (e.g., `my_paper.json`), you can run the flattener directly with a chosen model:
```bash
PYTHON_BIN=python  # or your conda/env python
"$PYTHON_BIN" SMAR/S/A_flatten_paper.py \
  -i SMAR/S/output/my_paper/my_paper.json \
  -o SMAR/S/output/my_paper/sentences.json \
  --model en_core_sci_md \
  --short-merge-threshold 6
```

Notes:
- Default CLI model is `en_core_sci_lg` (high quality, larger). Smaller `en_core_sci_md` is also supported.
- The script merges very short sentences into the previous sentence; tune with `--short-merge-threshold`.

### Troubleshooting

- **Grobid failed to start / Java errors**: Ensure JDK 11+ is installed and on PATH. Try `java -version`. If using WSL, install OpenJDK via apt.
- **OSError: Could not load SciSpaCy model**: Install the model in the Python used for `A_flatten_paper.py` (the `PYTHON_BIN` one). See install commands above. You may also use the medium model by installing `en_core_sci_md-0.5.4` and passing `--model en_core_sci_md` when running the flattener manually.
- **pip/venv errors when creating `.venv_doc2json`**: On Ubuntu, make sure `python3-venv` is installed: `sudo apt-get install -y python3-venv`.
- **Permission denied running the script**: Run with `bash run_pipeline.sh` or make it executable: `chmod +x run_pipeline.sh`.

### Re-running and performance tips

- Set `KEEP_GROBID=1` to avoid restarting Grobid between runs:
```bash
KEEP_GROBID=1 INPUT_PDF=/abs/path/to/your.pdf bash run_pipeline.sh
```
- Outputs are written to a stable directory (`output/<pdf_stem>/`), so each run overwrites the same files for that PDF stem.


