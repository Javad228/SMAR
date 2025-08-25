# 📄 PDF Processing Tools Setup  
This guide explains how to install and run **[GROBID](https://github.com/kermitt2/grobid)** (for extracting structured data from PDFs) and **[pdffigures2](https://github.com/allenai/pdffigures2)** (for extracting figures, tables, and captions from PDFs).  
It also documents helper scripts (`extract_images.py` and `xml_parser.py`) to simplify the workflow.

---

## 🔹 1. Install & Run GROBID (v0.8.2)

### Requirements
- **Java 11 or higher**
- **Maven or Gradle (bundled with repo)**
- **Docker (optional, recommended)**

### Installation (manual build)
```bash
# Download and extract GROBID
wget https://github.com/kermitt2/grobid/archive/0.8.2.zip
unzip 0.8.2.zip
cd grobid-0.8.2

# Build with Gradle
./gradlew clean install
```

### Run GROBID (with Docker)
```bash
# Pull image
docker pull lfoppiano/grobid:0.8.2

# Run container
docker run --rm --init --ulimit core=0 -p 8070:8070 lfoppiano/grobid:0.8.2
```

The service will be available at:  
👉 `http://localhost:8070`  

You can test it by opening the [Swagger UI](http://localhost:8070/swagger-ui.html).

---

## 🔹 2. Install & Run pdffigures2

### Installation
```bash
# Clone the repo
git clone https://github.com/allenai/pdffigures2.git
cd pdffigures2

# Build using SBT
sbt assembly
```

The built JAR will be located in:  
`target/scala-2.11/pdffigures2-assembly-0.0.12.jar` (version may vary).

### Run pdffigures2
```bash
# Extract figures and tables from a PDF
java -jar target/scala-2.11/pdffigures2-assembly-0.0.12.jar input.pdf -g output.json -d output_dir
```

- `input.pdf` → your PDF file  
- `output.json` → JSON file with figure metadata (captions, locations, etc.)  
- `output_dir` → directory where extracted images will be saved  

---

## 🔹 3. Helper Scripts

### `extract_images.py`
Wrapper for `pdffigures2` that extracts images and metadata.

```bash
python extract_images.py input.pdf output_dir
```

- `input.pdf` → path to your PDF  
- `output_dir` → directory where figures and metadata will be stored  

If `pdffigures2` is not in your PATH, update the script to point to your JAR:
```python
command = [
    "java", "-jar", "path/to/pdffigures2-assembly.jar",
    pdf_path,
    "-d", output_dir + os.path.sep
]
```

---

### `xml_parser.py`
Parses a GROBID TEI XML output into structured JSON.

```bash
python xml_parser.py your_paper.tei.xml
```

Example JSON output:
```json
{
  "doc_id": "test",
  "title": "Example Title",
  "abstract": "This is the abstract text.",
  "sections": [
    {
      "heading": "1 Introduction",
      "paragraphs": ["Text of the introduction..."]
    }
  ],
  "figures": [
    {
      "id": "Fig1",
      "caption": "An example figure caption."
    }
  ]
}
```

---

## 🔹 4. Example Workflow

```bash
# Run GROBID for structured metadata extraction
curl -s -X POST   -F "input=@/home/jbaghiro/presentune-dataset/datasets/SMAR/test.pdf"   -F "consolidateCitations=1"   http://localhost:8070/api/processFulltextDocument   -o your_paper.tei.xml

# Parse GROBID TEI output into JSON
python xml_parser.py your_paper.tei.xml > parsed.json

# Run pdffigures2 (via helper script)
python extract_images.py /home/jbaghiro/presentune-dataset/datasets/SMAR/test.pdf figures_out/
```

Now you have:
- **GROBID output** → structured text (title, authors, sections, references, etc.) in TEI XML / JSON  
- **pdffigures2 output** → extracted figures/tables as images and JSON metadata  

---

## 🔹 5. Troubleshooting
- If GROBID fails to start → check Java version (`java -version`) is **11+**.  
- If pdffigures2 build fails → ensure you have correct **Scala 2.11.x or 2.12.x** installed.  
- If `extract_images.py` cannot find `pdffigures2` → update script to point to your JAR.  
- For faster setup, prefer using **Docker** for GROBID.  

---
