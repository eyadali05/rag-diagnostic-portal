#!/usr/bin/env python3
"""NSW documentation ingestion.

Changes from the previous version:
  * ABSOLUTE PATHS. DOCS_DIR and CHROMA_DIR were relative to the current working directory,
    so running this from anywhere other than the project folder silently built a second
    vector store that model.py (which resolves against __file__) never opened. Same class of
    bug database.py already fixed for the sqlite file.
  * PINNED COLLECTION NAME, shared with model.py via NSW_CHROMA_COLLECTION. Previously this
    relied on both sides independently defaulting to "langchain".
  * HARDWARE METADATA. Each chunk is tagged with the pump and platform names found in its
    source filename and its text. model.retrieve_relevant() can then filter to the machine
    being diagnosed, which is the single biggest accuracy win available here - it stops
    a PistonPRO chunk from being offered as evidence about a SynchroPULSE.
  * SMALLER CHUNKS. 2500 chars diluted a qwen3-embedding:0.6b vector across two or three
    unrelated procedures. 1000/150 keeps one idea per vector; model.py compensates by
    retrieving more of them.
  * The separators list no longer contains "\\n## " / "\\n### ", which never appear in
    PyPDF output and therefore never fired.
  * --reset drops the collection first, so a re-ingest doesn't silently duplicate everything.

Usage:
    python ingest.py
    python ingest.py --reset
    NSW_DOCS_DIR=/srv/manuals python ingest.py
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

#from langchain_community.document_loaders import PyPDFDirectoryLoader
#from langchain_docling import DoclingLoader
import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = Path(os.environ.get("NSW_DOCS_DIR", BASE_DIR.parent / "documents" / "RAG" / "Master"))
CHROMA_DIR = Path(os.environ.get("NSW_CHROMA_DIR", BASE_DIR / "chroma_db"))
COLLECTION_NAME = os.environ.get("NSW_CHROMA_COLLECTION", "langchain")
EMBEDDING_MODEL = os.environ.get("NSW_EMBEDDING_MODEL", "qwen3-embedding:0.6b")

CHUNK_SIZE = int(os.environ.get("NSW_CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.environ.get("NSW_CHUNK_OVERLAP", 150))

# Single source of truth for hardware names + aliases, shared with model.py so that what is
# TAGGED here is exactly what is QUERIED there. A chunk is tagged with every name it mentions.
from model import PLATFORM_ALIASES, PUMP_ALIASES, hardware_tags

PUMP_NAMES = list(PUMP_ALIASES)
PLATFORM_NAMES = list(PLATFORM_ALIASES)
MIN_CHARS_PER_PAGE = 200   # below this a PDF is probably scanned/image-only and needs OCR


def _norm(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _tags(names, haystack_norm: str) -> str:
    """Comma-joined list of the names that appear in the text.

    Chroma metadata values must be scalars, so this is a string, not a list. Matching is
    done on an alphanumeric-only projection because "Time Pressure System", "time-pressure
    system" and "TimePressure System" are the same thing to an operator and to a manual.
    """
    hits = [n for n in names if _norm(n) in haystack_norm]
    return ", ".join(hits)


def run_ingestion(reset: bool = False):
    if not DOCS_DIR.exists():
        print(f"[ERROR] Directory '{DOCS_DIR}' not found. Place your PDFs there, "
              f"or set NSW_DOCS_DIR.")
        sys.exit(1)

    pdf_files = sorted(p.name for p in DOCS_DIR.iterdir() if p.suffix.lower() == ".pdf")
    if not pdf_files:
        print(f"[ERROR] No PDF files found in '{DOCS_DIR}'.")
        sys.exit(1)

    if reset and CHROMA_DIR.exists():
        print(f"[*] --reset: removing existing store at '{CHROMA_DIR}'")
        shutil.rmtree(CHROMA_DIR)

    print(f"[*] Source folder : {DOCS_DIR}")
    print(f"[*] Target store  : {CHROMA_DIR}  (collection '{COLLECTION_NAME}')")
    print(f"[*] Found {len(pdf_files)} PDF documents: {pdf_files}")


    print(f"[*] Parsing documents (this may take a moment for layout analysis)...")
    pdf_paths = [str(DOCS_DIR / f) for f in pdf_files]
    
    raw_documents = []
    coverage = []

    for file in pdf_files:
        pdf_path = str(DOCS_DIR / file)
        try:
            # Extract PDF directly into Markdown format, separated by page
            md_pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
            chars = 0
            for idx, page_data in enumerate(md_pages):
                # BUG FIX: depending on the pymupdf4llm version the 1-based page number is in
                # metadata["page"] or metadata["page_number"]. Reading only "page_number"
                # with a default of 1 put EVERY chunk on page 1 for older versions, so every
                # citation pointed at page 1.
                meta = page_data.get("metadata", {}) or {}
                page_no = meta.get("page_number") or meta.get("page") or (idx + 1)
                page_num = int(page_no) - 1
                text = page_data.get("text", "") or ""
                chars += len(text.strip())
                if not text.strip():
                    continue          # image-only page: nothing to embed

                doc = Document(
                    page_content=page_data.get("text", ""),
                    metadata={
                        "source": pdf_path,
                        "doc_name": file,
                        "page": page_num
                    }
                )
                raw_documents.append(doc)
            coverage.append((file, len(md_pages), chars))

        except Exception as e:
            print(f"[!] Failed to parse {file}: {e}")
            coverage.append((file, 0, 0))

    # This is the check for "did my vector database get ALL of my documents?" Brochures are
    # often image-heavy: if a PDF yields almost no text it contributes almost nothing to the
    # store and the model can never cite it.
    print("[*] Text extracted per document:")
    for file, pages, chars in coverage:
        per_page = chars / pages if pages else 0
        flag = "" if per_page >= MIN_CHARS_PER_PAGE else \
            "   <-- LOW TEXT: scanned/image PDF? run OCR (e.g. ocrmypdf) and re-ingest"
        print(f"      {file:60s} pages={pages:4d} chars={chars:8d}{flag}")
    print(f"[*] Extracted {len(raw_documents)} raw pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = [c for c in splitter.split_documents(raw_documents)
              if len(c.page_content.strip()) > 30]   # drop empty/stub fragments
    print(f"[*] Generated {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    tagged_pumps = tagged_platforms = 0
    for doc in chunks:
        src = str((doc.metadata or {}).get("source", ""))
        # The filename matters as much as the body: a chunk from "PistonPRO_manual.pdf" is
        # about the PistonPRO even on a page that never names it.
        haystack = os.path.basename(src) + " " + doc.page_content
        # Aliases + word boundaries: "PistonPRO" alone now counts, and "N4" no longer
        # matches inside unrelated words the way the alnum-only projection did.
        pumps = ", ".join(hardware_tags(PUMP_ALIASES, haystack))
        platforms = ", ".join(hardware_tags(PLATFORM_ALIASES, haystack))
        doc.metadata["pumps"] = pumps
        doc.metadata["platforms"] = platforms
        doc.metadata["doc_name"] = os.path.basename(src)
        tagged_pumps += bool(pumps)
        tagged_platforms += bool(platforms)

    print(f"[*] Tagged {tagged_pumps}/{len(chunks)} chunks with a pump, "
          f"{tagged_platforms}/{len(chunks)} with a platform.")
    if tagged_pumps == 0:
        print("[WARN] No chunk mentions any known pump. Check that PUMP_NAMES matches the "
              "wording in your manuals, or hardware filtering will find nothing.")

    print(f"[*] Embedding with '{EMBEDDING_MODEL}'...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )
    count = vector_store._collection.count()
    print(f"[SUCCESS] Vector store holds {count} records in '{CHROMA_DIR}'.")
    print("[*] Next: run the app, then open the sidebar's 'Retrieval diagnostics' expander "
          "to confirm the distance metric and cutoff.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true",
                    help="delete the existing store before ingesting (avoids duplicates)")
    run_ingestion(**vars(ap.parse_args()))
