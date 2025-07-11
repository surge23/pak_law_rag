import os
import re
import json
import requests
from io import BytesIO
import pdfplumber
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define official government PDFs
PDF_SOURCES = [
    ("constitution_clean.json", "https://www.pakp.gov.pk/wp-content/uploads/2024/07/Constitution.pdf", "Constitution"),
    ("ppc_clean.json", "https://www.fmu.gov.pk/docs/laws/Pakistan%20Penal%20Code.pdf", "PPC"),
    ("crpc_clean.json", "https://www.fmu.gov.pk/docs/laws/Code_of_criminal_procedure_1898.pdf", "CrPC"),
]

def parse_pdf_sections(pdf_bytes, source):
    docs = []
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    if source == "Constitution":
        # Match: Article 8 — Text
        matches = re.split(r"\n*Article\s+(\d+[A-Za-z]*)\s*", text)
        for i in range(1, len(matches), 2):
            number = matches[i].strip()
            body = matches[i + 1].strip()
            docs.append(Document(
                page_content=f"Article {number}\n{body}",
                metadata={"title": f"Article {number}", "source": source}
            ))
    else:
        # Match: Section 379 — Title and content
        matches = re.split(r"\n*Section\s+(\d+[A-Za-z]*)\.\s*(.+?)(?=\nSection\s+\d+|\Z)", text, flags=re.DOTALL)
        for i in range(1, len(matches), 3):
            number = matches[i].strip()
            title = matches[i + 1].strip()
            body = matches[i + 2].strip()
            docs.append(Document(
                page_content=f"Section {number} - {title}\n{body}",
                metadata={"title": f"Section {number}", "source": source}
            ))
    return docs

def main():
    all_docs = []
    os.makedirs("../data", exist_ok=True)

    for fname, url, source in PDF_SOURCES:
        print(f"📥 Downloading {source} from {url}")
        headers = {"User-Agent": "Mozilla/5.0"}  # bypass 403 error
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()

        docs = parse_pdf_sections(resp.content, source)
        print(f"✅ Parsed {len(docs)} sections/articles from {source}")

        # Save original docs to JSON
        with open(f"../data/{fname}", "w", encoding="utf-8") as f:
            json.dump(
                [{"title": d.metadata["title"], "source": d.metadata["source"], "text": d.page_content} for d in docs],
                f,
                indent=2,
                ensure_ascii=False
            )

        all_docs += docs

    # 🔥 CHUNK LONG DOCS
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    split_docs = splitter.split_documents(all_docs)
    print(f"✂️ Split into {len(split_docs)} smaller chunks for safe embedding")

    # Embed and store
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embed_model)
    vectorstore.save_local("../faiss_index/pak_law_combined")
    print("✅ Done! FAISS index saved at: faiss_index/pak_law_combined")

if __name__ == "__main__":
    main()
