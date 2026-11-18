import sys

# --- Streamlit Cloud ships an old sqlite3; Chroma needs >= 3.35 ----------------
# Requires `pysqlite3-binary` in requirements.txt. Must run before chroma imports.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import tempfile
from pathlib import Path
from uuid import uuid4

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
os.environ.setdefault("USER_AGENT", USER_AGENT)

COLLECTION_NAME = "real_estate"
# The repo directory can be read-only / wiped on reboot, so use a temp dir.
VECTOR_DB_DIR = str(Path(tempfile.gettempdir()) / "vectordb")
MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


@st.cache_resource(show_spinner=False)
def get_components():
    """Create the LLM, embeddings and vector DB once and reuse them across reruns."""
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.3,
        api_key=st.secrets["GROQ_API_KEY"],
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_DIR,
    )
    return llm, vectordb


def process_urls(urls):
    yield "Initializing the components..."
    llm, vectordb = get_components()

    yield "Resetting the Vector database..."
    vectordb.reset_collection()

    yield "Extracting the data from the URLs..."
    data = []
    for u in urls:
        try:
            loader = WebBaseLoader(web_paths=[u], header_template={"User-Agent": USER_AGENT})
            loaded_docs = loader.load()

            # Drop empty pages and tag every doc with its source URL
            loaded_docs = [d for d in loaded_docs if d.page_content.strip()]
            for doc in loaded_docs:
                doc.metadata["source"] = u

            data.extend(loaded_docs)
            yield f"Loaded {len(loaded_docs)} page(s) from {u}"
        except Exception as e:
            yield f"Failed to load {u}: {e}"

    if not data:
        raise ValueError("No content could be extracted from the provided URLs.")

    yield "Splitting the data into chunks..."
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = text_splitter.split_documents(data)

    if not docs:
        raise ValueError("No text chunks were created from the extracted content.")

    for doc in docs:
        if "source" not in doc.metadata:
            doc.metadata["source"] = "Unknown"

    yield "Adding the chunks to the Vector database..."
    uuids = [str(uuid4()) for _ in docs]
    vectordb.add_documents(documents=docs, ids=uuids)

    count = vectordb._collection.count()
    if count == 0:
        raise RuntimeError("Indexing finished but the vector database is empty.")
    yield f"Stored {count} chunks. Ready to answer questions."


PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a real estate research assistant. Answer the question using ONLY "
        "the context below, which was extracted from web articles. If the answer is "
        "not in the context, say you could not find it in the provided articles. "
        "Do not invent facts.\n\nContext:\n{context}",
    ),
    ("human", "{question}"),
])


def generate_answer(query):
    llm, vectordb = get_components()

    if vectordb._collection.count() == 0:
        raise RuntimeError("No documents are indexed yet. Please process a URL first.")

    docs = vectordb.as_retriever(search_kwargs={"k": 5}).invoke(query)
    if not docs:
        return "I couldn't find anything relevant in the indexed articles.", ""

    context = "\n\n".join(f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs))
    response = llm.invoke(PROMPT.format_messages(context=context, question=query))

    sources = "\n".join(dict.fromkeys(d.metadata.get("source", "Unknown") for d in docs))
    return response.content, sources
