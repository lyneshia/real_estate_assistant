import streamlit as st
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from uuid import uuid4
from langchain_classic.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
llm = None
vectordb = None
qa_chain = None

COLLECTION_NAME = "real_estate"
VECTOR_DB_DIR = str(Path(__file__).parent / "resources" / "vectordb")
MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 1000

def initialize_components():
    global llm, vectordb
    if llm is None:
        api_key = st.secrets["GROQ_API_KEY"]
        llm = ChatGroq(
            model = "openai/gpt-oss-120b",
            temperature=0.3,
            max_tokens=1024,
            api_key = api_key
        )

    if vectordb is None:
        vectordb = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=VECTOR_DB_DIR,
        )

def process_urls(urls):
    yield "Initializing the components..."
    initialize_components()

    yield "Resetting the Vector database..."
    vectordb.reset_collection()

    yield "Extracting the data from the URLs..."

    data = []
    for u in urls:
        try:
            loader = UnstructuredLoader(web_url=u)
            loaded_docs = loader.load()

            print(f"URL: {u}")
            print(f"Documents loaded: {len(loaded_docs)}")

            for doc in loaded_docs[:3]:
                print(
                    "Content length:",
                    len(doc.page_content)
                )

            data.extend(loaded_docs)

        except Exception as e:
            yield f"Failed to load {u}: {e}"
    if not data:
    raise ValueError(
        "No content could be extracted from the provided URLs."
    )


    yield "Splitting the data into chunks..."

    text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", " "],
    chunk_size=CHUNK_SIZE,
    chunk_overlap=150
    )

    docs = text_splitter.split_documents(data)


    if not docs:
        raise ValueError(
            "No text chunks were created from the extracted content."
        )


    print("Total source documents:", len(data))
    print("Total chunks:", len(docs))


    yield "Adding the chunks to the Vector database..."

    uuids = [
        str(uuid4())
        for _ in range(len(docs))
        ]

    vectordb.add_documents(
        documents=docs,
        ids=uuids
    )

    # Add 'source' metadata for RetrievalQAWithSourcesChain
    for doc in docs:
        if "source" not in doc.metadata and "url" in doc.metadata:
            doc.metadata["source"] = doc.metadata["url"]

    yield "Adding the chunks to the Vector database..."
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vectordb.add_documents(
        documents=docs,
        ids=uuids
    )
    yield "Generating the answers..."

def generate_answer(query):
    global qa_chain

    if not vectordb:
        raise RuntimeError("Vector database not initialized")

    if qa_chain is None:
        qa_chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectordb.as_retriever()
        )

    result = qa_chain.invoke({"question": query}, return_only_outputs=True)
    sources = result.get("sources", "")

    return result["answer"], sources

