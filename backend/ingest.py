"""Load html from files, clean up, split, ingest into Weaviate."""
import logging
import os
import re

from enum import Enum
from dotenv import load_dotenv
from parser_1 import langchain_docs_extractor

import weaviate
import chromadb
from chromadb.config import Settings
from bs4 import BeautifulSoup, SoupStrainer
from langchain.docstore.document import Document
from langchain.indexes import SQLRecordManager, index
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import Weaviate
from langchain_community.document_loaders import RecursiveUrlLoader, SitemapLoader

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser

from typing import List
from constants import WEAVIATE_DOCS_INDEX_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SourceDocumentFlag(Enum):
    WEB = 'web'
    LOCAL = 'local'


if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)


def get_openai_embeddings() -> Embeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=200)


def get_huggingface_embeddings() -> Embeddings:
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def metadata_extractor(meta: dict, soup: BeautifulSoup) -> dict:
    title = soup.find("title")
    description = soup.find("meta", attrs={"name": "description"})
    html = soup.find("html")
    return {
        "source": meta["loc"],
        "title": title.get_text() if title else "",
        "description": description.get("content", "") if description else "",
        "language": html.get("lang", "") if html else "",
        **meta,
    }


def load_langchain_docs():
    return SitemapLoader(
        "https://python.langchain.com/sitemap.xml",
        filter_urls=["https://python.langchain.com/"],
        parsing_function=langchain_docs_extractor,
        default_parser="lxml",
        bs_kwargs={
            "parse_only": SoupStrainer(
                name=("article", "title", "html", "lang", "content")
            ),
        },
        meta_function=metadata_extractor,
    ).load()


def load_langsmith_docs():
    return RecursiveUrlLoader(
        url="https://docs.smith.langchain.com/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
    ).load()


def simple_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def load_api_docs():
    return RecursiveUrlLoader(
        url="https://api.python.langchain.com/en/latest/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
        exclude_dirs=(
            "https://api.python.langchain.com/en/latest/_sources",
            "https://api.python.langchain.com/en/latest/_modules",
        ),
    ).load()


def load_web_page(target_url:str):
    return RecursiveUrlLoader(
        url=target_url,
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
    ).load()


def load_directory(dir_path: str) -> List[Document]:
    return GenericLoader.from_filesystem(
        dir_path,
        glob="**/[!.]*",
        suffixes=[".kt", ".gradle", ".xml"],
        parser=LanguageParser(),
    ).load()


def does_vectorstore_exist(embeddings: HuggingFaceEmbeddings) -> bool:
    """
    Checks if vectorstore exists
    """
    PERSIST_DIRECTORY = os.environ["PERSIST_DIRECTORY"]
    db = Chroma(persist_directory=PERSIST_DIRECTORY, 
                embedding_function=embeddings)
    if not db.get()['documents']:
        return False
    return True


def process_documents(documents: List[Document], ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"persistence. total docs: {len(documents)}")
    
    docs = [doc for doc in documents if doc.metadata['source'] not in ignored_files]
    print(f"persistence. docs to be saved: {len(docs)}")

    if not docs:
        print("No new documents to load")
        return docs
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    documents = text_splitter.split_documents(documents)
    print(f"Split into {len(documents)} chunks of text (max. 4000 tokens each)")

    return documents


def batch_chromadb_insertions(chroma_client, documents: List[Document]) -> List[Document]:
    """
    Split the total documents to be inserted into batches of documents that the local chroma client can process
    """
    # Get max batch size.
    max_batch_size = chroma_client.max_batch_size
    for i in range(0, len(documents), max_batch_size):
        yield documents[i:i + max_batch_size]


def ingest_docs_chroma(flag):
    LOAD_WEB_URL = os.environ["LOAD_WEB_URL"]
    LOAD_DIR_PATH = os.environ["LOAD_DIR_PATH"]
    PERSIST_DIRECTORY = os.environ["PERSIST_DIRECTORY"]

    texts = None

    if flag == SourceDocumentFlag.WEB:
        print(f"Fetching documents from web url: {LOAD_WEB_URL}")
        # load web page
        texts = load_web_page(LOAD_WEB_URL)
        logger.info(f"Loaded {len(texts)} docs from devies.se")
    elif flag == SourceDocumentFlag.LOCAL:
        print(f"Fetching documents from local directory: {LOAD_DIR_PATH}")
        # load local docs
        texts = load_directory(LOAD_DIR_PATH)
        logger.info(f"Loaded {len(texts)} docs from project folder")
    else:
        print(f"Invalid flag. Please provide '{SourceDocumentFlag.WEB.value}' or '{SourceDocumentFlag.LOCAL.value}'.")

    embeddings = get_openai_embeddings()
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

    CHROMA_SETTINGS = Settings(
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
    )

    if does_vectorstore_exist(embeddings):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at localhost")
        db = Chroma(persist_directory=PERSIST_DIRECTORY,
                    client_settings=CHROMA_SETTINGS,
                    embedding_function=embeddings,
                    client=chroma_client)
        print(f"Creating embeddings. May take some minutes...")
        collection = db.get()
        documents = process_documents(texts, [metadata['source'] for metadata in collection['metadatas']])

        if documents:
            print(f"Creating embeddings. May take some minutes...")
            for batched_chromadb_insertion in batch_chromadb_insertions(chroma_client, documents):
                db.add_documents(batched_chromadb_insertion)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        documents = process_documents(texts)
        print(f"Creating embeddings. May take some minutes...")
        batched_chromadb_insertions = batch_chromadb_insertions(chroma_client, documents)
        first_insertion = next(batched_chromadb_insertions)
        db = Chroma.from_documents(documents=first_insertion, 
                                   embedding=embeddings,
                                   persist_directory=PERSIST_DIRECTORY,
                                   client_settings=CHROMA_SETTINGS,
                                   client=chroma_client)
        # Add the rest of batches of documents
        for batched_chromadb_insertion in batched_chromadb_insertions:
            db.add_documents(batched_chromadb_insertion)

    print(f"Ingestion complete! You can now run GPT to query your documents")


def ingest_docs_weaviate():
    WEAVIATE_URL = os.environ["WEAVIATE_URL"]
    WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
    RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    embedding = get_openai_embeddings()

    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    )
    vectorstore = Weaviate(
        client=client,
        index_name=WEAVIATE_DOCS_INDEX_NAME,
        text_key="text",
        embedding=embedding,
        by_text=False,
        attributes=["source", "title"],
    )

    record_manager = SQLRecordManager(
        f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}", db_url=RECORD_MANAGER_DB_URL
    )
    record_manager.create_schema()

    docs_from_documentation = load_langchain_docs()
    logger.info(f"Loaded {len(docs_from_documentation)} docs from documentation")
    docs_from_api = load_api_docs()
    logger.info(f"Loaded {len(docs_from_api)} docs from API")
    docs_from_langsmith = load_langsmith_docs()
    logger.info(f"Loaded {len(docs_from_langsmith)} docs from Langsmith")

    docs_transformed = text_splitter.split_documents(
        docs_from_documentation + docs_from_api + docs_from_langsmith
    )
    docs_transformed = [doc for doc in docs_transformed if len(doc.page_content) > 10]

    # We try to return 'source' and 'title' metadata when querying vector store and
    # Weaviate will error at query time if one of the attributes is missing from a
    # retrieved document.
    for doc in docs_transformed:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""

    indexing_stats = index(
        docs_transformed,
        record_manager,
        vectorstore,
        cleanup="full",
        source_id_key="source",
        force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
    )

    logger.info(f"Indexing stats: {indexing_stats}")
    num_vecs = client.query.aggregate(WEAVIATE_DOCS_INDEX_NAME).with_meta_count().do()
    logger.info(
        f"LangChain now has this many vectors: {num_vecs}",
    )


if __name__ == "__main__":
    print("Starting to ingest documents for contextual reasoning!")
    print("Do you want to fetch documents from the web or from a local directory?")

    while True:
        choice = input(f"Enter '{SourceDocumentFlag.WEB.value}' or '{SourceDocumentFlag.LOCAL.value}' (or 'exit' to quit): ")
        
        if choice == 'exit':
            print("Exiting program.")
            break
        
        if choice in [flag.value for flag in SourceDocumentFlag]:
            ingest_docs_chroma(SourceDocumentFlag(choice))
            break
        else:
            print(f"Invalid choice. Please enter '{SourceDocumentFlag.WEB.value}' or '{SourceDocumentFlag.LOCAL.value}'.")

