"""Load html from files, clean up, split, ingest into Weaviate."""
import logging
import os

from langchain_elasticsearch import DenseVectorStrategy
from parser import langchain_docs_extractor
from dotenv import load_dotenv

from bs4 import BeautifulSoup, SoupStrainer
from constants import WEAVIATE_DOCS_INDEX_NAME
from langchain_community.document_loaders import RecursiveUrlLoader, SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain_community.vectorstores import ElasticsearchStore
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()


ELASTIC_DB_USER = os.environ["ELASTIC_DB_USER"]
ELASTIC_DB_PASS = os.environ["ELASTIC_DB_PASS"]
ELASTIC_DB_API_KEY = os.environ["ELASTIC_DB_API_KEY"]
ELASTIC_DB_URL = os.environ["ELASTIC_DB_URL"]
ELASTIC_DB_INDEX_NAME = os.environ["ELASTIC_DB_INDEX_NAME"]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_embeddings_model() -> Embeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=200)


def get_data_vector_store(embeddings: Embeddings):
    return ElasticsearchStore(
        ELASTIC_DB_INDEX_NAME,
        embedding=embeddings,
        es_user=ELASTIC_DB_USER,
        es_password=ELASTIC_DB_PASS,
        es_api_key=ELASTIC_DB_API_KEY,
        es_url=ELASTIC_DB_URL)


def delete_entire_index(vector_store):
    try:
        # Delete the entire index
        vector_store.client.indices.delete(
            index=vector_store.index_name,
            ignore_unavailable=True
        )
        print(f"Index {vector_store.index_name} deleted successfully")
    except Exception as e:
        print(f"Error deleting index: {e}")


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


def load_webpage_docs():
    return SitemapLoader(
        "https://www.devies.se/sitemap.xml",
        filter_urls=["https://www.devies.se/"],
        parsing_function=langchain_docs_extractor,
        default_parser="lxml",
        bs_kwargs={
            "parse_only": SoupStrainer(
                name=("article", "title", "html", "lang", "content")
            ),
        },
        meta_function=metadata_extractor,
    ).load()


def ingest_docs():
    embedding = get_embeddings_model()

    # create ElasticSearch instance for data
    data_vector_store = get_data_vector_store(embedding)

    docs_from_api = load_webpage_docs()
    logger.info(f"Loaded {len(docs_from_api)} docs from sitemap")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    docs_transformed = text_splitter.split_documents(docs_from_api)
    docs_transformed = [doc for doc in docs_transformed if len(doc.page_content) > 10]

    # clear index
    delete_entire_index(data_vector_store)

    # create new and populate with new docs
    data_vector_store = get_data_vector_store(embeddings=embedding)
    data_vector_store.add_documents(documents=docs_transformed)

    # We try to return 'source' and 'title' metadata when querying vector store and
    # Weaviate will error at query time if one of the attributes is missing from a
    # retrieved document.
    for doc in docs_transformed:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""


if __name__ == "__main__":
    ingest_docs()
